#!/usr/bin/env python3
"""
Timelapse Creator Script - OS Agnostic Version with Parallel Processing
Creates timelapse videos directly from timestamped images via ffmpeg concat
demuxer, bypassing the sequential-rename step entirely.

Features:
- Manifest-driven file ordering (no disk scan / regex parsing needed)
- ffmpeg concat demuxer eliminates sequential file copies
- Parallel timestamp loading for large datasets
- Multiple video preset generation
- Parallel chunked encoding: split the ordered list into contiguous chunks,
  encode them concurrently to MPEG-TS intermediates, stitch losslessly
- Pluggable encoder: x264 (default), NVENC h264/hevc, or auto (runtime-probed)
- Optional downscale (--scale) applied early to lighten every later stage
- Built-in --benchmark mode (decode-ceiling vs full-encode verdict)
- Timezone-offset flag for local-time display in logs
- Comprehensive logging and error handling
- Cross-platform compatibility (Windows 11 primary target)

The framerate is set as an INPUT option (`-r F` before `-i`). With the concat
demuxer feeding still images this is required: passing `-framerate`/`-r` after
`-i` is ignored (output inherits the demuxer's 25 fps default) or drops frames.
See BENCHMARK_REPORT.md (Phase 1, section A) for the measurements behind this.

Usage:
    python timelapse_creator.py [--source_dir PATH] [--output_dir PATH]
                                [--manifest PATH] [--tz_offset HOURS]
                                [--dry_run] [--video_only]
                                [--presets PRESET [PRESET ...]]
                                [--encoder {x264,nvenc,hevc_nvenc,auto}]
                                [--workers {N,auto}] [--scale WxH]
                                [--trust-manifest] [--benchmark]
"""

import os
import re
import math
import time
import shutil
import argparse
import subprocess
import logging
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class TimelapseCreator:
    """Orchestrates manifest-driven timelapse creation using ffmpeg."""

    def __init__(
        self,
        source_dir: str = None,
        output_dir: str = None,
        manifest_path: str = None,
        backup_originals: bool = True,
        tz_offset_hours: float = -6.0,
        encoder: str = "x264",
        workers: str = "1",
        threads_per_worker: int = None,
        scale: str = None,
        trust_manifest: bool = False,
    ):
        # Resolve paths relative to the script's location
        script_dir = Path(__file__).parent.resolve()

        self.source_dir = Path(source_dir).resolve() if source_dir else script_dir / "captures"
        self.output_dir = Path(output_dir).resolve() if output_dir else script_dir / "processed"
        # Manifest lives in source_dir by default; --manifest overrides if needed
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path
            else self.source_dir / "manifest.json"
        )
        self.backup_originals = backup_originals

        # Timezone offset for display / logging (does not affect sort order)
        self.local_tz = timezone(timedelta(hours=tz_offset_hours))
        tz_sign = "+" if tz_offset_hours >= 0 else ""
        self.tz_label = f"UTC{tz_sign}{tz_offset_hours:g}"

        # Output sub-directories
        self.videos_dir = self.output_dir / "videos"
        self.concat_list_path = self.output_dir / "concat_list.txt"
        # Scratch dir for per-chunk concat lists and intermediate segments
        self.chunks_dir = self.output_dir / "chunks"

        # Encode options
        self.encoder = encoder
        self.workers_arg = str(workers)
        self.threads_per_worker = threads_per_worker
        self.scale = scale
        self.trust_manifest = trust_manifest

        # Physical / logical core detection (cached) for worker auto-sizing
        self.logical_cores = os.cpu_count() or 1
        self.physical_cores = self._detect_physical_cores()
        # Cache of encoder name -> bool (does it initialize at runtime?)
        self._encoder_probe_cache: Dict[str, bool] = {}
        # Resolved at encode time (auto -> concrete working encoder)
        self.resolved_encoder: Optional[str] = None

        # Configuration
        self.config = {
            "log_level": "INFO",
            "image_extensions": {".jpg", ".jpeg", ".png", ".tiff", ".bmp"},
            "max_workers": min(16, (os.cpu_count() or 1) * 2),
            # Manifest resolution is now always threaded (real ~9,400-file
            # workloads previously ran sequentially under the 10,000 default).
            "parallel_threshold": 0,
            "ffmpeg_presets": {
                "preview": {"framerate": 60, "crf": 23, "suffix": "preview"},
                "standard": {"framerate": 30, "crf": 17, "suffix": "standard"},
                "high_quality": {"framerate": 24, "crf": 15, "suffix": "hq"},
                "smooth": {"framerate": 45, "crf": 20, "suffix": "smooth"},
                "construction": {"framerate": 20, "crf": 10, "suffix": "construction"},
            },
            # ----------------------------------------------------------------
            # Encoder definitions. x264 is the default and is bit-for-bit
            # comparable to prior runs (same codec/CRF/pix_fmt); the framerate
            # is the only corrected behavior. GPU encoders are strictly opt-in.
            #
            # NVENC ignores -crf, so each preset's CRF is mapped 1:1 to NVENC
            # -cq (same 0-51 scale, lower = higher quality) under VBR with no
            # bitrate cap (-b:v 0) -- the closest analog to x264's CRF. NVENC is
            # less efficient per bit than x264, so at a given number it produces
            # lower fidelity / more bits than x264; this is most visible at the
            # low-CRF presets (construction 10, high_quality 15). Raise --cq via
            # the table below for smaller files. Mapping lives ONLY here.
            # ----------------------------------------------------------------
            "encoders": {
                "x264": {
                    "kind": "cpu",
                    "codec": "libx264",
                    "hevc": False,
                },
                "nvenc": {
                    "kind": "gpu",
                    "codec": "h264_nvenc",
                    "hevc": False,
                },
                "hevc_nvenc": {
                    "kind": "gpu",
                    "codec": "hevc_nvenc",
                    "hevc": True,
                },
            },
            # CRF -> NVENC -cq, per preset (documented above). Defaults to the
            # preset CRF (1:1) but kept explicit so it is auditable / tunable.
            "nvenc_cq": {
                "preview": 23,
                "standard": 17,
                "high_quality": 15,
                "smooth": 20,
                "construction": 10,
            },
            # NVENC quality/rate-control knobs shared by all presets.
            "nvenc_common": ["-preset", "p5", "-tune", "hq",
                             "-rc", "vbr", "-b:v", "0"],
            # Consumer GeForce concurrent NVENC session cap (driver 551.76+ = 8;
            # was 5, and 3 on older drivers). Never spawn more GPU sessions.
            "nvenc_session_limit": 8,
            # Intermediate container for chunked encoding. MPEG-TS concatenates
            # cleanly under `-c copy` (clean packet boundaries, no moov/edit-
            # list to reconcile across segments) -- see BENCHMARK_REPORT.md C.
            "intermediate_format": "mpegts",
            "intermediate_ext": "ts",
        }

        self.setup_logging()
        self.log_system_info()
        self.create_directories()

        # Populated after load_manifest()
        self.ordered_files: List[Tuple[Path, datetime]] = []

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def log_system_info(self) -> None:
        """Log system and environment information."""
        self.logger.info(f"Operating System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python Version: {platform.python_version()}")
        self.logger.info(f"Script Directory: {Path(__file__).parent.resolve()}")
        self.logger.info(f"Current Working Directory: {Path.cwd()}")
        self.logger.info(f"Source Directory: {self.source_dir}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Manifest Path: {self.manifest_path}")
        self.logger.info(f"Display Timezone: {self.tz_label}")
        self.logger.info(
            f"CPU cores: {self.physical_cores} physical / "
            f"{self.logical_cores} logical"
        )
        self.logger.info(
            f"Encoder request: {self.encoder}  |  Workers request: "
            f"{self.workers_arg}  |  Scale: {self.scale or 'none'}  |  "
            f"Trust manifest: {self.trust_manifest}"
        )

        if platform.system() == "Linux":
            try:
                with open("/proc/version", "r") as fh:
                    if "microsoft" in fh.read().lower():
                        self.logger.info("Detected: Windows Subsystem for Linux (WSL)")
            except OSError:
                pass

    def setup_logging(self) -> None:
        """Configure file + console logging."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_dir / "timelapse_creation.log"

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def create_directories(self) -> None:
        """Create output sub-directories."""
        directories = [self.output_dir, self.videos_dir]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            except Exception as exc:
                self.logger.error(f"Failed to create directory {directory}: {exc}")
                raise

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def load_manifest(self) -> List[Tuple[Path, datetime]]:
        """
        Load file order from manifest.json.

        The manifest is a flat dict of the form:
            { "<filename>": { "timestamp": "<ISO-8601 UTC string>" }, ... }

        Returns a list of (Path, datetime) tuples sorted ascending by timestamp.
        Files referenced in the manifest but absent from source_dir are skipped
        with a warning.
        """
        if not self.manifest_path.exists():
            self.logger.error(f"Manifest not found: {self.manifest_path}")
            return []

        self.logger.info(f"Loading manifest: {self.manifest_path}")

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                manifest: dict = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.error(f"Failed to read manifest: {exc}")
            return []

        self.logger.info(f"Manifest contains {len(manifest)} entries")

        # Resolve each entry. Resolution is threaded whenever the manifest is
        # larger than parallel_threshold (default 0 => always threaded), since
        # the work is dominated by I/O-bound stat() calls. --trust-manifest
        # removes the stat() entirely (see _resolve_entry).
        if len(manifest) > self.config["parallel_threshold"]:
            self.logger.info(
                f"Resolving {len(manifest)} entries with threaded path "
                f"resolution (trust_manifest={self.trust_manifest})"
            )
            results = self._parallel_manifest_resolve(manifest)
        else:
            results = self._sequential_manifest_resolve(manifest)

        # Sort by UTC timestamp (chronological, oldest first)
        results.sort(key=lambda x: x[1])

        self.ordered_files = results
        self.logger.info(f"Resolved {len(results)} files from manifest")

        if results:
            start_local = results[0][1].astimezone(self.local_tz)
            end_local = results[-1][1].astimezone(self.local_tz)
            duration = results[-1][1] - results[0][1]
            self.logger.info(
                f"Time range ({self.tz_label}): "
                f"{start_local.strftime('%Y-%m-%d %H:%M:%S')} → "
                f"{end_local.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.logger.info(f"Duration: {duration}")

        return results

    def _resolve_entry(
        self, filename: str, entry: dict
    ) -> Optional[Tuple[Path, datetime]]:
        """Resolve a single manifest entry to (Path, datetime) or None."""
        raw_ts = entry.get("timestamp", "")
        if not raw_ts:
            self.logger.warning(f"Missing 'timestamp' in manifest entry for: {filename}")
            return None

        try:
            # Handles both trailing 'Z' and explicit '+00:00'
            dt_utc = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError as exc:
            self.logger.warning(f"Unparseable timestamp '{raw_ts}' for {filename}: {exc}")
            return None

        file_path = self.source_dir / filename
        # --trust-manifest skips the per-file stat(): the manifest is generated
        # alongside the captures and is treated as authoritative. On ~9,400
        # files over a network/again-cold FS this removes 9,400 stat() calls.
        if not self.trust_manifest and not file_path.exists():
            self.logger.warning(f"File in manifest not found on disk: {file_path}")
            return None

        return file_path, dt_utc

    def _sequential_manifest_resolve(
        self, manifest: dict
    ) -> List[Tuple[Path, datetime]]:
        results = []
        for filename, entry in manifest.items():
            resolved = self._resolve_entry(filename, entry)
            if resolved:
                results.append(resolved)
        return results

    def _parallel_manifest_resolve(
        self, manifest: dict
    ) -> List[Tuple[Path, datetime]]:
        results = []
        max_workers = min(self.config["max_workers"], len(manifest))
        self.logger.info(f"Using {max_workers} worker threads for manifest resolution")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._resolve_entry, fn, entry): fn
                for fn, entry in manifest.items()
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                if completed % 1000 == 0:
                    self.logger.info(
                        f"Resolved {completed}/{len(manifest)} manifest entries"
                    )

        return results

    # ------------------------------------------------------------------
    # Concat list
    # ------------------------------------------------------------------

    def write_concat_list(
        self, ordered_files: List[Tuple[Path, datetime]], dry_run: bool = False
    ) -> bool:
        """
        Write an ffmpeg concat-demuxer list pointing at source files in order.

        Each line is:
            file '<absolute_path>'

        This replaces the sequential-rename step entirely — ffmpeg reads the
        source files directly in the order specified by the list.
        """
        if not ordered_files:
            self.logger.error("No files to write concat list for")
            return False

        self.logger.info(
            f"Writing concat list ({len(ordered_files)} entries): "
            f"{self.concat_list_path}"
        )

        if dry_run:
            self.logger.info("DRY RUN: Skipping concat list write")
            return True

        try:
            with open(self.concat_list_path, "w", encoding="utf-8") as fh:
                for file_path, _ in ordered_files:
                    # ffmpeg requires forward slashes even on Windows
                    safe_path = str(file_path.resolve()).replace("\\", "/")
                    fh.write(f"file '{safe_path}'\n")
            self.logger.info("Concat list written successfully")
            return True
        except OSError as exc:
            self.logger.error(f"Failed to write concat list: {exc}")
            return False

    # ------------------------------------------------------------------
    # ffmpeg helpers
    # ------------------------------------------------------------------

    def check_ffmpeg(self) -> bool:
        """Return True if ffmpeg is available on PATH."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
            self.logger.info("ffmpeg is available")
            return True
        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg check timed out")
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"ffmpeg command failed: {exc}")
        except FileNotFoundError:
            self.logger.error("ffmpeg not found. Please install ffmpeg:")
            if platform.system() == "Windows":
                self.logger.error(
                    "  Windows: winget install ffmpeg  "
                    "or https://ffmpeg.org/download.html"
                )
            elif platform.system() == "Darwin":
                self.logger.error("  macOS: brew install ffmpeg")
            else:
                self.logger.error(
                    "  Linux/WSL: sudo apt install ffmpeg  "
                    "(or equivalent for your distro)"
                )
        except Exception as exc:
            self.logger.error(f"Unexpected error checking ffmpeg: {exc}")
        return False

    # ------------------------------------------------------------------
    # Hardware capability & encoder resolution
    # ------------------------------------------------------------------

    def _detect_physical_cores(self) -> int:
        """Best-effort physical-core count (stdlib only, cross-platform).

        os.cpu_count() reports *logical* cores; we cap CPU encode workers at
        physical cores. Falls back to a hyper-threading heuristic if the
        platform probe is unavailable.
        """
        logical = os.cpu_count() or 1
        try:
            system = platform.system()
            if system == "Linux":
                ids = set()
                phys = core = None
                with open("/proc/cpuinfo", "r") as fh:
                    for line in fh:
                        if line.startswith("physical id"):
                            phys = line.split(":")[1].strip()
                        elif line.startswith("core id"):
                            core = line.split(":")[1].strip()
                            if phys is not None:
                                ids.add((phys, core))
                if ids:
                    return len(ids)
            elif system == "Darwin":
                out = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip().isdigit():
                    return int(out.stdout.strip())
            elif system == "Windows":
                # PowerShell is the reliable source on modern Win11 (wmic is
                # deprecated / often absent). Sum cores across sockets.
                ps = (
                    "(Get-CimInstance Win32_Processor | "
                    "Measure-Object -Property NumberOfCores -Sum).Sum"
                )
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps],
                    capture_output=True, text=True, timeout=15)
                val = out.stdout.strip()
                if out.returncode == 0 and val.isdigit() and int(val) > 0:
                    return int(val)
        except Exception:
            pass
        # Heuristic fallback: assume hyper-threading on multi-core x86.
        return max(1, logical // 2) if logical > 2 else logical

    def probe_encoder(self, codec: str) -> bool:
        """Return True iff `codec` actually initializes (1-frame dry encode).

        Presence in `ffmpeg -encoders` is a compile-time fact, NOT a runtime
        guarantee (e.g. h264_nvenc is listed even with no NVIDIA driver). This
        does a real one-frame encode and surfaces the specific init error.
        """
        if codec in self._encoder_probe_cache:
            return self._encoder_probe_cache[codec]

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1",
            "-frames:v", "1", "-c:v", codec, "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            ok = proc.returncode == 0
            if not ok:
                # Surface the real cause (e.g. "Cannot load libcuda.so.1"),
                # not the generic "Nothing was written" tail.
                cause = self._extract_encoder_error(proc.stderr, codec)
                self.logger.warning(f"Encoder '{codec}' did not initialize: {cause}")
        except subprocess.TimeoutExpired:
            ok = False
            self.logger.warning(f"Encoder '{codec}' probe timed out")
        except Exception as exc:
            ok = False
            self.logger.warning(f"Encoder '{codec}' probe error: {exc}")

        self._encoder_probe_cache[codec] = ok
        return ok

    @staticmethod
    def _extract_encoder_error(stderr: str, codec: str) -> str:
        """Pull the most informative line out of an ffmpeg encoder failure."""
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
        noise = ("Nothing was written", "Error while filtering",
                 "Conversion failed", "Error while opening encoder")
        for ln in lines:
            if codec in ln or "Cannot load" in ln or "MFX" in ln or \
                    "device" in ln.lower() or "driver" in ln.lower():
                if not any(n in ln for n in noise):
                    return ln
        return lines[-1] if lines else "(no error output)"

    def resolve_encoder(self, force: bool = False) -> Optional[str]:
        """Resolve the requested --encoder to a concrete, working encoder key.

        - 'x264'                -> always available (CPU).
        - 'nvenc'/'hevc_nvenc'  -> probed; if it fails and `force`, abort
                                   (return None); auto-fallback is disabled when
                                   the user explicitly named a GPU encoder.
        - 'auto'                -> h264_nvenc if it initializes, else x264.

        Caches the result in self.resolved_encoder.
        """
        if self.resolved_encoder:
            return self.resolved_encoder

        req = self.encoder
        if req == "auto":
            if self.probe_encoder("h264_nvenc"):
                self.logger.info("--encoder auto -> h264_nvenc (probe OK)")
                self.resolved_encoder = "nvenc"
            else:
                self.logger.info(
                    "--encoder auto -> x264 (no working GPU encoder detected)")
                self.resolved_encoder = "x264"
            return self.resolved_encoder

        if req not in self.config["encoders"]:
            self.logger.error(f"Unknown encoder: {req}")
            return None

        spec = self.config["encoders"][req]
        if spec["kind"] == "gpu":
            if self.probe_encoder(spec["codec"]):
                self.logger.info(f"Using GPU encoder: {spec['codec']}")
                self.resolved_encoder = req
                return req
            # Explicitly requested GPU encoder that does not work: fail loudly.
            self.logger.error(
                f"Requested encoder '{req}' ({spec['codec']}) is not usable on "
                f"this machine. Re-run with --encoder x264, or fix the GPU "
                f"driver/runtime. Refusing to silently produce a different "
                f"result.")
            return None

        self.resolved_encoder = req
        return req

    # ------------------------------------------------------------------
    # Encode planning: codec args, scale, framerate, workers, chunks
    # ------------------------------------------------------------------

    def _video_codec_args(self, preset_name: str, encoder_key: str) -> List[str]:
        """Codec + quality args for a preset under the resolved encoder.

        The CRF->NVENC-cq mapping lives in config['nvenc_cq']. x264 uses CRF
        directly (unchanged from prior behavior).
        """
        preset = self.config["ffmpeg_presets"][preset_name]
        spec = self.config["encoders"][encoder_key]
        codec = spec["codec"]

        if spec["kind"] == "cpu":
            return ["-c:v", codec, "-crf", str(preset["crf"]),
                    "-pix_fmt", "yuv420p"]

        # NVENC: -crf is ignored; map to -cq under VBR with no bitrate cap.
        cq = self.config["nvenc_cq"].get(preset_name, preset["crf"])
        args = ["-c:v", codec] + list(self.config["nvenc_common"])
        args += ["-cq", str(cq), "-pix_fmt", "yuv420p"]
        return args

    def _scale_filter_args(self) -> List[str]:
        """Return ['-vf', 'scale=W:H'] if --scale was given, else []."""
        if not self.scale:
            return []
        return ["-vf", f"scale={self._normalized_scale()}"]

    def _normalized_scale(self) -> str:
        """Turn '1280x720' into ffmpeg's '1280:720' (accepts -1 for aspect)."""
        return self.scale.lower().replace("x", ":")

    def _hevc_mp4_tag_args(self, encoder_key: str) -> List[str]:
        """hvc1 tag so HEVC MP4s play in QuickTime/Apple stacks."""
        if self.config["encoders"][encoder_key]["hevc"]:
            return ["-tag:v", "hvc1"]
        return []

    def resolve_workers(self, encoder_key: str, n_files: int) -> Tuple[int, int]:
        """Resolve (--workers, threads_per_worker) with encoder-aware caps.

        x264 (CPU): cap at physical cores; give each worker logical/physical
            threads so total ≈ logical cores (no oversubscription). Phase 1
            showed "fewer, wider" instances avoid x264's thread-scaling
            plateau without oversubscribing.
        NVENC (GPU): cap at the driver session limit (8 on consumer GeForce,
            driver 551.76+) AND at physical cores (each session still needs a
            CPU thread to decode JPEGs). Never exceed the session limit.
        """
        spec = self.config["encoders"][encoder_key]
        is_gpu = spec["kind"] == "gpu"

        if self.workers_arg == "auto":
            if is_gpu:
                workers = max(1, min(self.config["nvenc_session_limit"],
                                     self.physical_cores))
            else:
                workers = max(1, self.physical_cores)
        else:
            try:
                workers = max(1, int(self.workers_arg))
            except ValueError:
                self.logger.warning(
                    f"Invalid --workers '{self.workers_arg}', using 1")
                workers = 1

        # Hard caps.
        if is_gpu:
            limit = self.config["nvenc_session_limit"]
            if workers > limit:
                self.logger.warning(
                    f"Requested {workers} GPU workers exceeds the NVENC session "
                    f"limit ({limit}); clamping to {limit}.")
                workers = limit
        else:
            if workers > self.logical_cores:
                self.logger.warning(
                    f"Requested {workers} workers exceeds {self.logical_cores} "
                    f"logical cores; performance may degrade.")

        # Never spawn more chunks than frames.
        workers = max(1, min(workers, n_files))

        # Per-worker CPU thread budget: split logical cores across workers so the
        # concurrent JPEG decoders (and, for x264, the encoders) do not
        # oversubscribe. Applies to GPU too -- there the budget caps the CPU-side
        # MJPEG decode that feeds the encoder (see _chunk_encode_command).
        if self.threads_per_worker:
            threads = max(1, self.threads_per_worker)
        else:
            threads = max(1, self.logical_cores // workers)

        return workers, threads

    @staticmethod
    def _chunk_ranges(n: int, k: int) -> List[Tuple[int, int]]:
        """k contiguous, near-equal ranges covering [0, n) with no gaps."""
        base, rem = divmod(n, k)
        ranges, start = [], 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            ranges.append((start, start + size))
            start += size
        return ranges

    # ------------------------------------------------------------------
    # Chunked encoding (the parallel workhorse)
    # ------------------------------------------------------------------

    def _write_chunk_list(
        self, ordered_slice: List[Tuple[Path, datetime]], path: Path
    ) -> None:
        """Write a concat sub-list for one chunk (forward-slash paths)."""
        with open(path, "w", encoding="utf-8") as fh:
            for file_path, _ in ordered_slice:
                safe_path = str(file_path.resolve()).replace("\\", "/")
                fh.write(f"file '{safe_path}'\n")

    def _chunk_encode_command(
        self, chunk_list: Path, preset_name: str, encoder_key: str,
        segment: Path, threads: int
    ) -> List[str]:
        """ffmpeg command to encode one chunk to an intermediate segment."""
        preset = self.config["ffmpeg_presets"][preset_name]
        is_gpu = self.config["encoders"][encoder_key]["kind"] == "gpu"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               # Input framerate (see Phase 1 A): correct, frame-exact timing.
               "-r", str(preset["framerate"])]
        # Cap DECODE threads on the INPUT side. Critical for GPU workers: N
        # concurrent uncapped 4K MJPEG decoders oversubscribe the CPU and starve
        # the encoder (the 8x-4K case that ran at 4 fps). x264's gate is its own
        # encode threading, set on the output side below.
        if threads and is_gpu:
            cmd += ["-threads", str(threads)]
        cmd += ["-f", "concat", "-safe", "0", "-i", str(chunk_list)]
        cmd += self._scale_filter_args()
        # Never drop/duplicate frames: pass every decoded frame through with its
        # (regular, -r-derived) PTS. Guarantees segment frames == input frames,
        # which is what the seam check enforces.
        cmd += ["-fps_mode", "passthrough"]
        cmd += self._video_codec_args(preset_name, encoder_key)
        if threads and not is_gpu:
            cmd += ["-threads", str(threads)]   # x264 encode threads
        cmd += ["-f", self.config["intermediate_format"], str(segment)]
        return cmd

    def _stitch_segments(
        self, segments: List[Path], output_path: Path, encoder_key: str
    ) -> bool:
        """Losslessly concatenate intermediate segments with `-c copy`."""
        stitch_list = self.chunks_dir / "stitch_list.txt"
        with open(stitch_list, "w", encoding="utf-8") as fh:
            for seg in segments:
                safe = str(seg.resolve()).replace("\\", "/")
                fh.write(f"file '{safe}'\n")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", str(stitch_list),
               "-c", "copy"]
        cmd += self._hevc_mp4_tag_args(encoder_key)
        cmd += ["-movflags", "+faststart", str(output_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            self.logger.error(f"Stitch failed: {result.stderr.strip()}")
            return False
        return True

    def _count_video_frames(self, path: Path) -> int:
        """Exact decoded frame count via ffprobe -count_frames."""
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True)
        # MPEG-TS may list the stream twice (program + global); take first int.
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
        return 0

    def create_video_chunked(
        self, preset_name: str, encoder_key: str, workers: int, threads: int,
        output_path: Path
    ) -> bool:
        """Encode one preset via N concurrent chunks, then stitch + verify.

        Correctness: chunks are contiguous chronological ranges; the stitched
        output's frame count must equal the sum of chunk frames (== input
        count). Verified here and aborted on mismatch.
        """
        n = len(self.ordered_files)
        ranges = self._chunk_ranges(n, workers)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"  Chunked encode: {workers} workers x {threads or 'auto'} threads, "
            f"{n} frames -> chunks {[b - a for a, b in ranges]}")

        segments: List[Path] = []
        procs = []
        ext = self.config["intermediate_ext"]
        t0 = time.perf_counter()
        for i, (a, b) in enumerate(ranges):
            chunk_list = self.chunks_dir / f"chunk_{preset_name}_{i:03d}.txt"
            segment = self.chunks_dir / f"seg_{preset_name}_{i:03d}.{ext}"
            self._write_chunk_list(self.ordered_files[a:b], chunk_list)
            cmd = self._chunk_encode_command(
                chunk_list, preset_name, encoder_key, segment, threads)
            procs.append((i, subprocess.Popen(cmd, stderr=subprocess.PIPE,
                                              text=True)))
            segments.append(segment)

        failed = False
        for i, proc in procs:
            _, err = proc.communicate()
            if proc.returncode != 0:
                failed = True
                self.logger.error(
                    f"  Chunk {i} encode failed: "
                    f"{(err or '').strip().splitlines()[-1:] or ['(no msg)']}")
        if failed:
            return False

        # Per-segment frame accounting (the seam-correctness ground truth).
        seg_frames = [self._count_video_frames(s) for s in segments]
        expected = sum(b - a for a, b in ranges)
        if sum(seg_frames) != expected:
            self.logger.error(
                f"  Segment frame sum {sum(seg_frames)} != expected {expected}; "
                f"refusing to stitch a corrupt result.")
            return False

        if not self._stitch_segments(segments, output_path, encoder_key):
            return False

        final_frames = self._count_video_frames(output_path)
        elapsed = time.perf_counter() - t0
        if final_frames != expected:
            self.logger.error(
                f"  Seam check FAILED: final {final_frames} != input {expected} "
                f"frames (dropped/duplicated at a boundary).")
            return False

        self.logger.info(
            f"  Seam check OK: {final_frames} frames == sum of chunks "
            f"({'+'.join(map(str, seg_frames))}) in {elapsed:.1f}s "
            f"({final_frames / elapsed:.1f} fps)")

        # Intermediates are ~the size of the final video; remove this preset's
        # segments + chunk lists now that the verified output exists. Left in
        # place on any failure above for debugging.
        for seg in segments:
            seg.unlink(missing_ok=True)
        for i in range(len(ranges)):
            (self.chunks_dir / f"chunk_{preset_name}_{i:03d}.txt").unlink(
                missing_ok=True)
        (self.chunks_dir / "stitch_list.txt").unlink(missing_ok=True)
        return True

    def generate_ffmpeg_command(
        self, preset_name: str, output_filename: str = None,
        encoder_key: str = "x264"
    ) -> List[str]:
        """Build the single-process ffmpeg command for a preset.

        Used when --workers resolves to 1 (default). Preserves the original
        output structure; the ONLY change vs. the legacy command is that the
        framerate is now an INPUT option (`-r` before `-i`) so the output plays
        at the preset's rate instead of the concat demuxer's 25 fps default.
        """
        preset = self.config["ffmpeg_presets"][preset_name]
        if output_filename is None:
            output_filename = f"timelapse_{preset['suffix']}.mp4"

        output_path = self.videos_dir / output_filename

        cmd = [
            "ffmpeg",
            "-y",
            # INPUT framerate — required for the concat demuxer (Phase 1 A).
            "-r", str(preset["framerate"]),
            # Concat demuxer — reads ordered file list directly (no rename)
            "-f", "concat",
            "-safe", "0",
            "-i", str(self.concat_list_path),
        ]
        cmd += self._scale_filter_args()
        # Never drop/duplicate frames during rate handling (see chunk builder).
        cmd += ["-fps_mode", "passthrough"]
        cmd += self._video_codec_args(preset_name, encoder_key)
        cmd += self._hevc_mp4_tag_args(encoder_key)
        cmd += ["-movflags", "+faststart", str(output_path)]
        return cmd

    def _load_ordered_from_concat_list(self) -> None:
        """Repopulate self.ordered_files from an existing concat_list.txt.

        Needed for chunked encoding under --video_only, where the manifest is
        not reloaded. The concat list already encodes the chronological order,
        so order is preserved; the datetime is irrelevant for chunking.
        """
        entries: List[Tuple[Path, datetime]] = []
        pat = re.compile(r"^file '(.*)'\s*$")
        with open(self.concat_list_path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = pat.match(line.strip())
                if m:
                    entries.append((Path(m.group(1)), None))
        self.ordered_files = entries

    def create_videos(
        self, presets: List[str] = None, dry_run: bool = False
    ) -> bool:
        """Render timelapse videos for each requested preset.

        Resolves the encoder (with a runtime GPU probe) and the worker count
        once, then renders each preset either single-process (workers==1, the
        default, output structure identical to legacy + framerate fix) or via
        concurrent chunked encoding (workers>1).
        """
        if presets is None:
            presets = list(self.config["ffmpeg_presets"].keys())

        self.logger.info(
            f"Creating videos with presets: {presets} (dry_run={dry_run})")

        if not dry_run and not self.check_ffmpeg():
            return False

        n_files = len(self.ordered_files)
        if n_files == 0 and self.concat_list_path.exists():
            # --video_only reuse path: recover order from the concat list so
            # chunked mode can split it.
            self._load_ordered_from_concat_list()
            n_files = len(self.ordered_files)

        # Resolve encoder. A dry run must not execute ffmpeg (no GPU probe), so
        # display the requested encoder unverified; a real run probes + may abort.
        if dry_run:
            encoder_key = self.encoder if self.encoder in self.config["encoders"] \
                else "x264"  # 'auto' shown as x264; probed for real at runtime
            if self.encoder in ("auto", "nvenc", "hevc_nvenc"):
                self.logger.info(
                    f"DRY RUN: '{self.encoder}' would be probed at runtime "
                    f"(showing '{encoder_key}' settings)")
        else:
            encoder_key = self.resolve_encoder()
            if encoder_key is None:
                return False

        workers, threads = self.resolve_workers(encoder_key, max(1, n_files))
        codec = self.config["encoders"][encoder_key]["codec"]
        self.logger.info(
            f"Encoder: {codec}  |  Workers: {workers}  |  "
            f"Threads/worker: {threads or 'auto'}")

        success_count = 0
        for preset_name in presets:
            if preset_name not in self.config["ffmpeg_presets"]:
                self.logger.warning(f"Unknown preset: {preset_name}")
                continue

            suffix = self.config["ffmpeg_presets"][preset_name]["suffix"]
            output_file = self.videos_dir / f"timelapse_{suffix}.mp4"

            self.logger.info(f"Generating {preset_name} video...")

            if dry_run:
                self._dry_run_show(preset_name, encoder_key, workers, threads,
                                   output_file, n_files)
                success_count += 1
                continue

            try:
                if workers == 1:
                    ok = self._encode_single(preset_name, encoder_key, output_file)
                else:
                    ok = self.create_video_chunked(
                        preset_name, encoder_key, workers, threads, output_file)
                if ok and output_file.exists():
                    size_mb = output_file.stat().st_size / (1024 * 1024)
                    self.logger.info(
                        f"Created {preset_name} video ({size_mb:.1f} MB): "
                        f"{output_file.name}")
                    success_count += 1
                elif not ok:
                    self.logger.error(f"Encode failed for preset '{preset_name}'")
            except subprocess.TimeoutExpired:
                self.logger.error(f"ffmpeg timed out for preset '{preset_name}'")
            except Exception as exc:
                self.logger.error(f"Error rendering '{preset_name}': {exc}")

        return success_count == len(presets)

    def _encode_single(
        self, preset_name: str, encoder_key: str, output_file: Path
    ) -> bool:
        """Single-process encode (default path)."""
        command = self.generate_ffmpeg_command(
            preset_name, output_filename=output_file.name, encoder_key=encoder_key)
        command_str = " ".join(
            f'"{a}"' if " " in str(a) else str(a) for a in command)
        self.logger.info(f"  Command: {command_str}")
        t0 = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=7200)
        if result.returncode != 0:
            self.logger.error(f"  ffmpeg failed:\n  STDERR: {result.stderr}")
            return False
        elapsed = time.perf_counter() - t0
        self.logger.info(f"  Encoded in {elapsed:.1f}s")
        return True

    def _dry_run_show(
        self, preset_name: str, encoder_key: str, workers: int, threads: int,
        output_file: Path, n_files: int
    ) -> None:
        """Log the exact plan/command(s) a real run would execute."""
        def s(cmd):
            return " ".join(f'"{a}"' if " " in str(a) else str(a) for a in cmd)

        if workers == 1:
            cmd = self.generate_ffmpeg_command(
                preset_name, output_filename=output_file.name,
                encoder_key=encoder_key)
            self.logger.info(f"  DRY RUN [single-process]: {s(cmd)}")
        else:
            ranges = self._chunk_ranges(max(1, n_files), workers)
            ext = self.config["intermediate_ext"]
            sample = self._chunk_encode_command(
                self.chunks_dir / f"chunk_{preset_name}_000.txt", preset_name,
                encoder_key, self.chunks_dir / f"seg_{preset_name}_000.{ext}",
                threads)
            self.logger.info(
                f"  DRY RUN [{workers}-way chunked + stitch]: "
                f"chunk sizes {[b - a for a, b in ranges]}")
            self.logger.info(f"    per-chunk (x{workers}): {s(sample)}")
            self.logger.info(
                f"    stitch: ffmpeg -f concat -safe 0 -i stitch_list.txt "
                f"-c copy -movflags +faststart {output_file}")

    # ------------------------------------------------------------------
    # Benchmark mode (decode ceiling vs full encode -> bottleneck verdict)
    # ------------------------------------------------------------------

    def _sample_source_stats(self, sample) -> Tuple[str, float]:
        """Return (WxH of the first readable frame, mean KiB/file over sample)."""
        dims = "unknown"
        for p, _ in sample:
            if p.exists():
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "csv=p=0:s=x", str(p)],
                    capture_output=True, text=True)
                if out.returncode == 0 and out.stdout.strip():
                    dims = out.stdout.strip()
                break
        sizes = []
        for p, _ in sample[:60]:
            try:
                sizes.append(p.stat().st_size)
            except OSError:
                pass
        mean_kib = (sum(sizes) / len(sizes) / 1024) if sizes else 0.0
        return dims, mean_kib

    def _parallel_decode_fps(self, block, k: int) -> Optional[float]:
        """Aggregate null-decode fps for `block` split across k processes.

        Tells us whether the decode gate actually parallelizes on THIS machine
        (CPU-bound / per-file-latency-bound decode scales with k; raw-disk-
        bandwidth-bound decode stays flat). `block` should be a fresh, cold
        slice so reads are not served from the page cache.
        """
        ranges = self._chunk_ranges(len(block), k)
        procs, lists = [], []
        t0 = time.perf_counter()
        for i, (a, b) in enumerate(ranges):
            lst = self.output_dir / f"bench_dec_{k}_{i}.txt"
            self._write_chunk_list(block[a:b], lst)
            lists.append(lst)
            procs.append(subprocess.Popen(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-f", "concat", "-safe", "0", "-i", str(lst), "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        rc = [p.wait() for p in procs]
        el = time.perf_counter() - t0
        for lst in lists:
            lst.unlink(missing_ok=True)
        if any(rc) or not el:
            return None
        return len(block) / el

    def _measure_encode_fps(self, block, workers: int, encoder_key: str,
                            preset_name: str) -> Tuple[Optional[float], int]:
        """Real chunked-encode wall-clock fps for `block` (encode only, no
        stitch), plus the produced frame count.

        This grounds the worker/encoder recommendation in measured throughput.
        A decode-ceiling estimate alone badly over-predicts GPU at high
        resolution: many concurrent 4K NVENC sessions contend instead of
        scaling, so the measured number is the honest one.
        """
        ranges = self._chunk_ranges(len(block), workers)
        threads = max(1, self.logical_cores // workers)
        procs, arts = [], []
        ext = self.config["intermediate_ext"]
        t0 = time.perf_counter()
        for i, (a, b) in enumerate(ranges):
            cl = self.output_dir / f"bench_enc_{workers}_{i}.txt"
            seg = self.output_dir / f"bench_enc_{workers}_{i}.{ext}"
            self._write_chunk_list(block[a:b], cl)
            arts.append((cl, seg))
            cmd = self._chunk_encode_command(
                cl, preset_name, encoder_key, seg, threads)
            procs.append(subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        rc = [p.wait() for p in procs]
        el = time.perf_counter() - t0
        frames = sum(self._count_video_frames(seg)
                     for _, seg in arts if seg.exists())
        for cl, seg in arts:
            cl.unlink(missing_ok=True)
            seg.unlink(missing_ok=True)
        if any(rc) or not el:
            return None, 0
        return len(block) / el, frames

    def run_benchmark(self, presets: List[str], sample_frames: int = 200) -> bool:
        """Measure the decode ceiling vs full-encode fps and recommend settings.

        Bakes the Phase 1 methodology into the tool so the verdict can be
        reproduced on any dataset / hardware. Uses a leading sample of the
        ordered files (kept small so it is quick and repeatable).
        """
        if not self.check_ffmpeg():
            return False
        if not self.ordered_files:
            self.logger.error("Benchmark needs a resolved file list (manifest).")
            return False

        sample = self.ordered_files[:min(sample_frames, len(self.ordered_files))]
        n = len(sample)
        bench_list = self.output_dir / "benchmark_concat.txt"
        self._write_chunk_list(sample, bench_list)
        inp = ["-f", "concat", "-safe", "0", "-i", str(bench_list)]

        def fps_of(extra_args, label):
            cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-nostdin"] + extra_args)
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            el = time.perf_counter() - t0
            if proc.returncode != 0:
                self.logger.warning(f"  {label}: FAILED")
                return None
            fps = n / el if el else 0
            self.logger.info(f"  {label:42s} {fps:7.1f} fps ({el:.2f}s)")
            return fps

        print("\n" + "=" * 64)
        print(f"BENCHMARK — decode ceiling vs full encode  (sample: {n} frames)")
        print("=" * 64)

        dims, mean_kib = self._sample_source_stats(sample)
        self.logger.info(f"Source: {dims} px, mean {mean_kib:.0f} KiB/frame")

        # NOTE on caching: the first read of each frame is cold (off disk); a
        # real one-pass job reads every frame exactly once, so the COLD number
        # is the realistic decode ceiling. Re-reads are served warm from the OS
        # page cache and overstate one-pass throughput.
        self.logger.info("Stage throughput (single process):")
        decode = fps_of(inp + ["-f", "null", "-"],
                        "decode only (cold, 1-pass ceiling)")
        fps_of(inp + ["-vf", "format=yuv420p", "-f", "null", "-"],
               "decode + yuvj420p->yuv420p convert (warm)")
        if self.scale:
            fps_of(inp + ["-vf", f"scale={self._normalized_scale()}",
                          "-f", "null", "-"], f"decode + scale {self.scale} (warm)")

        # Does the decode gate parallelize on THIS storage? This is the crux
        # when decode is the gate -- e.g. NVENC makes encode ~free, so
        # throughput becomes parallel decode throughput. Flat scaling =>
        # I/O-bound (storage, not CPU). Each point uses a FRESH, previously-
        # untouched block so reads are COLD (matching a real one-pass job).
        # If there are not enough fresh frames for an all-cold series, fall back
        # to a warm series (same sample for every k) and say so -- never MIX
        # cold and warm points, which would give meaningless non-monotonic fps.
        decode_points = sorted({1, max(2, self.physical_cores // 2),
                                self.physical_cores})
        cold_budget = len(self.ordered_files) - n
        block = min(sample_frames, 160)
        series_warm = cold_budget < block * len(decode_points)
        if series_warm and cold_budget >= 24 * len(decode_points):
            block = cold_budget // len(decode_points)   # smaller, still cold
            series_warm = False
        self.logger.info(
            f"Parallel decode scaling "
            f"({'WARM cache (CPU scaling only)' if series_warm else 'cold reads'}"
            f", null encoder):")
        decode_par, cursor = {}, n
        for k in decode_points:
            if series_warm:
                blk = sample
            else:
                blk = self.ordered_files[cursor:cursor + block]
                cursor += block
            fps = self._parallel_decode_fps(blk, k)
            decode_par[k] = fps
            if fps:
                base = decode_par.get(1) or fps
                self.logger.info(
                    f"  {k:2d} proc{'s' if k > 1 else ' '}: {fps:7.1f} fps "
                    f"({fps / base:.2f}x vs 1)")
        peak_decode = max((v for v in decode_par.values() if v), default=decode)
        decode_scaling = ((peak_decode / decode_par[1])
                          if decode_par.get(1) else 1.0)

        # GPU probe (presence != works).
        gpu_ok = self.probe_encoder("h264_nvenc")
        self.logger.info(
            f"GPU encoder h264_nvenc: "
            f"{'WORKS (runtime probe OK)' if gpu_ok else 'NOT USABLE here'}")

        verdicts = {}
        enc_fps_by_preset = {}
        self.logger.info("Full encode (x264, single process) per preset:")
        for name in presets:
            preset = self.config["ffmpeg_presets"][name]
            out = self.output_dir / f"benchmark_{name}.mp4"
            enc = fps_of(
                ["-r", str(preset["framerate"])] + inp
                + ["-c:v", "libx264", "-crf", str(preset["crf"]),
                   "-pix_fmt", "yuv420p", str(out)],
                f"encode x264 {name} (crf {preset['crf']})")
            if enc:
                enc_fps_by_preset[name] = enc
                if decode:
                    verdicts[name] = decode / enc

        print("\nVerdict per preset (decode_ceiling / encode_fps):")
        encode_bound_any = False
        for name, ratio in verdicts.items():
            if ratio >= 1.5:
                v = "ENCODE-bound"
                encode_bound_any = True
            elif ratio <= 1.15:
                v = "DECODE-bound"
            else:
                v = "MIXED"
            print(f"  {name:14s} {ratio:5.2f}x  -> {v}")

        decode_parallelizes = decode_scaling >= 1.5
        print(f"\nParallel decode scaling: {decode_scaling:.2f}x "
              f"(peak {peak_decode:.1f} fps at {self.physical_cores} procs)")
        if series_warm:
            print("  (scaling measured WARM — dataset too small for a cold test; "
                  "this reflects CPU decode scaling, not cold-disk parallelism)")

        if not decode_parallelizes:
            # Decode does not scale with processes -> storage/I/O is the gate.
            print("\n** DECODE DOES NOT PARALLELIZE on this storage (flat scaling)")
            print(f"   -> I/O-bound at ~{decode:.0f} fps; --workers/NVENC give "
                  f"limited gains. Fix the source read first (local NVMe/SSD,")
            print("   exclude from antivirus, no OneDrive placeholders), then "
                  "re-run --benchmark.")
            print("Recommendation:")
            print(f"  --encoder x264 --workers "
                  f"{max(2, self.physical_cores // 2)}")
            print("=" * 64)
            for name in presets:
                (self.output_dir / f"benchmark_{name}.mp4").unlink(missing_ok=True)
            bench_list.unlink(missing_ok=True)
            return True

        # Measure REAL parallel-encode throughput. Decode-ceiling alone badly
        # over-predicts GPU at high resolution (8x 4K NVENC sessions contend,
        # not scale), so we time actual chunked encodes and recommend the
        # measured winner. Use the heaviest (slowest x264) preset and a fresh
        # cold block continuing past the decode-scaling cursor.
        test_preset = (min(enc_fps_by_preset, key=enc_fps_by_preset.get)
                       if enc_fps_by_preset else presets[0])
        blk = self.ordered_files[cursor:cursor + min(sample_frames, 96)]
        if len(blk) < 16:
            blk = sample
        cores = self.physical_cores

        configs = [("x264", cores)]
        if gpu_ok:
            cap = min(self.config["nvenc_session_limit"], cores)
            lo = max(2, cores // 4)
            configs += [("nvenc", lo)]
            if cap != lo:
                configs += [("nvenc", cap)]

        self.logger.info(
            f"Measured parallel-encode throughput ({test_preset}, "
            f"{len(blk)} frames):")
        measured = []
        for enc_key, w in configs:
            w = min(w, len(blk))
            fps, frames = self._measure_encode_fps(blk, w, enc_key, test_preset)
            if fps:
                drop = "" if frames == len(blk) else f"  !! {len(blk)-frames} dropped"
                self.logger.info(
                    f"  {enc_key:5s} x{w:<2d}: {fps:6.1f} fps{drop}")
                measured.append((fps, enc_key, w))
        # Single-process x264 is the baseline already measured above.
        if enc_fps_by_preset.get(test_preset):
            measured.append((enc_fps_by_preset[test_preset], "x264", 1))

        print("\nRecommendation (measured):")
        if measured:
            best_fps, best_enc, best_w = max(measured, key=lambda x: x[0])
            print(f"  --encoder {best_enc} --workers {best_w}    "
                  f"(~{best_fps:.0f} fps on the {test_preset} preset)")
            base = enc_fps_by_preset.get(test_preset, 0)
            if base:
                print(f"  vs single-process x264 ~{base:.0f} fps "
                      f"({best_fps / base:.1f}x)")
            if best_enc == "nvenc":
                print("  NVENC trades fidelity-per-bit for speed and balloons "
                      "file size at low cq (construction=10, high_quality=15);")
                print("  use --encoder x264 for archival masters.")
            elif gpu_ok:
                print("  NVENC was available but did NOT beat x264 here (at this "
                      "resolution concurrent GPU sessions contend) -> x264 wins.")
        else:
            print(f"  --encoder x264 --workers {cores}  (encode measurement "
                  f"unavailable; default to parallel x264)")
        print("=" * 64)

        # Tidy up benchmark artifacts.
        for name in presets:
            (self.output_dir / f"benchmark_{name}.mp4").unlink(missing_ok=True)
        bench_list.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    # Statistics & summary
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return a dict of processing statistics."""
        stats = {
            "platform": platform.system(),
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "files_resolved": len(self.ordered_files),
            "max_workers": self.config["max_workers"],
            "parallel_threshold": self.config["parallel_threshold"],
            "display_timezone": self.tz_label,
            "encoder_requested": self.encoder,
            "encoder_resolved": self.resolved_encoder or "(unresolved)",
            "encode_workers": self.workers_arg,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "scale": self.scale or "none",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.videos_dir.exists():
            video_files = sorted(self.videos_dir.glob("*.mp4"))
            stats["videos_created"] = len(video_files)
            stats["video_files"] = [f.name for f in video_files]
        else:
            stats["videos_created"] = 0

        # Timestamps are None when the order was recovered from an existing
        # concat list (--video_only reuse), so guard before building the range.
        if self.ordered_files and self.ordered_files[0][1] is not None:
            start_utc = self.ordered_files[0][1]
            end_utc = self.ordered_files[-1][1]
            stats["time_range"] = {
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
                "start_local": start_utc.astimezone(self.local_tz).isoformat(),
                "end_local": end_utc.astimezone(self.local_tz).isoformat(),
                "duration_hours": (end_utc - start_utc).total_seconds() / 3600,
            }

        return stats

    def print_summary(self) -> None:
        """Print a human-readable processing summary."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("TIMELAPSE CREATION SUMMARY")
        print("=" * 60)
        print(f"Platform:            {stats['platform']}")
        print(f"Source Directory:    {stats['source_dir']}")
        print(f"Output Directory:    {stats['output_dir']}")
        print(f"Manifest:            {stats['manifest_path']}")
        print(f"Files Resolved:      {stats['files_resolved']}")
        print(f"Videos Created:      {stats['videos_created']}")
        print(f"Encoder:             {stats['encoder_requested']} "
              f"-> {stats['encoder_resolved']}")
        print(f"Encode Workers:      {stats['encode_workers']} "
              f"(cores: {stats['physical_cores']}P/{stats['logical_cores']}L)")
        print(f"Scale:               {stats['scale']}")
        print(f"Manifest Threads:    {stats['max_workers']}")
        print(f"Display Timezone:    {stats['display_timezone']}")

        if stats.get("video_files"):
            print("Video Files:")
            for video in stats["video_files"]:
                print(f"  - {video}")

        if "time_range" in stats:
            tr = stats["time_range"]
            print(f"Time Range ({self.tz_label}):  "
                  f"{tr['start_local']} → {tr['end_local']}")
            print(f"Duration:            {tr['duration_hours']:.1f} hours")

        print(f"Completed at (UTC):  {stats['timestamp']}")
        print("=" * 60)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> int:
    preset_choices = ["preview", "standard", "high_quality", "smooth", "construction"]

    parser = argparse.ArgumentParser(
        description=(
            "Create timelapse videos from a manifest of timestamped images. "
            "No file renaming required — ffmpeg reads source files directly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python timelapse_creator.py --dry_run
  python timelapse_creator.py --source_dir ./captures --output_dir ./out
  python timelapse_creator.py --manifest ./manifest.json --tz_offset -7
  python timelapse_creator.py --presets preview standard
  python timelapse_creator.py --video_only --presets high_quality
  python timelapse_creator.py --benchmark --presets high_quality construction
  python timelapse_creator.py --workers auto --presets high_quality
  python timelapse_creator.py --encoder auto --workers auto --trust-manifest
  python timelapse_creator.py --encoder nvenc --workers 6 --scale 1920x1080

Speed/quality notes:
  * Default is single-process x264 (output matches prior runs; only the
    framerate is corrected). Opt into speed with --workers auto (or N) and,
    if you have a working NVENC GPU, --encoder nvenc.
  * --workers >1 splits the ordered frames into contiguous chunks, encodes
    them concurrently to MPEG-TS intermediates, and stitches losslessly
    (-c copy). Frame count is verified across the seams.
  * NVENC ignores -crf; each preset's CRF is mapped 1:1 to NVENC -cq (VBR,
    no bitrate cap). NVENC trades fidelity-per-bit for speed -- most visible
    at the low-CRF presets (construction=10, high_quality=15), where files
    may be larger and detail lower than x264 at the same number.
  * Run --benchmark first to get a decode-bound vs encode-bound verdict and a
    recommended --encoder / --workers for YOUR machine and dataset.
        """,
    )

    parser.add_argument(
        "--source_dir",
        help="Directory containing source images (default: ./captures)",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for output files and videos (default: ./processed)",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        help="Path to manifest.json (default: <source_dir>/manifest.json)",
    )
    parser.add_argument(
        "--tz_offset",
        type=float,
        default=-6.0,
        metavar="HOURS",
        help=(
            "UTC offset in hours for local-time display in logs "
            "(e.g. -6 for Mountain Standard, -7 for MDT). "
            "Default: -6 (Mountain Standard Time)"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without executing ffmpeg or writing files",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        help=(
            "Skip straight to video rendering. Reuses an existing "
            "concat_list.txt in output_dir if present; otherwise loads "
            "the manifest and generates it automatically first."
        ),
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=preset_choices,
        default=preset_choices,
        help="Video presets to generate (default: all)",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not back up original files (backup is a no-op in manifest mode)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Max worker threads for parallel manifest resolution (default: auto)",
    )
    parser.add_argument(
        "--parallel_threshold",
        type=int,
        help=(
            "File count above which threaded manifest resolution is used "
            "(default: 0 = always threaded)"
        ),
    )

    # ------------------------------------------------------------------
    # Phase 2: encoder / parallelism / scaling / benchmark
    # ------------------------------------------------------------------
    parser.add_argument(
        "--encoder",
        choices=["x264", "nvenc", "hevc_nvenc", "auto"],
        default="x264",
        help=(
            "Video encoder. x264 (default, CPU, bit-for-bit comparable to "
            "prior runs); nvenc / hevc_nvenc (NVIDIA GPU, opt-in, faster but "
            "lower fidelity-per-bit -- see notes below); auto picks a working "
            "GPU encoder if one initializes, else falls back to x264. A forced "
            "GPU encoder that fails to initialize aborts loudly."
        ),
    )
    parser.add_argument(
        "--workers",
        default="1",
        metavar="N|auto",
        help=(
            "Concurrent encode workers (chunks). 1 = single process (default). "
            "'auto' = physical cores for x264, or min(NVENC session limit, "
            "cores) for GPU. >1 splits the ordered frames into contiguous "
            "chunks, encodes concurrently, and stitches losslessly."
        ),
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=None,
        help=(
            "x264 threads per worker (default: auto = logical_cores // workers "
            "to avoid oversubscription). Ignored for NVENC."
        ),
    )
    parser.add_argument(
        "--scale",
        default=None,
        metavar="WxH",
        help=(
            "Optional downscale, e.g. 1280x720 (use -1 to preserve aspect, e.g. "
            "1280x-1). Applied early via CPU scale to lighten every later stage."
        ),
    )
    parser.add_argument(
        "--trust-manifest",
        "--trust_manifest",
        dest="trust_manifest",
        action="store_true",
        help=(
            "Treat the manifest as authoritative and skip per-file existence "
            "checks (removes ~N stat() calls; the manifest is generated "
            "alongside the captures)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Measure the decode ceiling vs full-encode fps on a sample and "
            "print a decode-bound/encode-bound verdict + recommended settings, "
            "then exit (no full render)."
        ),
    )
    parser.add_argument(
        "--benchmark_frames",
        type=int,
        default=200,
        help="Frames sampled for --benchmark (default: 200)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    try:
        creator = TimelapseCreator(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            backup_originals=not args.no_backup,
            tz_offset_hours=args.tz_offset,
            encoder=args.encoder,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            scale=args.scale,
            trust_manifest=args.trust_manifest,
        )
    except Exception as exc:
        print(f"Error initialising timelapse creator: {exc}")
        return 1

    if args.max_workers:
        creator.config["max_workers"] = args.max_workers
        creator.logger.info(f"Custom max_workers: {args.max_workers}")

    if args.parallel_threshold:
        creator.config["parallel_threshold"] = args.parallel_threshold
        creator.logger.info(f"Custom parallel_threshold: {args.parallel_threshold}")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    try:
        # Benchmark mode: resolve the file order, measure, recommend, exit.
        if args.benchmark:
            if not creator.load_manifest() and creator.concat_list_path.exists():
                creator._load_ordered_from_concat_list()
            if not creator.ordered_files:
                print("Benchmark needs a manifest or existing concat list.")
                return 1
            return 0 if creator.run_benchmark(
                args.presets, sample_frames=args.benchmark_frames) else 1

        concat_list_exists = creator.concat_list_path.exists()

        if args.video_only and concat_list_exists:
            # Reuse the existing concat list — nothing to regenerate
            creator.logger.info(
                f"--video_only: reusing existing concat list: "
                f"{creator.concat_list_path}"
            )
        else:
            # Generate (or regenerate) the concat list from the manifest.
            # This path is also taken when --video_only is set but no
            # concat_list.txt exists yet (e.g. first run with --video_only).
            if args.video_only:
                creator.logger.info(
                    "--video_only requested but concat_list.txt not found — "
                    "loading manifest to generate it now"
                )

            # Step 1: Load manifest → ordered file list
            ordered_files = creator.load_manifest()
            if not ordered_files:
                print("No files resolved from manifest — aborting.")
                return 1

            # Step 2: Write concat list (replaces the rename step)
            if not creator.write_concat_list(ordered_files, dry_run=args.dry_run):
                print("Failed to write concat list — aborting.")
                return 1

        # Step 3: Render videos (create_videos handles dry_run internally,
        # resolving the encoder/worker plan and logging the exact commands).
        if not creator.create_videos(presets=args.presets, dry_run=args.dry_run):
            if not args.dry_run:
                print("Video creation failed!")
                return 1

        # Step 4: Summary
        creator.print_summary()
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as exc:
        creator.logger.error(f"Unexpected error: {exc}")
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
