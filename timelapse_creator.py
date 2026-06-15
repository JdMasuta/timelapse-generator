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
- Timezone-offset flag for local-time display in logs
- Comprehensive logging and error handling
- Cross-platform compatibility

Usage:
    python timelapse_creator.py [--source_dir PATH] [--output_dir PATH]
                                [--manifest PATH] [--tz_offset HOURS]
                                [--dry_run] [--video_only]
                                [--presets PRESET [PRESET ...]]
"""

import os
import argparse
import subprocess
import logging
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
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

        # Configuration
        self.config = {
            "log_level": "INFO",
            "image_extensions": {".jpg", ".jpeg", ".png", ".tiff", ".bmp"},
            "max_workers": min(16, (os.cpu_count() or 1) * 2),
            "parallel_threshold": 10_000,
            "ffmpeg_presets": {
                "preview": {"framerate": 60, "crf": 23, "suffix": "preview"},
                "standard": {"framerate": 30, "crf": 17, "suffix": "standard"},
                "high_quality": {"framerate": 24, "crf": 15, "suffix": "hq"},
                "smooth": {"framerate": 45, "crf": 20, "suffix": "smooth"},
                "construction": {"framerate": 20, "crf": 10, "suffix": "construction"},
            },
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

        # Resolve each entry — large manifests benefit from parallel path checks
        if len(manifest) > self.config["parallel_threshold"]:
            self.logger.info(
                f"Large manifest ({len(manifest)} entries): "
                "using parallel path resolution"
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
        if not file_path.exists():
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

    def generate_ffmpeg_command(
        self, preset_name: str, output_filename: str = None
    ) -> List[str]:
        """Build the ffmpeg command for a given preset using the concat demuxer."""
        preset = self.config["ffmpeg_presets"][preset_name]
        if output_filename is None:
            output_filename = f"timelapse_{preset['suffix']}.mp4"

        output_path = self.videos_dir / output_filename

        return [
            "ffmpeg",
            "-y",
            # Concat demuxer — reads ordered file list directly (no rename needed)
            "-f", "concat",
            "-safe", "0",
            "-i", str(self.concat_list_path),
            # Output settings
            "-framerate", str(preset["framerate"]),
            "-c:v", "libx264",
            "-crf", str(preset["crf"]),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

    def create_videos(
        self, presets: List[str] = None, dry_run: bool = False
    ) -> bool:
        """Render timelapse videos for each requested preset."""
        if presets is None:
            presets = list(self.config["ffmpeg_presets"].keys())

        self.logger.info(f"Creating videos with presets: {presets} (dry_run={dry_run})")

        if not dry_run and not self.check_ffmpeg():
            return False

        success_count = 0

        for preset_name in presets:
            if preset_name not in self.config["ffmpeg_presets"]:
                self.logger.warning(f"Unknown preset: {preset_name}")
                continue

            command = self.generate_ffmpeg_command(preset_name)
            command_str = " ".join(
                f'"{arg}"' if " " in str(arg) else str(arg) for arg in command
            )

            self.logger.info(f"Generating {preset_name} video...")
            self.logger.info(f"Command: {command_str}")

            if dry_run:
                self.logger.info(f"DRY RUN: Would execute: {command_str}")
                success_count += 1
                continue

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if result.returncode == 0:
                    suffix = self.config["ffmpeg_presets"][preset_name]["suffix"]
                    output_file = self.videos_dir / f"timelapse_{suffix}.mp4"
                    if output_file.exists():
                        size_mb = output_file.stat().st_size / (1024 * 1024)
                        self.logger.info(
                            f"Created {preset_name} video ({size_mb:.1f} MB): "
                            f"{output_file.name}"
                        )
                    success_count += 1
                else:
                    self.logger.error(f"ffmpeg failed for preset '{preset_name}':")
                    self.logger.error(f"STDERR: {result.stderr}")
            except subprocess.TimeoutExpired:
                self.logger.error(f"ffmpeg timed out for preset '{preset_name}'")
            except Exception as exc:
                self.logger.error(f"Error running ffmpeg for '{preset_name}': {exc}")

        return success_count == len(presets)

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.videos_dir.exists():
            video_files = sorted(self.videos_dir.glob("*.mp4"))
            stats["videos_created"] = len(video_files)
            stats["video_files"] = [f.name for f in video_files]
        else:
            stats["videos_created"] = 0

        if self.ordered_files:
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
        print(f"Worker Threads:      {stats['max_workers']}")
        print(f"Parallel Threshold:  {stats['parallel_threshold']} files")
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
  python timelapse_creator.py --max_workers 8
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
            "File count above which parallel resolution is used "
            "(default: 10000)"
        ),
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

        if not args.dry_run:
            # Step 3: Render videos
            if not creator.create_videos(presets=args.presets, dry_run=False):
                print("Video creation failed!")
                return 1

        # Step 4: Summary
        creator.print_summary()

        if args.dry_run:
            print("\nffmpeg commands that would be executed:")
            for preset in args.presets:
                command = creator.generate_ffmpeg_command(preset)
                command_str = " ".join(
                    f'"{arg}"' if " " in str(arg) else str(arg) for arg in command
                )
                print(f"  {preset}: {command_str}")

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
