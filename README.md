# Timelapse Creator

Builds timelapse videos directly from timestamped capture images via ffmpeg's
**concat demuxer** — no sequential file-renaming step. Ordering comes from a
`manifest.json` (`{ "<filename>": { "timestamp": "<ISO-8601 UTC>" }, ... }`)
located in the source directory.

```
python timelapse_creator.py --source_dir ./captures --output_dir ./out \
    --presets high_quality
```

## Optimized encode pipeline

The encode pipeline (`JPEG decode → scale/convert → H.264 encode`) was profiled
before any optimization was written — see **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**
for the methodology and numbers. Headline findings:

* The pipeline is **encode-bound** at the low-CRF presets (x264 was 5–11× slower
  than JPEG decode), so parallel encode / GPU offload is the lever — *not* GPU
  decode.
* The legacy command set `-framerate` **after** `-i`, which the concat demuxer
  ignores: every output played at **25 fps regardless of preset**. This is now
  fixed (see below).

### Behavioral fix: framerate

Framerate is now an **input** option (`-r F` before `-i`). Outputs play at the
preset's intended rate (e.g. `construction` = 20 fps, `preview` = 60 fps) with
the exact frame count and duration. Codec/CRF/pixel-format are unchanged, so
per-frame quality matches prior runs — only the (previously broken) playback
rate is corrected.

### New flags

| flag | default | purpose |
|---|---|---|
| `--encoder {x264,nvenc,hevc_nvenc,auto}` | `x264` | Encoder. GPU is opt-in and **runtime-probed** (a 1-frame dry encode); a forced GPU encoder that can't initialize **aborts loudly**. `auto` uses a working GPU encoder if present, else x264. |
| `--workers {N,auto}` | `1` | Concurrent encode workers. `1` = single process (legacy structure). `>1` splits the ordered frames into contiguous chunks, encodes them concurrently to MPEG-TS intermediates, and stitches losslessly (`-c copy`). `auto` = physical cores (x264) or `min(NVENC session limit, cores)` (GPU). |
| `--threads_per_worker N` | auto | x264 threads per worker (auto = `logical_cores // workers`, avoiding oversubscription). Ignored for NVENC. |
| `--scale WxH` | none | Optional CPU downscale, e.g. `1280x720` (`-1` preserves aspect). Applied early to lighten every later stage. |
| `--trust-manifest` | off | Skip per-file `exists()` checks (the manifest is authoritative); removes ~N `stat()` calls on large sets. |
| `--benchmark` | off | Measure decode-ceiling vs full-encode fps, **plus cold parallel-decode scaling**, print a decode-/encode-bound verdict + a recommended `--encoder`/`--workers`, then exit. |

**Run `--benchmark` first** to get a verdict and recommended settings for *your*
machine and dataset, then turn on speed with `--workers auto` (and `--encoder
nvenc` if you have a working NVIDIA GPU).

The benchmark reports the **cold** (first-read) decode ceiling — the realistic
rate for a one-pass job, since every frame is read once — measures whether
decode **parallelizes across processes** (fresh cold blocks), and then **times
real parallel encodes** (x264 across cores, plus NVENC at low and high worker
counts) and recommends the **measured** winner. Measuring matters: a
decode-ceiling estimate badly over-predicts NVENC at high resolution, where many
concurrent 4K sessions *contend* on the GPU instead of scaling (8× 4K NVENC can
be slower than single-process x264). If decode is flat the run is **I/O-bound**
(slow storage / antivirus per-file scan / OneDrive placeholders) and the
benchmark says so and recommends fixing the source read first.

> **High-resolution + NVENC:** fewer concurrent sessions are usually faster.
> NVENC throughput is per-GPU, not per-session, so 2–4 sessions typically
> saturate the encoder while 8× 4K sessions just contend. Use `--benchmark` to
> find the worker count that actually wins; don't assume more is better.

Every encode uses ffmpeg `-fps_mode passthrough`, so output frame count is
**exactly** the input frame count — no frames are dropped or duplicated during
rate handling (this is enforced again by the post-stitch seam check).

### CRF → NVENC `-cq` mapping

NVENC ignores `-crf`. Each preset's CRF is mapped 1:1 to NVENC `-cq` (same 0–51
scale, lower = higher quality) under VBR with no bitrate cap (`-rc vbr -cq N
-b:v 0 -preset p5 -tune hq -bf 0`). `-bf 0` (no B-frames) is required so the
chunked MPEG-TS intermediates don't drop the final reordered frame of each
segment. The mapping lives in one place (`config['nvenc_cq']` in
`timelapse_creator.py`):

| preset | framerate | x264 CRF | NVENC `-cq` |
|---|---|---|---|
| preview | 60 | 23 | 23 |
| standard | 30 | 17 | 17 |
| high_quality | 24 | 15 | 15 |
| smooth | 45 | 20 | 20 |
| construction | 20 | 10 | 10 |

**Speed vs quality.** NVENC is much faster but **less efficient per bit** than
x264 — at the same number it produces lower fidelity / more bits. This matters
most at the low-CRF presets (`construction` = 10, `high_quality` = 15), which
are the archival-quality ones. x264 stays the default for that reason; GPU is
strictly opt-in. Raise the `-cq` values in the table for smaller GPU files.

### Why MPEG-TS intermediates

Chunks are encoded to MPEG-TS, then stitched to MP4 with `-c copy`. MPEG-TS
concatenates cleanly under stream-copy (clean packet boundaries, no moov/edit-
list to reconcile across segments); the final MP4 gets `+faststart`. Seam
integrity (final frame count == Σ chunk frames == input count) is verified after
every chunked encode and aborts on mismatch.

### Deliberately omitted: GPU JPEG decode

NVDEC/`mjpeg_cuvid` for the concat-of-individual-JPEGs path is **not** used:
per-file decoder init overhead and inconsistent MJPEG-on-NVDEC behavior make it
fragile, and it could not be shown to help in profiling. CPU decode + parallel
chunks is the workhorse. See BENCHMARK_REPORT.md §E.

## Tests

```
python test_timelapse.py     # needs ffmpeg/ffprobe on PATH (Windows OK)
```

Verifies framerate correctness, chunked seam integrity (no dropped/duplicated
frames across boundaries) for several worker counts, single==chunked frame
counts, downscaling, and chunk-range math. Frames are synthesized with ffmpeg
itself, so no extra Python deps are required.
