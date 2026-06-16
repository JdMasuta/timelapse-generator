# Phase 1 — Encode-Pipeline Investigation Report

> Measure the gate first, then implement only what the measurements justify.

This report documents what was measured **before** any optimization code was
written. The investigation harness was run in a Linux CI container (4 vCPU,
15 GiB RAM, **no GPU**, FFmpeg 6.1.1). A synthetic but realistic dataset was
used: **300 frames @ 1920×1080, mean 599 KiB/JPEG** (photographic entropy, with
the slow frame-to-frame drift typical of a real timelapse). The manifest was
deliberately shuffled to prove chronological sorting works.

**What I could run here:** framerate-correctness verification, the
decode-vs-encode (CPU/x264) benchmark, decode/encode parallel-scaling,
intermediate-container stitch correctness, and GPU *presence-vs-runtime*
detection. **What only the user can run:** the same benchmark on their real
captures and, crucially, anything involving a *working* NVENC GPU (this
container has none). The tool now ships a `--benchmark` mode so the user can
reproduce every number below on their Windows box in one command.

---

## A. Framerate / timing correctness — **BUG CONFIRMED**

The current command passes `-framerate F` **after** `-i`. With the concat
demuxer feeding stills, that token is an *output* option that libx264/mp4 do
not honor, so it is silently ignored and the output inherits the concat
demuxer's default **25 fps**. Measured on a 120-frame slice, target 24 fps
(expected duration 5.000 s):

| variant | frames out | avg fps | duration | verdict |
|---|---|---|---|---|
| **V0 current** — `-framerate` after `-i` | 120 | **25.000** | **4.800 s** | ❌ WRONG (ignored) |
| V1 — input `-r` **before** `-i` | 120 | 24.000 | 5.000 s | ✅ correct |
| V2 — output `-r` after `-i` | **117** | 24.000 | 4.875 s | ❌ **drops 3 frames** |
| V3 — input `-framerate` before `-i` | — | — | — | ❌ errors (concat has no such option) |
| V4 — per-entry `duration` + output `-r` | 120 | 24.000 | 5.000 s | ✅ correct (but 2× list size) |

**Impact:** every video the tool produces today plays at **25 fps regardless of
preset**. The per-preset framerate is currently a no-op — `preview` (intended
60) plays 2.4× too slow, `construction` (intended 20) plays 1.25× too fast, etc.
Frame *count* is preserved; playback *rate/duration* is wrong for all five
presets.

**Fix (implemented in Phase 2):** input `-r F` **before** `-i` (V1). It is the
only option that preserves the exact frame count *and* sets the exact rate
without doubling the concat list. V2 is explicitly rejected because it resamples
and **drops frames** — which would also corrupt chunk seams. This same input
`-r` is used for every chunk in the parallel path so seams stay frame-exact.

---

## B. Decode vs encode — **ENCODE-BOUND on this CPU**

Stage isolation, 300 frames, single process (decode ceiling via the null
muxer = `wrapped_avframe` passthrough, no real encode):

| stage | throughput |
|---|---|
| decode only (`-f null`, ceiling) | **81.9 fps** |
| decode + `yuvj420p→yuv420p` range convert | 74.4 fps (≈ 9% cost) |
| decode + scale → 1280×720 | 76.3 fps |
| full encode x264 **preview** (CRF 23) | 16.2 fps |
| full encode x264 **high_quality** (CRF 15) | 8.1 fps |
| full encode x264 **construction** (CRF 10) | 7.3 fps |

**Decode-ceiling ÷ full-encode** (>1.5 ⇒ encode-bound):

| preset | ratio | verdict |
|---|---|---|
| preview (CRF 23) | 5.06× | **encode-bound** |
| high_quality (CRF 15) | 10.11× | **encode-bound** |
| construction (CRF 10) | 11.27× | **encode-bound** |

The task's working hypothesis ("JPEG decode is the gate") is **false on this
CPU**. x264 at the low-CRF presets is 5–11× slower than decode. The
`yuvj420p→yuv420p` conversion is real but minor (~9%); scaling down is
effectively free relative to encode.

> Caveat on transfer: absolute fps scales with CPU and resolution. But the
> *qualitative* result — CRF 10/15 x264 ≫ slower than JPEG decode — is robust,
> because those are extremely heavy quantizers. Expect the user's box to be
> encode-bound at `construction`/`high_quality` too; `--benchmark` confirms it
> on their hardware.

---

## C. When does parallelism help? — **measured**

**Decode scales ~linearly with concurrent processes** (300 frames split into K
contiguous null-decode jobs):

| processes | aggregate decode fps | scaling |
|---|---|---|
| 1 | 81.7 | 1.00× |
| 2 | 159.4 | 1.95× |
| 4 | 284.3 | 3.48× |

**Parallel *encode* on this 4-core box is marginal** because one x264 already
saturates the cores. Chunked into N processes, encoded to intermediates,
stitched with `-c copy` (240-frame set, high_quality):

| config | wall | speedup | seams (Σchunks→final) |
|---|---|---|---|
| single proc (x264 auto threads) | 30.8 s | 1.00× | — |
| 2 proc × 2 thr (mpegts) | 30.2 s | 1.02× | 240→240 ✅ |
| 4 proc × 1 thr (mpegts) | 29.7 s | 1.04× | 240→240 ✅ |
| 4 proc × 1 thr (**mp4**) | 29.0 s | 1.06× | 240→240 ✅ |
| 3 proc × 1 thr | 39.5 s | 0.78× | 240→240 ✅ (uneven split underutilizes) |

For the lighter `preview` (CRF 23) preset, parallel encode reaches ~1.10×
because a single x264 leaves a little headroom there.

**Interpretation.** On a CPU where one x264 already uses every core, splitting
into more *encode* processes adds ~nothing (but costs nothing and stays
seam-correct). The parallel-chunk win is large precisely when:
1. the per-process gate is **decode** (scales ~linearly — up to 3.5× on 4 cores
   here), e.g. fast presets, faster CPUs, or GPU encode;
2. the encoder **doesn't** saturate all cores — true on **high-core-count
   machines**, where one x264's frame-threading plateaus (typically past
   ~8–16 threads) and several narrower instances recover the lost efficiency;
3. **GPU encode** — concurrent NVENC sessions multiply encode throughput and
   the parallel CPU decode keeps them fed.

This is why the worker policy is "**fewer, wider** x264 instances" rather than
"many narrow ones": auto = *physical-core* workers, each x264 given
`logical÷physical` threads, so total threads ≈ logical cores with no
oversubscription. The 3-process row above shows the failure mode of an *uneven*
split underutilizing cores — the implementation uses near-equal contiguous
ranges.

---

## D. Hardware capability — **presence ≠ working runtime (demonstrated)**

This FFmpeg build **lists** GPU encoders:

```
h264_nvenc, hevc_nvenc, av1_nvenc, h264_qsv, hevc_qsv, h264_vaapi, hevc_vaapi
```

…but `nvidia-smi` is absent and **every one fails the 1-frame dry-encode init**:

```
h264_nvenc -> FAILS: Cannot load libcuda.so.1
h264_qsv   -> FAILS: Error creating a MFX session: -9
```

This is exactly the trap the task warns about: **the encoder list is a
compile-time fact, not a runtime guarantee.** Phase 2's `--encoder auto`
therefore selects a GPU encoder **only after a real 1-frame dry encode
succeeds**, and surfaces the specific init error (e.g. `Cannot load
libcuda.so.1`) on failure rather than a generic message. In this container the
only working path is x264 — which the tool correctly falls back to.

**The user must run the GPU benchmark themselves** (commands below); none of the
NVENC numbers could be produced here.

---

## E. Recommendation for Phase 2

1. **Fix the framerate** with input `-r` (V1) — correctness, applies everywhere.
2. **Parallel chunked encoding** is the portable workhorse: implement it,
   default off (`--workers 1` keeps today's single-stream output), recommend
   `--workers auto`. It is seam-exact and helps most on many-core/GPU/decode-
   bound configs — i.e. the user's real machine, not this 4-core container.
3. **`--encoder auto/nvenc/hevc_nvenc`**, gated on a runtime dry-encode probe,
   x264 default. On the user's encode-bound low-CRF presets a working NVENC is
   likely the single biggest win — at a documented fidelity-per-bit cost that
   matters most at `construction` (10) and `high_quality` (15).
4. **`--scale`** is cheap and lightens every later stage (CPU scale).
5. **No NVDEC/`mjpeg_cuvid`** for the concat-of-JPEGs path — couldn't be shown
   to work here and is fragile (per-file decoder init); CPU decode + parallel
   chunks is the workhorse. Deliberately omitted (see Phase 2 summary).

### Commands for the user to reproduce on Windows (RTX box)

```bat
:: 0. Bake-in benchmark on real captures (decode ceiling vs encode, GPU probe)
python timelapse_creator.py --source_dir C:\...\captures --output_dir C:\...\out ^
    --benchmark --presets high_quality construction

:: 1. Decode ceiling (max possible, zero encode)
ffmpeg -f concat -safe 0 -i concat_list.txt -f null -

:: 2. Does NVENC actually initialize? (presence != works)
ffmpeg -hide_banner -encoders | findstr nvenc
ffmpeg -y -f lavfi -i color=black:s=320x240 -frames:v 1 -c:v h264_nvenc nul

:: 3. x264 vs NVENC full-encode fps for the encode-bound preset
ffmpeg -r 24 -f concat -safe 0 -i concat_list.txt -c:v libx264   -crf 15 -pix_fmt yuv420p -f null -
ffmpeg -r 24 -f concat -safe 0 -i concat_list.txt -c:v h264_nvenc -rc vbr -cq 15 -b:v 0 -pix_fmt yuv420p -f null -
```

---

## Addendum — real-hardware validation (Win11, 8C/16T, working NVENC)

Running `--benchmark` on the real `maximus_teardown` set (9,168 frames, 1080p)
refined two things:

1. **Cold vs warm decode.** The Phase-1 "decode ceiling" of ~78–82 fps was
   **warm-cache** (those frames had already been read into the OS page cache by
   earlier runs). The realistic **cold, one-pass** decode — every frame read
   from disk exactly once, as in a real job — is far lower: **~17 fps** in the
   container and **~12.6 fps** on the user's machine. So both machines are only
   **~2–2.5× encode-bound on cold reads**, not 5–11×. The earlier ratios were an
   artifact of measuring warm decode against (CPU-bound) encode.

2. **NVENC works on the target box** (`h264_nvenc` initialized), which flips the
   recommendation: because NVENC makes encode nearly free, total throughput
   becomes the **parallel cold-decode** rate. A single NVENC stream would sit
   decode-bound at ~12.6 fps (the exact "GPU idles waiting for decode" trap from
   the brief) — so the win comes from **NVENC + parallel chunks**, which keep
   several cold decoders feeding the GPU.

**Tooling change:** `--benchmark` now reports the **cold** ceiling and measures
**parallel decode scaling on fresh cold blocks** (never mixing cold/warm), and
the recommendation keys off it.

## Addendum 2 — 4K reality check (what the first real encode taught us)

The first full run on the real data (3840×2160, **3.5 MiB/frame**, 9,168 frames)
exposed three things the synthetic/CPU-only testing could not:

1. **The low decode ceiling is CPU, not I/O.** Warm ≈ cold (12.3 ≈ 12.7 fps) on
   a Windows Dev Drive — it is simply expensive to JPEG-decode 4K frames.
   Decode *does* parallelize (5.6× at 8 procs), so the I/O-bound hypothesis was
   wrong here; the gate is CPU decode + encode.

2. **A decode-ceiling estimate over-predicts NVENC badly.** The benchmark
   recommended `nvenc --workers 8` expecting ~77 fps; the real run managed
   **4.2 fps** (slower than single-process x264) with a 4.4 GB file — 8
   concurrent 4K NVENC sessions *contend* on the GPU rather than scale, and the
   uncapped CPU-side decoders oversubscribed the cores. Fixes:
   - the benchmark now **times real parallel encodes** (x264 across cores, NVENC
     at low *and* high worker counts) and recommends the measured winner, so it
     cannot over-promise NVENC again;
   - GPU workers now get a **capped decode-thread budget** (`logical/workers` on
     the input side) so N concurrent decoders stop oversubscribing the CPU;
   - for high-res, **fewer NVENC sessions** (2–4) generally beat 8.

3. **A correctness bug: 6 dropped frames at 20 fps (`construction`).** The
   post-stitch seam check caught it and refused to stitch (working as designed),
   but the cause was vsync dropping frames during rate handling. Fixed by
   encoding every chunk and the single-process path with **`-fps_mode
   passthrough`**, which passes each decoded frame through with its `-r`-derived
   PTS — output frames now equal input frames by construction.
