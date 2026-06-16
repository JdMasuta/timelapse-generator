#!/usr/bin/env python3
"""Self-contained correctness tests for timelapse_creator.

Generates a tiny set of distinct JPEGs with ffmpeg itself (no Pillow needed),
builds a shuffled manifest, then drives the real TimelapseCreator to verify:

  * framerate correctness   — output avg fps & duration match the preset
                              (proves the input `-r` fix, not the 25 fps bug)
  * chunked seam integrity  — final frame count == input count == Σ chunks,
                              for several worker counts (no drop/dup at seams)
  * single == chunked       — same frame count both ways
  * chunk range math        — contiguous, gapless, complete coverage

Runnable anywhere ffmpeg/ffprobe are on PATH (Windows included):
    python test_timelapse.py
"""
import json
import random
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from timelapse_creator import TimelapseCreator

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
N_FRAMES = 48


def _gen_frames(captures: Path, n: int) -> None:
    """Create n distinct JPEGs via ffmpeg testsrc2 (frame-numbered overlay)."""
    captures.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=1:duration={n}",
         "-frames:v", str(n), str(captures / "capture_%05d.jpg")],
        check=True)


def _build_manifest(captures: Path) -> None:
    files = sorted(captures.glob("capture_*.jpg"))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest = {}
    order = list(range(len(files)))
    random.Random(42).shuffle(order)  # shuffled => proves chronological sort
    for i in order:
        ts = (start + timedelta(seconds=30 * i)).isoformat().replace("+00:00", "Z")
        manifest[files[i].name] = {"timestamp": ts}
    (captures / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _probe(path: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True)
    data = json.loads(out.stdout)
    rate = data["streams"][0]["avg_frame_rate"]
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return fps, float(data["format"]["duration"])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe required")
class TimelapseCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tl_test_"))
        cls.captures = cls.tmp / "captures"
        _gen_frames(cls.captures, N_FRAMES)
        _build_manifest(cls.captures)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _make(self, outdir, **kw):
        c = TimelapseCreator(
            source_dir=str(self.captures), output_dir=str(outdir),
            tz_offset_hours=0, **kw)
        files = c.load_manifest()
        self.assertEqual(len(files), N_FRAMES)
        # Order must be chronological despite the shuffled manifest.
        self.assertEqual([p.name for p, _ in files],
                         sorted(p.name for p, _ in files))
        self.assertTrue(c.write_concat_list(files))
        return c

    def test_chunk_ranges_cover_contiguously(self):
        for n, k in [(48, 4), (50, 4), (10, 3), (7, 7), (5, 8)]:
            kk = min(k, n)
            r = TimelapseCreator._chunk_ranges(n, kk)
            self.assertEqual(r[0][0], 0)
            self.assertEqual(r[-1][1], n)
            for (a, b), (c, d) in zip(r, r[1:]):
                self.assertEqual(b, c)            # no gap / overlap
            self.assertEqual(sum(b - a for a, b in r), n)  # full coverage

    def test_framerate_matches_preset_single(self):
        c = self._make(self.tmp / "single")
        ok = c.create_videos(presets=["construction"], dry_run=False)
        self.assertTrue(ok)
        out = c.videos_dir / "timelapse_construction.mp4"
        self.assertEqual(c._count_video_frames(out), N_FRAMES)
        fps, dur = _probe(out)
        self.assertAlmostEqual(fps, 20.0, places=2)         # NOT 25 (the bug)
        self.assertAlmostEqual(dur, N_FRAMES / 20.0, delta=0.05)

    def test_chunked_seams_preserve_all_frames(self):
        for workers in (2, 4, 5):
            c = self._make(self.tmp / f"chunk_{workers}",
                           workers=str(workers))
            ok = c.create_videos(presets=["preview"], dry_run=False)
            self.assertTrue(ok, f"workers={workers} encode failed")
            out = c.videos_dir / "timelapse_preview.mp4"
            self.assertEqual(c._count_video_frames(out), N_FRAMES,
                             f"seam drop/dup at workers={workers}")
            fps, dur = _probe(out)
            self.assertAlmostEqual(fps, 60.0, places=1)
            self.assertAlmostEqual(dur, N_FRAMES / 60.0, delta=0.05)

    def test_single_equals_chunked_framecount(self):
        cs = self._make(self.tmp / "eq_single")
        cs.create_videos(presets=["standard"], dry_run=False)
        single = cs._count_video_frames(
            cs.videos_dir / "timelapse_standard.mp4")
        cc = self._make(self.tmp / "eq_chunk", workers="4")
        cc.create_videos(presets=["standard"], dry_run=False)
        chunked = cc._count_video_frames(
            cc.videos_dir / "timelapse_standard.mp4")
        self.assertEqual(single, chunked)
        self.assertEqual(single, N_FRAMES)

    def test_scale_downscales_output(self):
        c = self._make(self.tmp / "scaled", scale="160x120", workers="2")
        c.create_videos(presets=["preview"], dry_run=False)
        out = c.videos_dir / "timelapse_preview.mp4"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True)
        self.assertEqual(probe.stdout.strip().split(",")[:2], ["160", "120"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
