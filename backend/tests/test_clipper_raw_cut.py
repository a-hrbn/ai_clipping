"""The raw-cut leaf-op ``services.clipper.cut_raw_segment`` — the real ffmpeg cutter the
archetype-retest render adapter (#99) points at ``storage_dir/<arm_job_id>/clips/``.

ffmpeg is a Docker-only leaf, so this stubs the module and asserts ``cut_raw_segment``
issues the SAME ``ss``/``t``/codec invocation ``cut_clips`` does (reusing
``_video_output_kwargs``), just to an explicit output path — never copying the cached video.
"""
from __future__ import annotations

import sys
import types

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))
sys.modules.setdefault("ffmpeg", types.ModuleType("ffmpeg"))

from services import clipper


class _RecordingFfmpeg:
    """A fluent ffmpeg stub that records the one cut invocation it receives."""

    def __init__(self) -> None:
        self.calls: dict = {}

    def input(self, path, **kwargs):
        self.calls["input"] = (path, kwargs)
        return self

    def output(self, path, **kwargs):
        self.calls["output"] = (path, kwargs)
        return self

    def overwrite_output(self):
        self.calls["overwrite"] = True
        return self

    def run(self, **kwargs):
        self.calls["run"] = kwargs
        return None


def test_cut_raw_segment_cuts_the_exact_window_to_the_output_path_reusing_codec_kwargs(
    monkeypatch, tmp_path
) -> None:
    rec = _RecordingFfmpeg()
    monkeypatch.setattr(clipper, "ffmpeg", rec)
    # Force the libx264 fallback so the codec assert is deterministic (no real NVENC probe),
    # while still exercising the REAL _video_output_kwargs — proving cut_raw_segment reuses it.
    monkeypatch.setattr(clipper, "_get_nvenc_available", lambda _vp: False)

    video = tmp_path / "_download_cache" / "twitch" / "vod_v1.mp4"
    out = tmp_path / "archetype_retest_v1_A" / "clips" / "clip_01.mp4"

    clipper.cut_raw_segment(video, 100.0, 120.0, out)

    # Reads the cached video IN PLACE (no copy) and cuts the EXACT [start, end) window via
    # ss/t — the same ffmpeg invocation cut_clips uses.
    assert rec.calls["input"] == (str(video), {"ss": 100.0, "t": 20.0})
    # Writes to the EXPLICIT output path (not video.parent/clips), reusing the shared
    # _video_output_kwargs codec/container settings — here the libx264 fallback.
    assert rec.calls["output"][0] == str(out)
    assert rec.calls["output"][1] == {
        "vcodec": "libx264", "preset": "ultrafast", "acodec": "aac", "movflags": "+faststart"}
    assert rec.calls["overwrite"] is True
    assert rec.calls["run"] == {"quiet": True}
    # The arm's clips dir is created so the cut never fails on a missing parent.
    assert out.parent.is_dir()
