import unittest
from pathlib import Path

from producers.transcript import TranscriptResource

# Segments are opaque to TranscriptResource (it only shuttles the list around),
# so the tests use plain sentinels — keeping this module torch/whisper-free.
CACHED_SEGS = ["cached-0", "cached-1"]
FRESH_SEGS = ["fresh-0", "fresh-1"]

VIDEO = Path("/jobs/job/video.mp4")
AUDIO = Path("/jobs/job/audio.wav")


def _tripwire(*_args, **_kwargs):
    raise AssertionError("must not be called")


class _FakeAudio:
    """Stand-in for AudioResource: counts ``path()`` pulls; can trip a wire so a
    cache-hit run that touches audio fails loudly."""

    def __init__(self, result: Path | None = AUDIO, *, tripwire: bool = False) -> None:
        self._result = result
        self._tripwire = tripwire
        self.calls = 0

    def path(self) -> Path | None:
        if self._tripwire:
            raise AssertionError("AudioResource must not be pulled on a cache hit")
        self.calls += 1
        return self._result


class _RecordingTranscribe:
    def __init__(self, result: list) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, audio_path, progress_cb=None, *, cache_key=None,
                 isolate_chunks=False, enable_diarization=True):
        self.calls.append({
            "audio_path": audio_path,
            "cache_key": cache_key,
            "isolate_chunks": isolate_chunks,
            "enable_diarization": enable_diarization,
        })
        return self.result


class _RecordingLoadCached:
    def __init__(self, result: list | None) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, cache_source, *, expected_duration_sec, progress_cb=None):
        self.calls.append({
            "cache_source": cache_source,
            "expected_duration_sec": expected_duration_sec,
        })
        return self.result


def _make(audio, *, cache_source="twitch_vod:1", probe=lambda _p: 120.0,
          load_cached=None, transcribe=None, isolate_chunks=False,
          enable_diarization=True) -> TranscriptResource:
    return TranscriptResource(
        audio=audio,
        cache_source=cache_source,
        video_path=VIDEO,
        isolate_chunks=isolate_chunks,
        enable_diarization=enable_diarization,
        load_cached=load_cached if load_cached is not None else _RecordingLoadCached(None),
        transcribe=transcribe if transcribe is not None else _RecordingTranscribe(FRESH_SEGS),
        probe=probe,
    )


class TranscriptResourceTests(unittest.TestCase):
    def test_cache_hit_returns_cached_without_pulling_audio(self) -> None:
        audio = _FakeAudio(tripwire=True)
        load_cached = _RecordingLoadCached(CACHED_SEGS)
        tr = _make(audio, load_cached=load_cached, transcribe=_tripwire)

        self.assertEqual(tr.segments(), CACHED_SEGS)
        self.assertTrue(tr.from_cache())
        self.assertEqual(audio.calls, 0)
        self.assertEqual(len(load_cached.calls), 1)
        self.assertEqual(load_cached.calls[0]["expected_duration_sec"], 120.0)

    def test_cache_miss_pulls_audio_then_transcribes(self) -> None:
        audio = _FakeAudio(AUDIO)
        transcribe = _RecordingTranscribe(FRESH_SEGS)
        tr = _make(
            audio,
            load_cached=_RecordingLoadCached(None),
            transcribe=transcribe,
            isolate_chunks=True,
            enable_diarization=False,
        )

        self.assertEqual(tr.segments(), FRESH_SEGS)
        self.assertFalse(tr.from_cache())
        self.assertEqual(audio.calls, 1)
        self.assertEqual(len(transcribe.calls), 1)
        call = transcribe.calls[0]
        self.assertEqual(call["audio_path"], AUDIO)
        self.assertEqual(call["cache_key"], "twitch_vod:1")
        self.assertTrue(call["isolate_chunks"])
        self.assertFalse(call["enable_diarization"])

    def test_missing_probe_is_a_miss_without_consulting_cache(self) -> None:
        audio = _FakeAudio(AUDIO)
        load_cached = _RecordingLoadCached(CACHED_SEGS)  # would HIT if consulted
        transcribe = _RecordingTranscribe(FRESH_SEGS)
        tr = _make(audio, probe=lambda _p: None, load_cached=load_cached, transcribe=transcribe)

        self.assertEqual(tr.segments(), FRESH_SEGS)
        self.assertEqual(load_cached.calls, [])  # a missing probe never touches the cache
        self.assertEqual(audio.calls, 1)
        self.assertEqual(len(transcribe.calls), 1)

    def test_no_cache_source_skips_probe_and_cache(self) -> None:
        audio = _FakeAudio(AUDIO)
        transcribe = _RecordingTranscribe(FRESH_SEGS)
        tr = _make(
            audio,
            cache_source=None,
            probe=_tripwire,
            load_cached=_tripwire,
            transcribe=transcribe,
        )

        self.assertEqual(tr.segments(), FRESH_SEGS)
        self.assertEqual(audio.calls, 1)
        self.assertEqual(len(transcribe.calls), 1)

    def test_progress_is_reported_as_a_fraction(self) -> None:
        fractions: list[float] = []

        def transcribe(audio_path, progress_cb=None, **_kwargs):
            progress_cb(1, 4, None)
            progress_cb(4, 4, None)
            return FRESH_SEGS

        tr = _make(_FakeAudio(AUDIO), load_cached=_RecordingLoadCached(None), transcribe=transcribe)
        tr.segments(progress=fractions.append)

        self.assertEqual(fractions, [0.25, 1.0])

    def test_segments_are_memoized(self) -> None:
        transcribe = _RecordingTranscribe(FRESH_SEGS)
        tr = _make(_FakeAudio(AUDIO), load_cached=_RecordingLoadCached(None), transcribe=transcribe)

        first = tr.segments()
        second = tr.segments()

        self.assertIs(first, second)
        self.assertEqual(len(transcribe.calls), 1)


if __name__ == "__main__":
    unittest.main()
