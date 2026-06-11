import dataclasses
import unittest
from pathlib import Path

from producers.audio import AudioResource
from producers.energy import VoiceEnergy

AUDIO = Path("/jobs/job/audio.wav")


@dataclasses.dataclass
class _Seg:
    """A torch-free stand-in for RichSegment — VoiceEnergy only needs a dataclass
    carrying ``start`` and the ``voice_energy_burst_score`` field it stamps."""

    start: float
    voice_energy_burst_score: float = 0.0


def _segs() -> list[_Seg]:
    return [_Seg(start=0.0), _Seg(start=2.0)]


class _FakeAudio:
    """Stand-in for AudioResource: counts ``path()`` pulls."""

    def __init__(self, result: Path | None = AUDIO) -> None:
        self._result = result
        self.calls = 0

    def path(self) -> Path | None:
        self.calls += 1
        return self._result


class _CountingExtract:
    """Call-counting ExtractFn so the shared-resource reuse assertion can bite."""

    def __init__(self, result: Path | None = AUDIO) -> None:
        self._result = result
        self.calls = 0

    def __call__(self, video_path: Path, job_id: str) -> Path | None:
        self.calls += 1
        return self._result


class _UnusedCache:
    def load(self, *_args, **_kwargs):
        raise AssertionError("cache must not be touched when cache_source is None")

    def store(self, *_args, **_kwargs):
        raise AssertionError("cache must not be touched when cache_source is None")


class VoiceEnergyTests(unittest.TestCase):
    def test_cache_hit_stamps_without_pulling_audio_or_computing(self) -> None:
        audio = _FakeAudio()
        save_calls: list = []
        compute_calls: list = []
        ve = VoiceEnergy(
            audio=audio,
            cache_source="twitch_vod:1",
            load=lambda _src, _segs: [0.4, 0.8],
            save=lambda *a: save_calls.append(a),
            compute=lambda *a: compute_calls.append(a) or [],
        )

        out = ve.apply(_segs())

        self.assertEqual([s.voice_energy_burst_score for s in out], [0.4, 0.8])
        self.assertEqual(audio.calls, 0)  # cache hit → audio never pulled
        self.assertEqual(compute_calls, [])  # no compute on a hit
        self.assertEqual(save_calls, [])  # nothing to re-save on a hit

    def test_cache_miss_pulls_shared_audio_computes_then_saves(self) -> None:
        audio = _FakeAudio(AUDIO)
        saved: list = []
        ve = VoiceEnergy(
            audio=audio,
            cache_source="twitch_vod:1",
            load=lambda _src, _segs: None,
            save=lambda src, segs, scores: saved.append((src, scores)),
            compute=lambda path, segs: [0.1, 0.9],
        )

        out = ve.apply(_segs())

        self.assertEqual([s.voice_energy_burst_score for s in out], [0.1, 0.9])
        self.assertEqual(audio.calls, 1)
        self.assertEqual(saved, [("twitch_vod:1", [0.1, 0.9])])

    def test_no_audio_returns_transcript_unchanged(self) -> None:
        audio = _FakeAudio(None)  # nothing to extract → no energy obtainable
        segs = _segs()
        ve = VoiceEnergy(
            audio=audio,
            cache_source=None,
            load=lambda _src, _segs: None,
            save=lambda *_a: self.fail("must not save without audio"),
            compute=lambda *_a: self.fail("must not compute without audio"),
        )

        out = ve.apply(segs)

        self.assertIs(out, segs)  # returned unchanged, not rebuilt
        self.assertEqual(audio.calls, 1)

    def test_reuses_shared_audio_resource_without_re_extracting(self) -> None:
        # The headline invariant: VoiceEnergy pulls the SAME AudioResource that
        # transcription already pulled — extraction happens at most once per job.
        extract = _CountingExtract(AUDIO)
        audio = AudioResource(
            video_path=Path("/v.mp4"),
            job_id="job",
            cache_source=None,
            extract=extract,
            cache=_UnusedCache(),
        )
        audio.path()  # transcription's cache-miss pull
        self.assertEqual(extract.calls, 1)

        ve = VoiceEnergy(
            audio=audio,
            cache_source=None,
            load=lambda _src, _segs: None,
            save=lambda *_a: None,
            compute=lambda _path, _segs: [0.5, 0.5],
        )
        out = ve.apply(_segs())

        self.assertEqual(extract.calls, 1)  # no second extraction
        self.assertEqual([s.voice_energy_burst_score for s in out], [0.5, 0.5])

    def test_empty_compute_result_leaves_transcript_unchanged(self) -> None:
        audio = _FakeAudio(AUDIO)
        ve = VoiceEnergy(
            audio=audio,
            cache_source=None,
            load=lambda _src, _segs: None,
            save=lambda *_a: self.fail("must not save an empty score list"),
            compute=lambda _path, _segs: [],
        )
        segs = _segs()

        out = ve.apply(segs)

        self.assertIs(out, segs)


if __name__ == "__main__":
    unittest.main()
