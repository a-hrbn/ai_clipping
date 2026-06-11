import json
import tempfile
import unittest
from pathlib import Path

from producers.adapters import FsAudioCache
from producers.audio import AudioResource


class _FakeExtract:
    """Call-counting extractor: ``extract.calls == 1`` is the load-bearing
    assertion for the keystone memoization."""

    def __init__(self, result: Path | None) -> None:
        self._result = result
        self.calls = 0
        self.seen: list[tuple] = []

    def __call__(self, video_path: Path, job_id: str) -> Path | None:
        self.calls += 1
        self.seen.append((video_path, job_id))
        return self._result


def _tripwire_extract(*_args, **_kwargs):
    raise AssertionError("extract must not be called on a cache hit")


class _RecordingCache:
    def __init__(self, hit: Path | None = None) -> None:
        self._hit = hit
        self.loads: list[tuple] = []
        self.stores: list[tuple] = []

    def load(self, cache_source, *, video_path, job_id):
        self.loads.append((cache_source, video_path, job_id))
        return self._hit

    def store(self, cache_source, *, video_path, audio_path):
        self.stores.append((cache_source, video_path, audio_path))


class AudioResourceTests(unittest.TestCase):
    def test_path_extracts_at_most_once_across_consumers(self) -> None:
        extract = _FakeExtract(Path("/jobs/job/audio.wav"))
        cache = _RecordingCache(hit=None)
        audio = AudioResource(
            video_path=Path("/jobs/job/video.mp4"),
            job_id="job",
            cache_source="twitch_vod:1",
            extract=extract,
            cache=cache,
        )

        first = audio.path()
        second = audio.path()

        self.assertEqual(first, Path("/jobs/job/audio.wav"))
        self.assertEqual(second, first)
        self.assertEqual(extract.calls, 1)
        self.assertEqual(len(cache.loads), 1)
        self.assertEqual(len(cache.stores), 1)

    def test_cache_hit_returns_cached_without_extracting(self) -> None:
        cache = _RecordingCache(hit=Path("/jobs/job/audio.wav"))
        audio = AudioResource(
            video_path=Path("/jobs/job/video.mp4"),
            job_id="job",
            cache_source="twitch_vod:1",
            extract=_tripwire_extract,
            cache=cache,
        )

        self.assertEqual(audio.path(), Path("/jobs/job/audio.wav"))
        self.assertEqual(audio.path(), Path("/jobs/job/audio.wav"))
        self.assertEqual(len(cache.loads), 1)
        self.assertEqual(cache.stores, [])

    def test_cache_miss_extracts_then_stores(self) -> None:
        extract = _FakeExtract(Path("/jobs/job/audio.wav"))
        cache = _RecordingCache(hit=None)
        audio = AudioResource(
            video_path=Path("/jobs/job/video.mp4"),
            job_id="job",
            cache_source="twitch_vod:1",
            extract=extract,
            cache=cache,
        )

        audio.path()

        self.assertEqual(extract.calls, 1)
        self.assertEqual(len(cache.stores), 1)
        _source, _video, stored_audio = cache.stores[0]
        self.assertEqual(stored_audio, Path("/jobs/job/audio.wav"))

    def test_cache_source_none_skips_cache_and_extracts_once(self) -> None:
        extract = _FakeExtract(Path("/jobs/job/audio.wav"))
        cache = _RecordingCache(hit=Path("/should/not/be/used.wav"))
        audio = AudioResource(
            video_path=Path("/jobs/job/video.mp4"),
            job_id="job",
            cache_source=None,
            extract=extract,
            cache=cache,
        )

        self.assertEqual(audio.path(), Path("/jobs/job/audio.wav"))
        audio.path()
        self.assertEqual(extract.calls, 1)
        self.assertEqual(cache.loads, [])
        self.assertEqual(cache.stores, [])

    def test_materialized_reflects_resolution_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"audio")
            audio = AudioResource(
                video_path=Path(tmp) / "video.mp4",
                job_id="job",
                cache_source=None,
                extract=_FakeExtract(wav),
                cache=_RecordingCache(),
            )

            self.assertIsNone(audio.materialized())
            audio.path()
            self.assertEqual(audio.materialized(), wav)

    def test_teardown_removes_wav_and_chunks_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            wav = job_dir / "audio.wav"
            wav.write_bytes(b"audio")
            chunks = job_dir / "chunks"
            chunks.mkdir()
            (chunks / "chunk0.wav").write_bytes(b"chunk")

            audio = AudioResource(
                video_path=job_dir / "video.mp4",
                job_id="job",
                cache_source=None,
                extract=_FakeExtract(wav),
                cache=_RecordingCache(),
            )
            audio.path()
            audio.teardown()

            self.assertFalse(wav.exists())
            self.assertFalse(chunks.exists())
            self.assertIsNone(audio.materialized())
            audio.teardown()  # idempotent — no raise


class FsAudioCacheTests(unittest.TestCase):
    def _make(self, tmp: str, *, duration: float | None = 120.0) -> FsAudioCache:
        return FsAudioCache(
            tmp,
            cache_key=lambda _source: "dig",
            probe=lambda _path: duration,
        )

    def test_store_then_load_roundtrips_into_job_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make(tmp)
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            source_wav = Path(tmp) / "src.wav"
            source_wav.write_bytes(b"x" * 200)

            cache.store("twitch_vod:1", video_path=video, audio_path=source_wav)
            target = cache.load("twitch_vod:1", video_path=video, job_id="job")

            self.assertIsNotNone(target)
            self.assertEqual(target, Path(tmp) / "job" / "audio.wav")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), b"x" * 200)

    def test_stale_metadata_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            source_wav = Path(tmp) / "src.wav"
            source_wav.write_bytes(b"x" * 200)

            self._make(tmp, duration=120.0).store(
                "twitch_vod:1", video_path=video, audio_path=source_wav
            )
            # A later run probes a different duration → metadata no longer matches.
            target = self._make(tmp, duration=999.0).load(
                "twitch_vod:1", video_path=video, job_id="job"
            )

            self.assertIsNone(target)

    def test_corrupt_metadata_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make(tmp)
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            cache_wav = Path(tmp) / "_audio_cache" / "dig" / "audio.wav"
            cache_wav.parent.mkdir(parents=True)
            cache_wav.write_bytes(b"x" * 200)
            cache_wav.with_name("audio_metadata.json").write_text("not json", encoding="utf-8")

            self.assertIsNone(cache.load("twitch_vod:1", video_path=video, job_id="job"))

    def test_empty_wav_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make(tmp)
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            cache_wav = Path(tmp) / "_audio_cache" / "dig" / "audio.wav"
            cache_wav.parent.mkdir(parents=True)
            cache_wav.write_bytes(b"x" * 10)  # <= 44-byte empty-wav guard
            cache_wav.with_name("audio_metadata.json").write_text(
                json.dumps({"version": 1}), encoding="utf-8"
            )

            self.assertIsNone(cache.load("twitch_vod:1", video_path=video, job_id="job"))

    def test_no_cache_key_disables_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FsAudioCache(tmp, cache_key=lambda _s: None, probe=lambda _p: 1.0)
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            self.assertIsNone(cache.load(None, video_path=video, job_id="job"))
            # store is a no-op (no key) and must not raise
            cache.store(None, video_path=video, audio_path=video)


if __name__ == "__main__":
    unittest.main()
