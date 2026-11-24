"""The one public clip route (#124): serves generated clips, 404s everything else."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class ClipRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Path(self._tmp.name)
        clips = self.storage / "job-1" / "clips"
        clips.mkdir(parents=True)
        (clips / "clip_01.mp4").write_bytes(b"fake-mp4")
        review = self.storage / "job-1" / "review_clips"
        review.mkdir()
        (review / "review_01.mp4").write_bytes(b"fake-mp4")
        (self.storage / "job-1" / "debug").mkdir()
        (self.storage / "job-1" / "debug" / "secret.mp4").write_bytes(b"x")
        self._patch = patch.object(main.settings, "storage_dir", str(self.storage))
        self._patch.start()
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def test_serves_an_existing_clip(self) -> None:
        response = self.client.get("/static/job-1/clips/clip_01.mp4")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-mp4")
        self.assertEqual(response.headers["content-type"], "video/mp4")

    def test_missing_clip_is_404(self) -> None:
        self.assertEqual(self.client.get("/static/job-1/clips/clip_02.mp4").status_code, 404)
        self.assertEqual(self.client.get("/static/job-2/clips/clip_01.mp4").status_code, 404)

    def test_invalid_names_are_404(self) -> None:
        for url in (
            "/static/job-1/clips/clip_01.txt",      # not a video extension
            "/static/job-1/clips/.hidden.mp4",      # must start alphanumeric
            "/static/job-1/clips/..%2Fdebug%2Fsecret.mp4",
            "/static/-job/clips/clip_01.mp4",       # bad job id
            "/static/..%2Fjob-1/clips/clip_01.mp4",
            "/static/job-1/debug/secret.mp4",       # only clips/ is addressable
        ):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)

    def test_review_clips_are_no_longer_served(self) -> None:
        self.assertEqual(
            self.client.get("/static/job-1/review_clips/review_01.mp4").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
