import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import keyframe_sampler
from services.timeline_types import Candidate, RankedCandidate


class KeyframeSamplerTests(unittest.TestCase):
    def test_frame_times_cover_start_hook_anchor_and_payoff(self) -> None:
        frames = keyframe_sampler._frame_times_for_candidate(10.0, 30.0, 18.0, 4)

        self.assertEqual([label for label, _ in frames], ["start", "hook", "anchor", "payoff"])
        self.assertGreaterEqual(frames[0][1], 10.0)
        self.assertLessEqual(frames[-1][1], 30.0)
        self.assertEqual(frames[2][1], 18.0)

    def test_sample_candidate_keyframes_writes_expected_manifest(self) -> None:
        ranked = [
            RankedCandidate(
                candidate=Candidate(id="reaction:0:200", start=0.0, end=12.0, source="reaction", anchor_time=3.0),
                features=None,
                rank_score=5.0,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp) / "debug"

            def fake_extract(_video_path: Path, _time_sec: float, output_path: Path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"jpeg")

            with patch("services.keyframe_sampler._extract_frame", side_effect=fake_extract):
                sampled = keyframe_sampler.sample_candidate_keyframes(
                    Path("fake.mp4"),
                    ranked,
                    debug_dir,
                    3,
                )

        self.assertIn("reaction:0:200", sampled)
        self.assertEqual(len(sampled["reaction:0:200"]), 3)
        self.assertTrue(sampled["reaction:0:200"][0].path.endswith("start.jpg"))


if __name__ == "__main__":
    unittest.main()
