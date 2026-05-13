import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda *_args, **_kwargs: None))
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import backfill_clip_judge_snapshots as backfill


class BackfillClipJudgeSnapshotsTests(unittest.TestCase):
    def test_vod_id_from_url_accepts_twitch_urls_and_raw_ids(self) -> None:
        self.assertEqual(backfill._vod_id_from_url("https://www.twitch.tv/videos/2730792911?t=10s"), "2730792911")
        self.assertEqual(backfill._vod_id_from_url("2730792911"), "2730792911")
        self.assertEqual(backfill._vod_id_from_url("not-a-vod"), "")

    def test_planned_vods_dedupes_skips_existing_and_applies_limit(self) -> None:
        urls = [
            "https://www.twitch.tv/videos/2730792911",
            "https://www.twitch.tv/videos/2730792911?t=10s",
            "https://www.twitch.tv/videos/2735012372",
            "https://www.twitch.tv/videos/2739861273",
        ]

        planned = backfill._planned_vods(
            urls,
            skip_vod_ids={"2735012372"},
            limit=1,
        )

        self.assertEqual(
            planned,
            [
                {
                    "vod_id": "2730792911",
                    "vod_url": "https://www.twitch.tv/videos/2730792911",
                    "input_mode": "vod_url",
                }
            ],
        )

    def test_snapshot_vod_ids_reads_candidate_ids_and_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"candidate_id": "twitch_vod_2730792911_12:hype_peak:0:120"}),
                        json.dumps({"source_url": "https://www.twitch.tv/videos/2735012372"}),
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                backfill._snapshot_vod_ids(path),
                {"2730792911", "2735012372"},
            )

    def test_read_vod_manifest_supports_csv_jsonl_and_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "vods.csv"
            csv_path.write_text("vod_id\n2730792911\n", encoding="utf-8")
            jsonl_path = root / "vods.jsonl"
            jsonl_path.write_text(
                json.dumps({"vod_url": "https://www.twitch.tv/videos/2735012372"}) + "\n",
                encoding="utf-8",
            )
            txt_path = root / "vods.txt"
            txt_path.write_text("https://www.twitch.tv/videos/2739861273\n", encoding="utf-8")

            self.assertEqual(backfill._read_vod_manifest(csv_path), ["https://www.twitch.tv/videos/2730792911"])
            self.assertEqual(backfill._read_vod_manifest(jsonl_path), ["https://www.twitch.tv/videos/2735012372"])
            self.assertEqual(backfill._read_vod_manifest(txt_path), ["https://www.twitch.tv/videos/2739861273"])

    def test_discover_vod_urls_explicit_broadcaster_does_not_expand_profile_pool(self) -> None:
        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

        with patch.object(backfill.settings, "twitch_client_id", "client"), \
             patch.object(backfill.settings, "twitch_client_secret", "secret"), \
             patch.object(backfill, "TwitchClient", FakeClient), \
             patch.object(backfill, "_resolved_broadcasters") as resolved_mock, \
             patch.object(backfill, "_get_user", return_value={"id": "123"}), \
             patch.object(backfill, "_get_vods", return_value=[{"id": "2730792911"}]):
            urls = backfill._discover_vod_urls(
                broadcaster="jynxzi",
                profile="jynxzi",
                date_from="2026-03-01",
                date_to="2026-03-02",
                sources=Path("/tmp/unused.csv"),
            )

        self.assertEqual(urls, ["https://www.twitch.tv/videos/2730792911"])
        self.assertFalse(resolved_mock.called)

    def test_source_vod_urls_returns_empty_for_empty_discovery(self) -> None:
        args = types.SimpleNamespace(
            vod_url=[],
            vod_manifest="",
            date_from="2026-03-01",
            date_to="2026-03-02",
            broadcaster="jynxzi",
            profile="jynxzi",
            sources="/tmp/unused.csv",
            fallback_label_manifest=False,
            label_manifest="/tmp/labels.jsonl",
        )
        with patch.object(backfill, "_discover_vod_urls", return_value=[]):
            self.assertEqual(backfill._source_vod_urls(args), [])

    def test_label_manifest_fallback_recovers_old_date_range_vods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels.jsonl"
            labels.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "broadcaster": "jynxzi",
                                "vod_id": "2711274372",
                                "vod_url": "https://www.twitch.tv/videos/2711274372",
                                "created_at": "2026-03-01T12:00:00Z",
                            }
                        ),
                        json.dumps(
                            {
                                "broadcaster": "caseoh_",
                                "vod_id": "9999999999",
                                "created_at": "2026-03-01T12:00:00Z",
                            }
                        ),
                        json.dumps(
                            {
                                "broadcaster": "jynxzi",
                                "vod_id": "2720000000",
                                "created_at": "2026-03-03T12:00:00Z",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            urls = backfill._discover_vod_urls_from_label_manifest(
                labels,
                broadcaster="jynxzi",
                date_from="2026-03-01",
                date_to="2026-03-02",
            )

        self.assertEqual(urls, ["https://www.twitch.tv/videos/2711274372"])

    def test_source_vod_urls_uses_label_manifest_fallback_after_empty_twitch_discovery(self) -> None:
        args = types.SimpleNamespace(
            vod_url=[],
            vod_manifest="",
            date_from="2026-03-01",
            date_to="2026-03-02",
            broadcaster="jynxzi",
            profile="jynxzi",
            sources="/tmp/unused.csv",
            fallback_label_manifest=True,
            label_manifest="/tmp/labels.jsonl",
        )
        with patch.object(backfill, "_discover_vod_urls", return_value=[]), \
             patch.object(
                 backfill,
                 "_discover_vod_urls_from_label_manifest",
                 return_value=["https://www.twitch.tv/videos/2711274372"],
             ) as fallback_mock:
            urls = backfill._source_vod_urls(args)

        self.assertEqual(urls, ["https://www.twitch.tv/videos/2711274372"])
        self.assertTrue(fallback_mock.called)

    def test_cached_windows_by_vod_repairs_container_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=backfill.BACKEND_ROOT / "tmp") as tmp:
            root = Path(tmp)
            media = root / "vod_feature_cache" / "2711274372" / "scoped_windows" / "abc123" / "source.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"fake mp4")
            container_path = "/app/backend/" + str(media.relative_to(backfill.BACKEND_ROOT))
            manifest = root / "cached_vods.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "vod_id": "2711274372",
                        "vod_url": "https://www.twitch.tv/videos/2711274372",
                        "video_path": container_path,
                        "source_start_offset": 120.0,
                        "source_end_offset": 180.0,
                        "scoped_window_id": "abc123",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = backfill._cached_windows_by_vod(manifest, vod_ids={"2711274372"})

        self.assertEqual(list(rows), ["2711274372"])
        self.assertEqual(rows["2711274372"][0]["video_path"], str(media))
        self.assertEqual(rows["2711274372"][0]["input_mode"] if "input_mode" in rows["2711274372"][0] else None, None)

    def test_expand_cached_windows_prefers_local_video_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=backfill.BACKEND_ROOT / "tmp") as tmp:
            root = Path(tmp)
            media = root / "source.mp4"
            media.write_bytes(b"fake mp4")
            manifest = root / "cached_vods.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "vod_id": "2711274372",
                                "video_path": str(media),
                                "source_start_offset": 10.0,
                                "source_end_offset": 30.0,
                                "scoped_window_id": "window_a",
                            }
                        ),
                        json.dumps(
                            {
                                "vod_id": "2712128863",
                                "video_path": str(root / "missing.mp4"),
                                "scoped_window_id": "missing",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            vod_plan = [
                {"vod_id": "2711274372", "vod_url": "https://www.twitch.tv/videos/2711274372", "input_mode": "vod_url"},
                {"vod_id": "2712128863", "vod_url": "https://www.twitch.tv/videos/2712128863", "input_mode": "vod_url"},
            ]

            expanded = backfill._expand_cached_windows(
                vod_plan,
                cached_manifest=manifest,
                use_cached_windows=True,
                max_windows_per_vod=0,
            )

        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0]["vod_id"], "2711274372")
        self.assertEqual(expanded[0]["input_mode"], "cached_window")
        self.assertEqual(expanded[0]["video_path"], str(media))
        self.assertEqual(expanded[1]["vod_id"], "2712128863")
        self.assertEqual(expanded[1]["input_mode"], "vod_url")

    def test_run_one_vod_passes_cached_video_path_to_pipeline(self) -> None:
        calls: list[dict] = []

        async def fake_run_pipeline(**kwargs):
            calls.append(kwargs)

        job = types.SimpleNamespace(status="done", error="")
        fake_job_manager = types.SimpleNamespace(
            create_job=lambda *_args, **_kwargs: None,
            get_job=lambda *_args, **_kwargs: job,
        )
        fake_pipeline = types.SimpleNamespace(run_pipeline=fake_run_pipeline)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules,
            {
                "job_manager": fake_job_manager,
                "pipeline": fake_pipeline,
            },
        ):
            media = Path(tmp) / "source.mp4"
            media.write_bytes(b"fake mp4")
            result = asyncio.run(
                backfill._run_one_vod(
                    {
                        "vod_id": "2711274372",
                        "vod_url": "https://www.twitch.tv/videos/2711274372",
                        "input_mode": "cached_window",
                        "video_path": str(media),
                        "scoped_window_id": "window_a",
                    },
                    profile="jynxzi",
                    index=1,
                    total=1,
                )
            )

        self.assertEqual(result["status"], "done")
        self.assertEqual(calls[0]["url"], "https://www.twitch.tv/videos/2711274372")
        self.assertEqual(calls[0]["video_path"], media)
        self.assertEqual(calls[0]["profile"], "jynxzi")


if __name__ == "__main__":
    unittest.main()
