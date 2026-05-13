import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml import prepare_cross_platform_fingerprint_rows as prepare


class PrepareCrossPlatformFingerprintRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._sleep_patch = patch.object(prepare.time, "sleep")
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)

    def test_build_vod_manifest_from_scoped_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scoped = root / "123" / "scoped_windows" / "abc"
            scoped.mkdir(parents=True)
            (scoped / "source.mp4").write_bytes(b"video")
            (scoped / "source.metadata.json").write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "expanded_start": 42.5,
                        "expanded_end": 90.0,
                        "source": {
                            "vod_url": "https://www.twitch.tv/videos/123",
                            "window_start": 50.0,
                            "window_end": 60.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = prepare.build_vod_manifest_from_cache(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vod_id"], "123")
        self.assertEqual(rows[0]["source_start_offset"], 42.5)
        self.assertEqual(rows[0]["manifest_source"], "vod_feature_cache_scoped_window")

    def test_build_vod_manifest_can_exclude_scoped_windows_for_audio_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scoped = root / "123" / "scoped_windows" / "abc"
            scoped.mkdir(parents=True)
            (scoped / "source.mp4").write_bytes(b"video")
            (scoped / "source.metadata.json").write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "expanded_start": 42.5,
                        "expanded_end": 90.0,
                        "source": {"vod_url": "https://www.twitch.tv/videos/123"},
                    }
                ),
                encoding="utf-8",
            )

            rows = prepare.build_vod_manifest_from_cache(root, include_scoped_windows=False)

        self.assertEqual(rows, [])

    def test_build_vod_manifest_from_full_vod_audio_fingerprint_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "123" / "full_vod_audio"
            audio.mkdir(parents=True)
            (audio / "source.fpcalc.json").write_text(
                json.dumps({"duration": 120.0, "fingerprint": [1, 2, 3]}),
                encoding="utf-8",
            )
            (audio / "source.metadata.json").write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "expanded_start": 0.0,
                        "expanded_end": 120.0,
                        "source_kind": "full_vod_audio_fingerprint",
                        "source": {"vod_url": "https://www.twitch.tv/videos/123"},
                    }
                ),
                encoding="utf-8",
            )

            rows = prepare.build_vod_manifest_from_cache(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vod_id"], "123")
        self.assertEqual(rows[0]["fingerprint_path"], str(audio / "source.fpcalc.json"))
        self.assertEqual(rows[0]["source_start_offset"], 0.0)
        self.assertEqual(rows[0]["source_end_offset"], 120.0)
        self.assertEqual(rows[0]["manifest_source"], "vod_feature_cache_full_vod_audio_fingerprint")

    def test_build_full_vod_transcript_caches_downloads_and_caches_missing_vod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vod_rows = [{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}]
            transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]

            with patch.object(
                prepare.build_twitch_training_dataset,
                "_load_compatible_transcript_cache",
                side_effect=[None, None],
            ), patch.object(
                prepare,
                "_download_full_vod_audio",
            ) as download_mock, patch(
                "services.transcriber.transcribe",
                return_value=transcript,
            ) as transcribe_mock, patch.object(
                prepare.build_twitch_training_dataset,
                "_write_transcript_cache",
            ) as write_mock, patch.object(
                prepare.build_twitch_training_dataset,
                "_prune_compatible_transcript_caches",
            ) as prune_mock:
                payload = prepare.build_full_vod_transcript_caches(
                    vod_rows=vod_rows,
                    vod_cache_root=root,
                    keep_audio=False,
                )

            expected_audio = root / "123" / "full_vod_audio" / "source.m4a"
            download_mock.assert_called_once_with("https://www.twitch.tv/videos/123", expected_audio)
            transcribe_mock.assert_called_once_with(expected_audio, enable_diarization=False)
            self.assertEqual(write_mock.call_count, 1)
            self.assertEqual(prune_mock.call_count, 1)
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["processed_vods"], 1)
            self.assertEqual(payload["built_transcripts"], 1)
            self.assertEqual(payload["cached_transcripts"], 0)
            self.assertEqual(payload["pipeline_mode"], "download_queue_ahead")
            self.assertEqual(payload["download_workers"], 1)
            self.assertEqual(payload["transcribe_workers"], 1)

    def test_build_full_vod_transcript_caches_queues_all_audio_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vod_rows = [
                {"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"},
                {"vod_id": "456", "vod_url": "https://www.twitch.tv/videos/456"},
                {"vod_id": "789", "vod_url": "https://www.twitch.tv/videos/789"},
            ]
            transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]
            events: list[tuple[str, str]] = []

            def _fake_download(vod_url: str, _audio_path: Path) -> None:
                events.append(("download", vod_url.rsplit("/", 1)[-1]))

            def _fake_transcribe(_audio_path: Path, **_kwargs):
                if _audio_path.parents[1].name == "123":
                    deadline = time.monotonic() + 1.0
                    while ("download", "789") not in events and time.monotonic() < deadline:
                        time.sleep(0.01)
                events.append(("transcribe", _audio_path.parents[1].name))
                return transcript

            with patch.object(
                prepare.build_twitch_training_dataset,
                "_load_compatible_transcript_cache",
                return_value=None,
            ), patch.object(
                prepare,
                "_download_full_vod_audio",
                side_effect=_fake_download,
            ) as download_mock, patch(
                "services.transcriber.transcribe",
                side_effect=_fake_transcribe,
            ) as transcribe_mock, patch.object(
                prepare.build_twitch_training_dataset,
                "_write_transcript_cache",
            ), patch.object(
                prepare.build_twitch_training_dataset,
                "_prune_compatible_transcript_caches",
            ):
                payload = prepare.build_full_vod_transcript_caches(
                    vod_rows=vod_rows,
                    vod_cache_root=root,
                    keep_audio=False,
                )

            self.assertEqual(download_mock.call_count, 3)
            self.assertEqual(transcribe_mock.call_count, 3)
            self.assertEqual(payload["built_transcripts"], 3)
            self.assertLess(events.index(("download", "789")), events.index(("transcribe", "123")))
            self.assertEqual(payload["pipeline_mode"], "download_queue_ahead")
            self.assertEqual(payload["download_workers"], 1)
            self.assertEqual(payload["transcribe_workers"], 1)

    def test_build_full_vod_transcript_caches_reuses_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_path = root / "123" / "transcript_existing.json"
            transcript = [{"start": 0.0, "end": 1.0, "text": "cached"}]

            with patch.object(
                prepare.build_twitch_training_dataset,
                "_load_compatible_transcript_cache",
                return_value=(transcript_path, transcript),
            ), patch.object(
                prepare,
                "_download_full_vod_audio",
                side_effect=AssertionError("cached transcript should not download audio"),
            ), patch(
                "services.transcriber.transcribe",
                side_effect=AssertionError("cached transcript should not transcribe audio"),
            ):
                payload = prepare.build_full_vod_transcript_caches(
                    vod_rows=[{"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123"}],
                    vod_cache_root=root,
                    keep_audio=False,
                )

            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["processed_vods"], 1)
            self.assertEqual(payload["built_transcripts"], 0)
            self.assertEqual(payload["cached_transcripts"], 1)

    def test_prepare_rebuilds_stale_vod_manifest_after_full_audio_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            for vod_id in ("full1", "full2"):
                audio = cache_root / vod_id / "full_vod_audio"
                audio.mkdir(parents=True)
                (audio / "source.fpcalc.json").write_text(
                    json.dumps({"duration": 120.0, "fingerprint": [1, 2, 3]}),
                    encoding="utf-8",
                )
                (audio / "source.metadata.json").write_text(
                    json.dumps(
                        {
                            "vod_id": vod_id,
                            "expanded_start": 0.0,
                            "expanded_end": 120.0,
                            "source_kind": "full_vod_audio_fingerprint",
                            "source": {"vod_url": f"https://www.twitch.tv/videos/{vod_id}"},
                        }
                    ),
                    encoding="utf-8",
                )
            stale_media = root / "stale.mp4"
            stale_media.write_bytes(b"video")
            repost_media = root / "repost.mp4"
            repost_media.write_bytes(b"video")
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            vods.write_text(json.dumps({"vod_id": "stale", "video_path": str(stale_media)}) + "\n", encoding="utf-8")
            reposts.write_text(json.dumps({"video_path": str(repost_media), "platform": "youtube"}) + "\n", encoding="utf-8")
            passed_audio_summary = {
                "status": "passed",
                "selected_vods": 2,
                "processed_vods": 2,
                "built_fingerprints": 2,
                "cached_fingerprints": 0,
            }
            captured_vod_rows: list[dict] = []

            def _fake_match(vod_rows, *_args, **_kwargs):
                captured_vod_rows.extend(vod_rows)
                return [], {"best_match_rows": [], "top_best_matches": [], "confidence_distribution": {"count": 0}}

            with patch.object(prepare, "build_full_vod_audio_fingerprints", return_value=passed_audio_summary), \
                 patch.object(prepare.matcher, "match_reposts_to_vods_with_diagnostics", side_effect=_fake_match):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=cache_root,
                    repost_source_manifest=root / "missing_sources.csv",
                    repost_url_manifest=root / "missing_urls.txt",
                    full_vod_audio_fingerprints=True,
                    refresh_vods=False,
                    reuse_existing_output=False,
                    discover_reposts=False,
                    download_reposts=False,
                )

            manifest_rows = [json.loads(line) for line in vods.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(payload["auto_manifest_summary"]["vod_manifest_built"])
            self.assertEqual(payload["auto_manifest_summary"]["vod_manifest_build_reason"], "full_vod_audio_fingerprints")
            self.assertEqual(payload["auto_manifest_summary"]["vod_manifest_rows"], 2)
            self.assertEqual({row["vod_id"] for row in manifest_rows}, {"full1", "full2"})
            self.assertEqual({row["vod_id"] for row in captured_vod_rows}, {"full1", "full2"})
            self.assertTrue(all("fingerprint_path" in row for row in captured_vod_rows))

    def test_build_vod_manifest_prefers_full_audio_fingerprint_over_scoped_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "123" / "full_vod_audio"
            scoped = root / "123" / "scoped_windows" / "abc"
            audio.mkdir(parents=True)
            scoped.mkdir(parents=True)
            (audio / "source.fpcalc.json").write_text(
                json.dumps({"duration": 120.0, "fingerprint": [1, 2, 3]}),
                encoding="utf-8",
            )
            (audio / "source.metadata.json").write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "expanded_start": 0.0,
                        "expanded_end": 120.0,
                        "source": {"vod_url": "https://www.twitch.tv/videos/123"},
                    }
                ),
                encoding="utf-8",
            )
            (scoped / "source.mp4").write_bytes(b"video")
            (scoped / "source.metadata.json").write_text(
                json.dumps(
                    {
                        "vod_id": "123",
                        "expanded_start": 10.0,
                        "expanded_end": 20.0,
                        "source": {"vod_url": "https://www.twitch.tv/videos/123"},
                    }
                ),
                encoding="utf-8",
            )

            rows = prepare.build_vod_manifest_from_cache(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["manifest_source"], "vod_feature_cache_full_vod_audio_fingerprint")
        self.assertIn("fingerprint_path", rows[0])
        self.assertNotIn("video_path", rows[0])

    def test_build_full_vod_audio_fingerprints_writes_fingerprint_and_drops_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            inventory = root / "vod_inventory.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            labels.write_text("", encoding="utf-8")
            inventory.write_text(
                json.dumps(
                    {
                        "broadcaster": "jynxzi",
                        "vod_id": "111",
                        "vod_url": "https://www.twitch.tv/videos/111",
                        "vod_duration_seconds": 1000,
                        "inventory_source": "twitch_vod_api",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selected.write_text(
                json.dumps({"vod_id": "111", "vod_url": "https://www.twitch.tv/videos/111"}) + "\n",
                encoding="utf-8",
            )

            def _fake_download(_url, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"audio")
                return output_path

            def _fake_fingerprint(media_path, output_path, _fpcalc):
                self.assertTrue(media_path.exists())
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps({"duration": 1000.0, "fingerprint": [1, 2, 3]}), encoding="utf-8")
                from ml.match_cross_platform_audio_fingerprints import Fingerprint
                return Fingerprint(duration=1000.0, values=[1, 2, 3])

            with patch.object(prepare.shutil, "which", return_value="/usr/bin/fpcalc"), \
                 patch.object(prepare, "_download_full_vod_audio", side_effect=_fake_download), \
                 patch.object(prepare.matcher, "fingerprint_media_to_file", side_effect=_fake_fingerprint):
                summary = prepare.build_full_vod_audio_fingerprints(
                    labels_path=labels,
                    vod_inventory_path=inventory,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    max_vods=0,
                    keep_audio=False,
                )

            audio_dir = cache_root / "111" / "full_vod_audio"
            self.assertEqual(summary["built_fingerprints"], 1)
            self.assertEqual(summary["input_label_rows"], 0)
            self.assertEqual(summary["input_vod_inventory_rows"], 1)
            self.assertEqual(summary["processed_vods"], 1)
            self.assertFalse((audio_dir / "source.m4a").exists())
            self.assertTrue((audio_dir / "source.fpcalc.json").exists())
            self.assertTrue((audio_dir / "source.metadata.json").exists())

    def test_full_vod_audio_filters_cache_only_vods_without_existing_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            inventory = root / "vod_inventory.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            labels.write_text("", encoding="utf-8")
            inventory.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {
                            "broadcaster": "jynxzi",
                            "vod_id": "api",
                            "vod_url": "https://www.twitch.tv/videos/api",
                            "vod_duration_seconds": 100,
                            "inventory_source": "twitch_vod_api",
                        },
                        {
                            "broadcaster": "jynxzi",
                            "vod_id": "cache",
                            "vod_url": "https://www.twitch.tv/videos/cache",
                            "vod_duration_seconds": 100,
                            "inventory_source": "vod_feature_cache",
                        },
                    )
                ),
                encoding="utf-8",
            )
            selected.write_text(
                "".join(
                    json.dumps({"vod_id": vod_id, "vod_url": f"https://www.twitch.tv/videos/{vod_id}"}) + "\n"
                    for vod_id in ("api", "cache")
                ),
                encoding="utf-8",
            )
            downloaded_urls: list[str] = []

            def _fake_download(url, output_path):
                downloaded_urls.append(url)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"audio")
                return output_path

            def _fake_fingerprint(media_path, output_path, _fpcalc):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps({"duration": 100.0, "fingerprint": [1]}), encoding="utf-8")
                from ml.match_cross_platform_audio_fingerprints import Fingerprint
                return Fingerprint(duration=100.0, values=[1])

            with patch.object(prepare.shutil, "which", return_value="/usr/bin/fpcalc"), \
                 patch.object(prepare, "_download_full_vod_audio", side_effect=_fake_download), \
                 patch.object(prepare.matcher, "fingerprint_media_to_file", side_effect=_fake_fingerprint):
                summary = prepare.build_full_vod_audio_fingerprints(
                    labels_path=labels,
                    vod_inventory_path=inventory,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    max_vods=0,
                    keep_audio=False,
                )

            self.assertEqual(summary["input_selected_vods"], 2)
            self.assertEqual(summary["selected_vods"], 1)
            self.assertEqual(summary["skipped_cache_only_without_full_audio_fingerprint"], 1)
            self.assertEqual(downloaded_urls, ["https://www.twitch.tv/videos/api"])

    def test_full_vod_audio_uses_parallel_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            inventory = root / "vod_inventory.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            labels.write_text("", encoding="utf-8")
            inventory.write_text(
                "".join(
                    json.dumps(
                        {
                            "broadcaster": "jynxzi",
                            "vod_id": vod_id,
                            "vod_url": f"https://www.twitch.tv/videos/{vod_id}",
                            "vod_duration_seconds": 100,
                            "inventory_source": "twitch_vod_api",
                        }
                    )
                    + "\n"
                    for vod_id in ("api1", "api2")
                ),
                encoding="utf-8",
            )
            selected.write_text(
                "".join(
                    json.dumps({"vod_id": vod_id, "vod_url": f"https://www.twitch.tv/videos/{vod_id}"}) + "\n"
                    for vod_id in ("api1", "api2")
                ),
                encoding="utf-8",
            )

            def _fake_download(_url, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"audio")
                return output_path

            def _fake_fingerprint(_media_path, output_path, _fpcalc):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps({"duration": 100.0, "fingerprint": [1]}), encoding="utf-8")
                from ml.match_cross_platform_audio_fingerprints import Fingerprint
                return Fingerprint(duration=100.0, values=[1])

            with patch.object(prepare.shutil, "which", return_value="/usr/bin/fpcalc"), \
                 patch.object(prepare, "_download_full_vod_audio", side_effect=_fake_download), \
                 patch.object(prepare.matcher, "fingerprint_media_to_file", side_effect=_fake_fingerprint):
                summary = prepare.build_full_vod_audio_fingerprints(
                    labels_path=labels,
                    vod_inventory_path=inventory,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    max_vods=0,
                    keep_audio=False,
                    workers=2,
                )

            self.assertEqual(summary["workers"], 2)
            self.assertEqual(summary["built_fingerprints"], 2)
            self.assertEqual(summary["processed_vods"], 2)

    def test_full_vod_audio_accepts_cache_only_vods_with_existing_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            inventory = root / "vod_inventory.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            audio_dir = cache_root / "cache" / "full_vod_audio"
            audio_dir.mkdir(parents=True)
            (audio_dir / "source.fpcalc.json").write_text(
                json.dumps({"duration": 100.0, "fingerprint": [1, 2, 3]}),
                encoding="utf-8",
            )
            (audio_dir / "source.metadata.json").write_text(
                json.dumps(
                    {
                        "vod_id": "cache",
                        "source_kind": "full_vod_audio_fingerprint",
                        "source": {
                            "kind": "full_vod_audio_fingerprint",
                            "vod_url": "https://www.twitch.tv/videos/cache",
                            "horizon_source": "vod_duration",
                        },
                    }
                ),
                encoding="utf-8",
            )
            labels.write_text("", encoding="utf-8")
            inventory.write_text(
                json.dumps(
                    {
                        "broadcaster": "jynxzi",
                        "vod_id": "cache",
                        "vod_url": "https://www.twitch.tv/videos/cache",
                        "vod_duration_seconds": 100,
                        "inventory_source": "vod_feature_cache",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selected.write_text(
                json.dumps({"vod_id": "cache", "vod_url": "https://www.twitch.tv/videos/cache"}) + "\n",
                encoding="utf-8",
            )

            with patch.object(prepare.shutil, "which", return_value="/usr/bin/fpcalc"), \
                 patch.object(prepare, "_download_full_vod_audio") as download_mock:
                summary = prepare.build_full_vod_audio_fingerprints(
                    labels_path=labels,
                    vod_inventory_path=inventory,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    max_vods=0,
                    keep_audio=False,
                )

            self.assertEqual(summary["selected_vods"], 1)
            self.assertEqual(summary["cached_fingerprints"], 1)
            self.assertEqual(summary["skipped_cache_only_without_full_audio_fingerprint"], 0)
            download_mock.assert_not_called()

    def test_full_vod_audio_download_error_logs_ffmpeg_stderr(self) -> None:
        exc = prepare.FullVodAudioDownloadError(
            "ffmpeg failed while extracting full VOD audio",
            returncode=1,
            stderr="HTTP error 403 Forbidden",
            stdout="",
            format_summary={"format_id": "audio", "protocol": "m3u8_native"},
        )

        fields = prepare._exception_log_fields(exc)

        self.assertEqual(fields["error_type"], "FullVodAudioDownloadError")
        self.assertEqual(fields["ffmpeg_returncode"], 1)
        self.assertIn("403 Forbidden", fields["ffmpeg_stderr"])
        self.assertEqual(fields["ffmpeg_format"]["protocol"], "m3u8_native")

    def test_full_vod_audio_temp_path_keeps_media_extension_last(self) -> None:
        path = Path("/tmp/cache/source.m4a")

        self.assertEqual(prepare._full_vod_audio_temp_path(path), Path("/tmp/cache/source.part.m4a"))

    def test_select_vod_audio_input_returns_format_summary(self) -> None:
        url, headers, summary = prepare._select_vod_audio_input(
            {
                "formats": [
                    {
                        "url": "https://example.invalid/video.m3u8",
                        "format_id": "video",
                        "acodec": "mp4a",
                        "vcodec": "avc1",
                        "height": 160,
                        "tbr": 200,
                    },
                    {
                        "url": "https://example.invalid/audio.m3u8",
                        "format_id": "audio",
                        "acodec": "mp4a",
                        "vcodec": "none",
                        "abr": 64,
                        "protocol": "m3u8_native",
                        "http_headers": {"User-Agent": "test-agent"},
                    },
                ],
            }
        )

        self.assertEqual(url, "https://example.invalid/audio.m3u8")
        self.assertEqual(headers["User-Agent"], "test-agent")
        self.assertEqual(summary["format_id"], "audio")
        self.assertEqual(summary["vcodec"], "none")
        self.assertEqual(summary["protocol"], "m3u8_native")

    def test_build_coarse_vod_tiles_uses_selected_vods_and_label_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            labels.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "broadcaster": "jynxzi",
                            "vod_id": "111",
                            "vod_url": "https://www.twitch.tv/videos/111",
                            "window_start": 100.0,
                            "window_end": 1250.0,
                        },
                        {
                            "broadcaster": "jynxzi",
                            "vod_id": "222",
                            "vod_url": "https://www.twitch.tv/videos/222",
                            "window_start": 10.0,
                            "window_end": 60.0,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            selected.write_text(
                json.dumps({"vod_id": "111", "vod_url": "https://www.twitch.tv/videos/111"}) + "\n",
                encoding="utf-8",
            )

            def _fake_download(_vod_id, _label, source_path, metadata_path, *, metadata_payload, **_kwargs):
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(b"video")
                metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
                return source_path

            with patch.object(
                prepare.build_twitch_training_dataset,
                "_load_or_download_window_scoped_vod_range",
                side_effect=_fake_download,
            ):
                summary = prepare.build_coarse_vod_tiles(
                    labels_path=labels,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    tile_sec=30,
                    stride_sec=600,
                    max_tiles_per_vod=3,
                    max_vods=0,
                    horizon_padding_sec=0,
                )

            manifest_rows = prepare.build_vod_manifest_from_cache(cache_root)

        self.assertEqual(summary["processed_vods"], 1)
        self.assertEqual(summary["built_tiles"], 3)
        self.assertEqual({row["vod_id"] for row in manifest_rows}, {"111"})
        self.assertEqual(sorted(row["source_start_offset"] for row in manifest_rows), [0.0, 600.0, 1200.0])
        self.assertTrue(all("coarse_" in row["video_path"] for row in manifest_rows))

    def test_build_coarse_vod_tiles_covers_full_vod_duration_without_tile_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            labels.write_text(
                json.dumps(
                    {
                        "broadcaster": "jynxzi",
                        "vod_id": "111",
                        "vod_url": "https://www.twitch.tv/videos/111",
                        "vod_duration_seconds": 1000,
                        "window_start": 100.0,
                        "window_end": 125.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selected.write_text(
                json.dumps({"vod_id": "111", "vod_url": "https://www.twitch.tv/videos/111"}) + "\n",
                encoding="utf-8",
            )

            def _fake_download(_vod_id, _label, source_path, metadata_path, *, metadata_payload, **_kwargs):
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(b"video")
                metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
                return source_path

            with patch.object(
                prepare.build_twitch_training_dataset,
                "_load_or_download_window_scoped_vod_range",
                side_effect=_fake_download,
            ):
                summary = prepare.build_coarse_vod_tiles(
                    labels_path=labels,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    tile_sec=360,
                    stride_sec=180,
                    max_tiles_per_vod=0,
                    max_vods=0,
                    horizon_padding_sec=900,
                )

            manifest_rows = prepare.build_vod_manifest_from_cache(cache_root)

        self.assertEqual(summary["coverage_mode"], "full_vod_overlap")
        self.assertEqual(summary["full_duration_vods"], 1)
        self.assertEqual(summary["label_horizon_vods"], 0)
        self.assertEqual(summary["total_tiles"], 6)
        self.assertEqual(summary["built_tiles"], 6)
        self.assertEqual(sorted(row["source_start_offset"] for row in manifest_rows), [0.0, 180.0, 360.0, 540.0, 720.0, 900.0])
        self.assertEqual(max(row["source_end_offset"] for row in manifest_rows), 1000.0)

    def test_build_coarse_vod_tiles_aborts_deleted_twitch_vod_after_first_failed_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            selected = root / "selected.jsonl"
            cache_root = root / "cache"
            labels.write_text(
                json.dumps(
                    {
                        "broadcaster": "jynxzi",
                        "vod_id": "111",
                        "vod_url": "https://www.twitch.tv/videos/111",
                        "vod_duration_seconds": 1000,
                        "window_start": 100.0,
                        "window_end": 125.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selected.write_text(
                json.dumps({"vod_id": "111", "vod_url": "https://www.twitch.tv/videos/111"}) + "\n",
                encoding="utf-8",
            )

            def _deleted_vod(*_args, **_kwargs):
                raise RuntimeError("ERROR: [twitch:vod] 111: Video 111 does not exist")

            with patch.object(
                prepare.build_twitch_training_dataset,
                "_load_or_download_window_scoped_vod_range",
                side_effect=_deleted_vod,
            ) as download_mock:
                summary = prepare.build_coarse_vod_tiles(
                    labels_path=labels,
                    selected_vods_path=selected,
                    vod_cache_root=cache_root,
                    profile="jynxzi",
                    tile_sec=360,
                    stride_sec=180,
                    max_tiles_per_vod=0,
                    max_vods=0,
                    horizon_padding_sec=900,
                )

        self.assertEqual(download_mock.call_count, 1)
        self.assertEqual(summary["total_tiles"], 6)
        self.assertEqual(summary["failed_tiles"], 1)
        self.assertEqual(summary["unavailable_vods"], 1)
        self.assertEqual(summary["built_tiles"], 0)

    def test_prepare_rebuilds_stale_vod_manifest_with_missing_media_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            scoped = cache_root / "123" / "scoped_windows" / "fresh"
            scoped.mkdir(parents=True)
            (scoped / "source.mp4").write_bytes(b"video")
            (scoped / "source.metadata.json").write_text(
                json.dumps({"vod_id": "123", "expanded_start": 5.0, "expanded_end": 30.0}),
                encoding="utf-8",
            )
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/missing/host/source.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=([], {"best_match_rows": [], "top_best_matches": [], "confidence_distribution": {"count": 0}}),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=root / "rows.jsonl",
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=cache_root,
                    repost_source_manifest=root / "missing_sources.csv",
                    repost_url_manifest=root / "missing_urls.txt",
                    full_vod_audio_fingerprints=False,
                    discover_reposts=False,
                    download_reposts=False,
                )

            rebuilt_rows = [json.loads(line) for line in vods.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(payload["auto_manifest_summary"]["vod_manifest_build_reason"], "missing_or_empty_or_unusable")
            self.assertEqual(len(rebuilt_rows), 1)
            self.assertEqual(rebuilt_rows[0]["video_path"], str(scoped / "source.mp4"))

    def test_prepare_repairs_host_backend_paths_when_container_path_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "tmp" / "vod_feature_cache" / "123" / "scoped_windows" / "abc" / "source.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"video")
            stale = f"/home/antonio/ai_clipping/backend/{media.relative_to(root).as_posix()}"
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            vods.write_text(json.dumps({"vod_id": "123", "video_path": stale}) + "\n", encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")

            with patch.object(prepare.matcher, "BACKEND_ROOT", root), \
                 patch.object(
                     prepare.matcher,
                     "match_reposts_to_vods_with_diagnostics",
                     return_value=([], {"best_match_rows": [], "top_best_matches": [], "confidence_distribution": {"count": 0}}),
                 ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=root / "rows.jsonl",
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=root / "tmp" / "vod_feature_cache",
                    repost_source_manifest=root / "missing_sources.csv",
                    repost_url_manifest=root / "missing_urls.txt",
                    discover_reposts=False,
                    download_reposts=False,
                    full_vod_audio_fingerprints=False,
                )

            repaired = [json.loads(line) for line in vods.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertFalse(payload["auto_manifest_summary"]["vod_manifest_built"])
            self.assertEqual(repaired[0]["video_path"], str(media))
            self.assertEqual(repaired[0]["original_video_path"], stale)

    def test_build_repost_manifest_discovers_and_downloads_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            media = root / "short.mp4"
            media.write_bytes(b"video")
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,https://www.youtube.com/@Jynxzi/shorts,en,jynxzi,2\n",
                encoding="utf-8",
            )
            discovered = [{
                "url": "https://www.youtube.com/shorts/abc",
                "streamer": "jynxzi",
                "platform": "youtube",
                "title": "Jynxzi wins a ranked clutch",
                "uploader": "Jynxzi Clips",
                "discovery_source_url": "ytsearch500:jynxzi shorts",
                "view_count": "12345",
            }]

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", return_value=media):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_url"], "https://www.youtube.com/shorts/abc")
        self.assertEqual(rows[0]["video_path"], str(media))
        self.assertEqual(rows[0]["views"], 12345.0)
        self.assertEqual(rows[0]["title"], "Jynxzi wins a ranked clutch")
        self.assertEqual(rows[0]["uploader"], "Jynxzi Clips")
        self.assertEqual(rows[0]["profile_filter_version"], prepare.REPOST_PROFILE_FILTER_VERSION)
        self.assertEqual(summary["downloaded_or_cached_rows"], 1)
        self.assertEqual(summary["skipped_non_profile_reposts"], 0)

    def test_build_repost_manifest_downloads_required_shorts_serially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,https://www.youtube.com/@Clips.Delivery/shorts,en,"
                "curated_repost_source;clips_delivery;youtube_shorts,3\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": f"https://www.youtube.com/shorts/{index}",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": f"curated clip {index}",
                    "uploader": "Clips Delivery",
                    "discovery_source_url": "https://www.youtube.com/@Clips.Delivery/shorts",
                    "tags": "curated_repost_source;clips_delivery;youtube_shorts",
                    "upload_date": "20260503",
                    "view_count": str(100 - index),
                }
                for index in range(3)
            ]
            download_order: list[str] = []
            sleeps: list[float] = []

            def fake_download(url: str) -> Path:
                download_order.append(url)
                path = root / f"{url.rsplit('/', 1)[-1]}.mp4"
                path.write_bytes(b"video")
                return path

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", side_effect=fake_download), \
                 patch.object(prepare.settings, "yt_dlp_short_form_min_sleep_interval_sec", 5.0), \
                 patch.object(prepare.settings, "yt_dlp_short_form_max_sleep_interval_sec", 10.0), \
                 patch.object(prepare.random, "uniform", return_value=7.5), \
                 patch.object(prepare.time, "sleep", side_effect=lambda sec: sleeps.append(sec)):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                    repost_date_window=(
                        prepare._parse_datetime_value("2026-05-01T00:00:00+00:00"),
                        prepare._parse_datetime_value("2026-05-31T00:00:00+00:00"),
                    ),
                )

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            download_order,
            [
                "https://www.youtube.com/shorts/0",
                "https://www.youtube.com/shorts/1",
                "https://www.youtube.com/shorts/2",
            ],
        )
        self.assertEqual(summary["repost_download_mode"], "serial")
        self.assertEqual(summary["repost_download_submitted_rows"], 3)
        self.assertEqual(summary["download_failures"], 0)
        self.assertEqual(sleeps, [7.5, 7.5, 7.5])
        self.assertEqual(summary["repost_download_request_cooldowns"], 3)
        self.assertEqual(summary["repost_download_success_cooldowns"], 3)

    def test_build_repost_manifest_sleeps_and_continues_after_download_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            urls = [
                "https://www.youtube.com/shorts/Zsa94sijQQ8",
                "https://www.youtube.com/shorts/next",
            ]
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,https://www.youtube.com/@Clips.Delivery/shorts,en,"
                "curated_repost_source;clips_delivery;youtube_shorts,2\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": url,
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": f"curated clip {index}",
                    "uploader": "Clips Delivery",
                    "discovery_source_url": "https://www.youtube.com/@Clips.Delivery/shorts",
                    "tags": "curated_repost_source;clips_delivery;youtube_shorts",
                    "upload_date": "20260503",
                    "view_count": str(100 - index),
                }
                for index, url in enumerate(urls)
            ]
            download_calls: list[str] = []
            sleeps: list[float] = []

            def fake_download(download_url: str) -> Path:
                download_calls.append(download_url)
                if len(download_calls) == 1:
                    raise RuntimeError(
                        "[youtube] Zsa94sijQQ8: This content isn't available, try again later. "
                        "The current session has been rate-limited by YouTube for up to an hour."
                    )
                media = root / f"{download_url.rsplit('/', 1)[-1]}.mp4"
                media.write_bytes(b"video")
                return media

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", side_effect=fake_download), \
                 patch.object(prepare.settings, "yt_dlp_short_form_min_sleep_interval_sec", 5.0), \
                 patch.object(prepare.settings, "yt_dlp_short_form_max_sleep_interval_sec", 10.0), \
                 patch.object(prepare.random, "uniform", return_value=7.5), \
                 patch.object(prepare.time, "sleep", side_effect=lambda sec: sleeps.append(sec)):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                    repost_date_window=(
                        prepare._parse_datetime_value("2026-05-01T00:00:00+00:00"),
                        prepare._parse_datetime_value("2026-05-31T00:00:00+00:00"),
                    ),
                )

        self.assertEqual([row["source_url"] for row in rows], [urls[1]])
        self.assertEqual(download_calls, urls)
        self.assertEqual(sleeps, [7.5, 7.5])
        self.assertEqual(summary["download_failures"], 1)
        self.assertEqual(summary["repost_download_error_cooldowns"], 1)
        self.assertEqual(summary["repost_download_request_cooldowns"], 2)
        self.assertEqual(summary["repost_download_rate_limit_cooldowns"], 1)
        self.assertEqual(summary["repost_download_success_cooldowns"], 1)

    def test_build_repost_manifest_filters_discovered_non_profile_search_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            media = root / "short.mp4"
            media.write_bytes(b"video")
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,ytsearch500:jynxzi shorts,en,jynxzi,2\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": "https://www.youtube.com/shorts/good",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi funny R6 clip",
                    "uploader": "Siege Clips",
                    "discovery_source_url": "ytsearch500:jynxzi shorts",
                    "view_count": "1000",
                },
                {
                    "url": "https://www.youtube.com/shorts/bad",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "unrelated streamer funny moment",
                    "uploader": "Random Gaming",
                    "discovery_source_url": "ytsearch500:jynxzi shorts",
                    "tags": "jynxzi",
                    "view_count": "999999",
                },
                {
                    "url": "https://www.tiktok.com/@random/video/123",
                    "streamer": "jynxzi",
                    "platform": "tiktok",
                    "title": "unrelated tiktok clip",
                    "uploader": "Random TikTok",
                    "discovery_source_url": "tiktoksearch500:jynxzi clips",
                    "tags": "jynxzi",
                    "view_count": "888888",
                },
            ]

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", return_value=media):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                )

        self.assertEqual([row["source_url"] for row in rows], ["https://www.youtube.com/shorts/good"])
        self.assertEqual(summary["skipped_non_profile_reposts"], 2)
        self.assertEqual(summary["downloaded_or_cached_rows"], 1)

    def test_build_repost_manifest_trusts_curated_channel_but_keeps_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            media = root / "short.mp4"
            media.write_bytes(b"video")
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,https://www.youtube.com/@Clips.Delivery/shorts,en,"
                "curated_repost_source;clips_delivery;youtube_shorts,3600\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": "https://www.youtube.com/shorts/inside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "ranked clip from curated channel",
                    "uploader": "Clips Delivery",
                    "discovery_source_url": "https://www.youtube.com/@Clips.Delivery/shorts",
                    "tags": "curated_repost_source;clips_delivery;youtube_shorts",
                    "upload_date": "20260503",
                    "view_count": "1000",
                },
                {
                    "url": "https://www.youtube.com/shorts/outside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "older curated channel clip",
                    "uploader": "Clips Delivery",
                    "discovery_source_url": "https://www.youtube.com/@Clips.Delivery/shorts",
                    "tags": "curated_repost_source;clips_delivery;youtube_shorts",
                    "upload_date": "20240101",
                    "view_count": "900",
                },
            ]

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", return_value=media):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                    repost_date_window=(
                        prepare._parse_datetime_value("2026-05-01T00:00:00+00:00"),
                        prepare._parse_datetime_value("2026-05-31T00:00:00+00:00"),
                    ),
                )

        self.assertEqual([row["source_url"] for row in rows], ["https://www.youtube.com/shorts/inside"])
        self.assertEqual(rows[0]["uploader"], "Clips Delivery")
        self.assertEqual(summary["skipped_non_profile_reposts"], 0)
        self.assertEqual(summary["skipped_outside_vod_date_window"], 1)
        self.assertEqual(summary["downloaded_or_cached_rows"], 1)

    def test_build_repost_manifest_reuses_existing_youtube_media_by_video_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            media = root / "short.mp4"
            media.write_bytes(b"video")
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,https://www.youtube.com/@Iboclipss/shorts,en,"
                "curated_repost_source;iboclipss;youtube_shorts,3600\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": "https://www.youtube.com/watch?v=same-id",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "fresh manifest row",
                    "uploader": "iboclips",
                    "discovery_source_url": "https://www.youtube.com/@Iboclipss/shorts",
                    "tags": "curated_repost_source;iboclipss;youtube_shorts",
                    "upload_date": "20260503",
                    "view_count": "1000",
                },
            ]
            existing_rows = [
                {
                    "source_url": "https://www.youtube.com/shorts/same-id",
                    "platform": "youtube",
                    "video_path": str(media),
                }
            ]

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(
                     prepare.collect_viral_clips,
                     "_ensure_short_form_media_cached",
                     side_effect=AssertionError("existing media should be reused"),
                 ):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                    existing_repost_rows=existing_rows,
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_path"], str(media))
        self.assertEqual(summary["repost_existing_media_reused_rows"], 1)
        self.assertEqual(summary["repost_download_submitted_rows"], 0)

    def test_build_repost_manifest_filters_to_vod_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            media = root / "short.mp4"
            media.write_bytes(b"video")
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,ytsearch500:jynxzi,en,jynxzi,4\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": "https://www.youtube.com/shorts/inside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi inside date",
                    "uploader": "Jynxzi Clips",
                    "discovery_source_url": "ytsearch500:jynxzi",
                    "upload_date": "20260503",
                    "view_count": "1000",
                },
                {
                    "url": "https://www.youtube.com/shorts/outside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi old date",
                    "uploader": "Jynxzi Clips",
                    "discovery_source_url": "ytsearch500:jynxzi",
                    "upload_date": "20240101",
                    "view_count": "900",
                },
                {
                    "url": "https://www.youtube.com/shorts/missing",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi missing date",
                    "uploader": "Jynxzi Clips",
                    "discovery_source_url": "ytsearch500:jynxzi",
                    "view_count": "800",
                },
            ]

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", return_value=media), \
                 patch.object(prepare.auto_train_short_form, "_extract_single_entry_metadata") as metadata_mock:
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=True,
                    repost_date_window=(
                        prepare._parse_datetime_value("2026-05-01T00:00:00+00:00"),
                        prepare._parse_datetime_value("2026-05-31T00:00:00+00:00"),
                    ),
                )
                metadata_mock.assert_not_called()

        self.assertEqual([row["source_url"] for row in rows], ["https://www.youtube.com/shorts/inside"])
        self.assertEqual(summary["skipped_outside_vod_date_window"], 1)
        self.assertEqual(summary["skipped_missing_publish_date"], 1)
        self.assertEqual(summary["downloaded_or_cached_rows"], 1)

    def test_build_repost_manifest_picks_most_viewed_inside_vod_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = root / "sources.csv"
            url_manifest = root / "urls.txt"
            media = root / "short.mp4"
            media.write_bytes(b"video")
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,ytsearchdate500:jynxzi,en,jynxzi,4\n",
                encoding="utf-8",
            )
            discovered = [
                {
                    "url": "https://www.youtube.com/shorts/low-inside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi low view current clip",
                    "uploader": "Jynxzi Clips",
                    "discovery_source_url": "ytsearchdate500:jynxzi",
                    "upload_date": "20260503",
                    "view_count": "1000",
                    "viral_score": "2.0",
                },
                {
                    "url": "https://www.youtube.com/shorts/high-outside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi old high view clip",
                    "uploader": "Jynxzi Clips",
                    "discovery_source_url": "ytsearchdate500:jynxzi",
                    "upload_date": "20240101",
                    "view_count": "999999",
                    "viral_score": "99.0",
                },
                {
                    "url": "https://www.youtube.com/shorts/high-inside",
                    "streamer": "jynxzi",
                    "platform": "youtube",
                    "title": "Jynxzi high view current clip",
                    "uploader": "Jynxzi Clips",
                    "discovery_source_url": "ytsearchdate500:jynxzi",
                    "upload_date": "20260504",
                    "view_count": "50000",
                    "viral_score": "8.0",
                },
            ]

            with patch.object(prepare.auto_train_short_form, "_discover_seeds", return_value=discovered), \
                 patch.object(prepare.collect_viral_clips, "_ensure_short_form_media_cached", return_value=media):
                rows, summary = prepare.build_repost_manifest_from_sources(
                    source_manifest=source_manifest,
                    url_manifest=url_manifest,
                    profile="jynxzi",
                    max_reposts=1,
                    discover_reposts=True,
                    download_reposts=True,
                    repost_date_window=(
                        prepare._parse_datetime_value("2026-05-01T00:00:00+00:00"),
                        prepare._parse_datetime_value("2026-05-31T00:00:00+00:00"),
                    ),
                )

        self.assertEqual([row["source_url"] for row in rows], ["https://www.youtube.com/shorts/high-inside"])
        self.assertEqual(summary["candidate_entries"], 3)
        self.assertEqual(summary["eligible_profile_date_rows"], 2)
        self.assertEqual(summary["selected_repost_candidates"], 1)
        self.assertEqual(summary["skipped_outside_vod_date_window"], 1)

    def test_vod_repost_date_window_uses_only_usable_vod_rows(self) -> None:
        window = prepare._vod_repost_date_window(
            vod_rows=[{"vod_id": "kept"}],
            vod_inventory_rows=[
                {"vod_id": "kept", "created_at": "2026-05-05T12:00:00Z"},
                {"vod_id": "ignored", "created_at": "2025-01-01T00:00:00Z"},
            ],
            lookback_days=1,
            lookahead_days=30,
        )

        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window[0].isoformat(), "2026-05-04T12:00:00+00:00")
        self.assertEqual(window[1].isoformat(), "2026-06-04T12:00:00+00:00")

    def test_missing_manifests_preserve_existing_output_and_write_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            output.write_text('{"dataset_source":"cross_platform_audio_fingerprint","label":1}\n', encoding="utf-8")

            payload = prepare.prepare_cross_platform_fingerprint_rows(
                vod_manifest=root / "missing_vods.jsonl",
                repost_manifest=root / "missing_reposts.jsonl",
                output_path=output,
                summary_path=summary,
                auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
            )

            self.assertEqual(payload["status"], "missing_inputs")
            self.assertEqual(payload["output_rows"], 1)
            self.assertTrue(payload["preserved_existing_output"])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            written_summary = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(written_summary["missing_inputs"], ["vod_manifest", "repost_manifest"])

    def test_empty_existing_output_is_created_when_inputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "rows.jsonl"

            payload = prepare.prepare_cross_platform_fingerprint_rows(
                vod_manifest=root / "missing_vods.jsonl",
                repost_manifest=root / "missing_reposts.jsonl",
                output_path=output,
                summary_path=root / "summary.json",
                auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
            )

            self.assertEqual(payload["status"], "missing_inputs")
            self.assertTrue(output.exists())
            self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_empty_input_refresh_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            existing = {"dataset_source": "cross_platform_audio_fingerprint", "label": 1}
            vods.write_text("", encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            output.write_text(json.dumps(existing) + "\n", encoding="utf-8")

            payload = prepare.prepare_cross_platform_fingerprint_rows(
                vod_manifest=vods,
                repost_manifest=reposts,
                output_path=output,
                summary_path=root / "summary.json",
                auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                reuse_existing_output=False,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(payload["status"], "empty_inputs")
            self.assertEqual(payload["output_rows"], 1)
            self.assertTrue(payload["preserved_existing_output"])
            self.assertEqual(rows, [existing])

    def test_failed_full_vod_audio_refresh_preserves_existing_manifest_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            existing_vod_media = root / "old_vod.mp4"
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            existing_row = {
                "vod_id": "old-vod",
                "dataset_source": "cross_platform_audio_fingerprint",
                "fingerprint_confidence": 0.9,
                "label": 1,
            }
            existing_vod_media.write_bytes(b"video")
            vods.write_text(json.dumps({"vod_id": "old-vod", "video_path": str(existing_vod_media)}) + "\n", encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            output.write_text(json.dumps(existing_row) + "\n", encoding="utf-8")
            failed_audio_summary = {
                "status": "failed_no_full_vod_audio_fingerprints",
                "selected_vods": 1,
                "processed_vods": 0,
                "built_fingerprints": 0,
                "cached_fingerprints": 0,
                "failed_vods": 1,
            }

            with patch.object(prepare, "build_full_vod_audio_fingerprints", return_value=failed_audio_summary), \
                 patch.object(
                     prepare.matcher,
                     "match_reposts_to_vods_with_diagnostics",
                     side_effect=AssertionError("matching must be blocked when full-audio refresh fails"),
                 ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=cache_root,
                    repost_source_manifest=root / "missing_sources.csv",
                    repost_url_manifest=root / "missing_urls.txt",
                    full_vod_audio_fingerprints=True,
                    refresh_vods=True,
                    reuse_existing_output=False,
                    discover_reposts=False,
                    download_reposts=False,
                )

            manifest_rows = [json.loads(line) for line in vods.read_text(encoding="utf-8").splitlines() if line.strip()]
            output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(payload["status"], "blocked_full_vod_audio_fingerprints")
            self.assertEqual(payload["blocked_reason"], "failed_no_full_vod_audio_fingerprints")
            self.assertEqual(payload["output_rows"], 1)
            self.assertTrue(payload["preserved_existing_output"])
            self.assertEqual(manifest_rows[0]["video_path"], str(existing_vod_media))
            self.assertEqual(output_rows, [existing_row])

    def test_insufficient_repost_rows_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            existing = {
                "vod_id": "old-vod",
                "dataset_source": "cross_platform_audio_fingerprint",
                "fingerprint_confidence": 0.91,
                "label": 1,
            }
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            output.write_text(json.dumps(existing) + "\n", encoding="utf-8")

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                side_effect=AssertionError("matching should not run before repost minimum is met"),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=summary,
                    min_confidence=0.5,
                    min_reposts=2,
                    min_matched_rows=2,
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            written_summary = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "insufficient_repost_rows")
            self.assertEqual(payload["output_rows"], 1)
            self.assertEqual(payload["repost_rows_needed"], 1)
            self.assertTrue(payload["preserved_existing_output"])
            self.assertEqual(rows, [existing])
            self.assertEqual(written_summary["output_rows"], 1)

    def test_repost_manifest_below_max_does_not_force_rebuild_when_minimum_is_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vod_media = root / "vod.mp4"
            repost_media = root / "repost.mp4"
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            vod_media.write_bytes(b"vod")
            repost_media.write_bytes(b"repost")
            vods.write_text(json.dumps({"vod_id": "123", "video_path": str(vod_media)}) + "\n", encoding="utf-8")
            reposts.write_text(json.dumps({"video_path": str(repost_media), "platform": "youtube"}) + "\n", encoding="utf-8")

            with patch.object(
                prepare,
                "build_repost_manifest_from_sources",
                side_effect=AssertionError("max_reposts is a cap and must not force rebuild"),
            ), patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=([], {"best_match_rows": [], "confidence_distribution": {"count": 0}}),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=root / "rows.jsonl",
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=root / "cache",
                    repost_source_manifest=root / "missing_sources.csv",
                    repost_url_manifest=root / "missing_urls.txt",
                    full_vod_audio_fingerprints=False,
                    min_reposts=1,
                    max_reposts=10,
                    discover_reposts=False,
                    download_reposts=False,
                )

            auto_summary = payload["auto_manifest_summary"]
            self.assertFalse(auto_summary["repost_manifest_built"])
            self.assertTrue(auto_summary["repost_manifest_below_max_reposts"])
            self.assertEqual(auto_summary["repost_manifest_rows"], 1)

    def test_stale_repost_manifest_rebuilds_for_profile_filter_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vod_media = root / "vod.mp4"
            repost_media = root / "repost.mp4"
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            source_manifest = root / "sources.csv"
            vod_media.write_bytes(b"vod")
            repost_media.write_bytes(b"repost")
            vods.write_text(json.dumps({"vod_id": "123", "video_path": str(vod_media)}) + "\n", encoding="utf-8")
            reposts.write_text(json.dumps({"video_path": str(repost_media), "platform": "youtube"}) + "\n", encoding="utf-8")
            output.write_text(
                json.dumps(
                    {
                        "dataset_source": "cross_platform_audio_fingerprint",
                        "label": 1,
                        "vod_id": "old",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,ytsearch500:jynxzi shorts,en,jynxzi,2\n",
                encoding="utf-8",
            )
            rebuilt_rows = [
                {
                    "video_path": str(repost_media),
                    "platform": "youtube",
                    "source_url": "https://www.youtube.com/shorts/good",
                    "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
                }
            ]
            matched_rows = [
                {
                    "dataset_source": "cross_platform_audio_fingerprint",
                    "label": 1,
                    "vod_id": "new",
                    "candidate_id": "new:1",
                    "window_start": 1.0,
                    "window_end": 2.0,
                }
            ]

            with patch.object(
                prepare,
                "build_repost_manifest_from_sources",
                return_value=(rebuilt_rows, {"downloaded_or_cached_rows": 1}),
            ), patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=(matched_rows, {"best_match_rows": [], "confidence_distribution": {"count": 1}}),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=root / "cache",
                    repost_source_manifest=source_manifest,
                    repost_url_manifest=root / "missing_urls.txt",
                    full_vod_audio_fingerprints=False,
                    min_reposts=1,
                    min_matched_rows=2,
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=False,
                )

            auto_summary = payload["auto_manifest_summary"]
            manifest_rows = [json.loads(line) for line in reposts.read_text(encoding="utf-8").splitlines() if line.strip()]
            output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(auto_summary["repost_manifest_profile_filter_rebuild_required"])
            self.assertTrue(auto_summary["repost_manifest_built"])
            self.assertEqual(auto_summary["repost_manifest_build_reason"], "profile_filter_policy")
            self.assertEqual(manifest_rows, rebuilt_rows)
            self.assertEqual(payload["status"], "insufficient_matched_rows")
            self.assertFalse(payload["preserved_existing_output"])
            self.assertEqual(output_rows, matched_rows)

    def test_curated_youtube_channel_rebuilds_repost_manifest_for_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vod_media = root / "vod.mp4"
            old_repost_media = root / "old-repost.mp4"
            new_repost_media = root / "new-repost.mp4"
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            source_manifest = root / "sources.csv"
            vod_media.write_bytes(b"vod")
            old_repost_media.write_bytes(b"old")
            new_repost_media.write_bytes(b"new")
            vods.write_text(json.dumps({"vod_id": "123", "video_path": str(vod_media)}) + "\n", encoding="utf-8")
            reposts.write_text(
                json.dumps(
                    {
                        "video_path": str(old_repost_media),
                        "platform": "youtube",
                        "source_url": "https://www.youtube.com/watch?v=old",
                        "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_manifest.write_text(
                "streamer,platform,source_url,language,tags,max_clips\n"
                "jynxzi,youtube,https://www.youtube.com/@Iboclipss/shorts,en,"
                "curated_repost_source;iboclipss;youtube_shorts,3600\n",
                encoding="utf-8",
            )
            rebuilt_rows = [
                {
                    "video_path": str(new_repost_media),
                    "platform": "youtube",
                    "source_url": "https://www.youtube.com/watch?v=new",
                    "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
                }
            ]
            matched_rows = [
                {
                    "dataset_source": "cross_platform_audio_fingerprint",
                    "label": 1,
                    "vod_id": "new",
                    "candidate_id": "new:1",
                    "window_start": 1.0,
                    "window_end": 2.0,
                }
            ]

            with patch.object(
                prepare,
                "build_repost_manifest_from_sources",
                return_value=(rebuilt_rows, {"downloaded_or_cached_rows": 1}),
            ) as build_mock, patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=(matched_rows, {"best_match_rows": [], "confidence_distribution": {"count": 1}}),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                    vod_cache_root=root / "cache",
                    repost_source_manifest=source_manifest,
                    repost_url_manifest=root / "missing_urls.txt",
                    full_vod_audio_fingerprints=False,
                    min_reposts=1,
                    min_matched_rows=2,
                    max_reposts=10,
                    discover_reposts=True,
                    download_reposts=False,
                )

            auto_summary = payload["auto_manifest_summary"]
            manifest_rows = [json.loads(line) for line in reposts.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(build_mock.call_count, 1)
            self.assertTrue(auto_summary["repost_manifest_curated_source_freshness_required"])
            self.assertTrue(auto_summary["repost_manifest_built"])
            self.assertEqual(auto_summary["repost_manifest_build_reason"], "curated_source_freshness")
            self.assertEqual(manifest_rows, rebuilt_rows)

    def test_matching_replaces_output_and_dedupes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"tiktok"}\n', encoding="utf-8")
            output.write_text('{"dataset_source":"stale","label":1}\n', encoding="utf-8")
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-a",
                "window_start": 10.0,
                "window_end": 20.0,
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint",
            }

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=(
                    [matched, dict(matched)],
                    {"best_match_rows": [matched], "top_best_matches": [matched], "confidence_distribution": {"count": 1}},
                ),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=summary,
                    min_confidence=0.8,
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(payload["status"], "matched")
            self.assertEqual(payload["matched_rows"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["min_confidence"], 0.8)

    def test_default_verification_mode_uses_transcript_matcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-t",
                "window_start": 10.0,
                "window_end": 20.0,
                "label": 1,
                "dataset_source": "cross_platform_transcript_match",
                "label_source": "cross_platform_transcript_match",
                "transcript_similarity": 0.82,
                "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
            }

            with patch.object(
                prepare.transcript_matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=(
                    [matched],
                    {
                        "verification_mode": "transcript",
                        "best_match_rows": [matched],
                        "confidence_distribution": {"count": 1},
                    },
                ),
            ) as transcript_mock, patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                side_effect=AssertionError("audio matcher must not be used by default"),
            ), patch.object(
                prepare,
                "build_full_vod_transcript_caches",
                return_value={"status": "passed", "processed_vods": 1, "cached_transcripts": 1},
            ) as transcript_cache_mock:
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=summary,
                    min_confidence=0.8,
                    auto_build_manifests=False,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(transcript_mock.call_count, 1)
            self.assertEqual(transcript_cache_mock.call_count, 1)
            self.assertEqual(payload["verification_mode"], "transcript")
            self.assertEqual(payload["full_vod_transcript_summary"]["processed_vods"], 1)
            self.assertEqual(payload["status"], "matched")
            self.assertEqual(rows, [matched])

    def test_transcript_matching_reuses_unchanged_match_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            diagnostics = root / "diagnostics.jsonl"
            vod_rows = [{"vod_id": "123", "video_path": "/tmp/vod.mp4"}]
            repost_rows = [{"video_path": "/tmp/repost.mp4", "platform": "youtube"}]
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-t",
                "window_start": 10.0,
                "window_end": 20.0,
                "label": 1,
                "dataset_source": "cross_platform_transcript_match",
                "label_source": "cross_platform_transcript_match",
                "transcript_similarity": 0.82,
                "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
            }
            vods.write_text("\n".join(json.dumps(row) for row in vod_rows), encoding="utf-8")
            reposts.write_text("\n".join(json.dumps(row) for row in repost_rows), encoding="utf-8")
            output.write_text(json.dumps(matched) + "\n", encoding="utf-8")
            diagnostics.write_text(json.dumps({"accepted": True, "vod_id": "123"}) + "\n", encoding="utf-8")
            match_signature = prepare._cross_platform_match_input_signature(
                vod_rows=vod_rows,
                repost_rows=repost_rows,
                verification_mode="transcript",
                min_confidence=0.8,
                vod_cache_root=root / "vod_cache",
            )
            summary.write_text(
                json.dumps(
                    {
                        "status": "insufficient_matched_rows",
                        "matched_rows": 1,
                        "matched_vods": 1,
                        "matched_rows_needed": 69,
                        "matched_vods_needed": 0,
                        "insufficient_reasons": ["matched_rows"],
                        "match_diagnostics": {"verification_mode": "transcript"},
                        "match_input_signature": match_signature,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                prepare,
                "build_full_vod_transcript_caches",
                side_effect=AssertionError("transcript cache should not be rebuilt"),
            ), patch.object(
                prepare.transcript_matcher,
                "match_reposts_to_vods_with_diagnostics",
                side_effect=AssertionError("transcript matching should be skipped"),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=summary,
                    diagnostic_output_path=diagnostics,
                    min_confidence=0.8,
                    min_matched_rows=70,
                    auto_build_manifests=False,
                    vod_cache_root=root / "vod_cache",
                )

            self.assertEqual(payload["status"], "insufficient_matched_rows")
            self.assertTrue(payload["reused_match_result"])
            self.assertEqual(payload["reuse_reason"], "match_input_signature_unchanged")
            self.assertEqual(payload["matched_rows"], 1)
            self.assertEqual(payload["diagnostic_rows"], 1)
            self.assertEqual(output.read_text(encoding="utf-8"), json.dumps(matched) + "\n")

    def test_transcript_mode_does_not_preserve_old_audio_rows_when_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            output.write_text(
                json.dumps(
                    {
                        "dataset_source": "cross_platform_audio_fingerprint",
                        "label_source": "cross_platform_audio_fingerprint",
                        "fingerprint_confidence": 0.9,
                        "label": 1,
                        "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-t",
                "window_start": 10.0,
                "window_end": 20.0,
                "label": 1,
                "dataset_source": "cross_platform_transcript_match",
                "label_source": "cross_platform_transcript_match",
                "transcript_similarity": 0.82,
                "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
            }

            with patch.object(
                prepare.transcript_matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=(
                    [matched],
                    {"verification_mode": "transcript", "best_match_rows": [matched]},
                ),
            ), patch.object(
                prepare,
                "build_full_vod_transcript_caches",
                return_value={"status": "passed", "processed_vods": 1, "cached_transcripts": 1},
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    min_confidence=0.6,
                    min_matched_rows=70,
                    auto_build_manifests=False,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(payload["status"], "insufficient_matched_rows")
            self.assertFalse(payload["preserved_existing_output"])
            self.assertEqual(payload["output_rows"], 1)
            self.assertEqual(rows, [matched])

    def test_reuses_existing_output_when_minimum_is_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            output.write_text(
                json.dumps(
                    {
                        "dataset_source": "cross_platform_audio_fingerprint",
                        "label_source": "cross_platform_audio_fingerprint",
                        "fingerprint_confidence": 0.73,
                        "label": 1,
                        "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                prepare,
                "_build_missing_manifests",
                side_effect=AssertionError("manifest build should be skipped"),
            ), patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                side_effect=AssertionError("matching should be skipped"),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=root / "missing_vods.jsonl",
                    repost_manifest=root / "missing_reposts.jsonl",
                    output_path=output,
                    summary_path=summary,
                    min_confidence=0.5,
                    min_matched_rows=1,
                    refresh_reposts=False,
                    auto_build_manifests=True,
                    verification_mode="audio_fingerprint",
                )

            self.assertEqual(payload["status"], "reused_existing_output")
            self.assertEqual(payload["matched_rows"], 1)
            self.assertEqual(payload["matched_vods"], 0)
            self.assertTrue(payload["preserved_existing_output"])
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["status"], "reused_existing_output")

    def test_existing_output_with_stale_profile_filter_version_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text(
                json.dumps(
                    {
                        "video_path": "/tmp/repost.mp4",
                        "platform": "youtube",
                        "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "vod_id": "old-vod",
                        "dataset_source": "cross_platform_audio_fingerprint",
                        "label_source": "cross_platform_audio_fingerprint",
                        "fingerprint_confidence": 0.9,
                        "label": 1,
                        "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION - 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-b",
                "window_start": 12.0,
                "window_end": 22.0,
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint",
                "fingerprint_confidence": 0.82,
                "profile_filter_version": prepare.REPOST_PROFILE_FILTER_VERSION,
            }

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=([matched], {"best_match_rows": [matched], "confidence_distribution": {"count": 1}}),
            ) as match_mock:
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    min_confidence=0.5,
                    min_matched_rows=1,
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                )

            self.assertEqual(match_mock.call_count, 1)
            self.assertEqual(payload["status"], "matched")
            self.assertFalse(payload["preserved_existing_output"])

    def test_existing_output_is_not_reused_when_vod_minimum_is_not_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            output.write_text(
                json.dumps(
                    {
                        "vod_id": "old-vod",
                        "dataset_source": "cross_platform_audio_fingerprint",
                        "label_source": "cross_platform_audio_fingerprint",
                        "fingerprint_confidence": 0.9,
                        "label": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-b",
                "window_start": 12.0,
                "window_end": 22.0,
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint",
                "fingerprint_confidence": 0.82,
            }

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=([matched], {"best_match_rows": [matched], "confidence_distribution": {"count": 1}}),
            ) as match_mock:
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    min_confidence=0.5,
                    min_matched_rows=1,
                    min_matched_vods=2,
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                )

            self.assertEqual(match_mock.call_count, 1)
            self.assertEqual(payload["status"], "insufficient_matched_vods")
            self.assertEqual(payload["matched_vods"], 1)
            self.assertEqual(payload["matched_vods_needed"], 1)

    def test_refresh_reposts_forces_matching_even_when_existing_output_is_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            output.write_text(
                '{"dataset_source":"cross_platform_audio_fingerprint","fingerprint_confidence":0.9,"label":1}\n',
                encoding="utf-8",
            )
            matched = {
                "vod_id": "123",
                "candidate_id": "clip-b",
                "window_start": 12.0,
                "window_end": 22.0,
                "label": 1,
                "dataset_source": "cross_platform_audio_fingerprint",
                "fingerprint_confidence": 0.82,
            }

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=([matched], {"best_match_rows": [matched], "confidence_distribution": {"count": 1}}),
            ) as match_mock:
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    min_confidence=0.5,
                    min_matched_rows=1,
                    refresh_reposts=True,
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                )

            self.assertEqual(payload["status"], "matched")
            self.assertEqual(match_mock.call_count, 1)

    def test_prepare_writes_below_threshold_diagnostics_when_no_rows_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            summary = root / "summary.json"
            diagnostics = root / "diagnostics.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"youtube"}\n', encoding="utf-8")
            best_match = {
                "accepted": False,
                "confidence": 0.56,
                "vod_id": "123",
                "window_start": 10.0,
                "window_end": 20.0,
            }
            match_diagnostics = {
                "best_match_rows": [best_match],
                "top_best_matches": [best_match],
                "confidence_distribution": {"count": 1, "median": 0.56, "p95": 0.56},
                "threshold_ladder": {"0.55": 1, "0.72": 0},
                "recommendation": {"status": "consider_lowering_threshold"},
            }

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                return_value=([], match_diagnostics),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=summary,
                    diagnostic_output_path=diagnostics,
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                )

            diagnostic_rows = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()]
            written_summary = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "no_matches_below_threshold")
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertEqual(diagnostic_rows, [best_match])
            self.assertEqual(written_summary["match_diagnostics"]["recommendation"]["status"], "consider_lowering_threshold")
            self.assertNotIn("best_match_rows", written_summary["match_diagnostics"])

    def test_missing_fpcalc_is_blocked_but_does_not_abort_or_delete_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vods = root / "vods.jsonl"
            reposts = root / "reposts.jsonl"
            output = root / "rows.jsonl"
            vods.write_text('{"vod_id":"123","video_path":"/tmp/vod.mp4"}\n', encoding="utf-8")
            reposts.write_text('{"video_path":"/tmp/repost.mp4","platform":"tiktok"}\n', encoding="utf-8")
            output.write_text('{"dataset_source":"cross_platform_audio_fingerprint","label":1}\n', encoding="utf-8")

            with patch.object(
                prepare.matcher,
                "match_reposts_to_vods_with_diagnostics",
                side_effect=RuntimeError("fpcalc not found. Install chromaprint-tools."),
            ):
                payload = prepare.prepare_cross_platform_fingerprint_rows(
                    vod_manifest=vods,
                    repost_manifest=reposts,
                    output_path=output,
                    summary_path=root / "summary.json",
                    auto_build_manifests=False,
                    verification_mode="audio_fingerprint",
                    reuse_existing_output=False,
                )

            self.assertEqual(payload["status"], "blocked_missing_fpcalc")
            self.assertEqual(payload["output_rows"], 1)
            self.assertTrue(payload["preserved_existing_output"])


if __name__ == "__main__":
    unittest.main()
