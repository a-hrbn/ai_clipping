import json
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from urllib import error
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(Whisper=object))

from services import chat_analyzer


class _FakeResponse:
    def __init__(self, payload, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ChatAnalyzerDownloadTests(unittest.TestCase):
    def test_twitch_graphql_headers_use_web_client_id_even_when_env_client_id_is_set(self) -> None:
        with patch.object(chat_analyzer.settings, "twitch_client_id", "not-a-real-client-id"):
            headers = chat_analyzer._twitch_headers()

        self.assertEqual(headers["Client-ID"], chat_analyzer._DEFAULT_TWITCH_WEB_CLIENT_ID)

    def test_fetch_graphql_comments_includes_http_error_body(self) -> None:
        body = b'{"error":"Bad Request","status":400,"message":"The \\"Client-ID\\" header is invalid."}'

        def fake_urlopen(req, **_kwargs):
            raise error.HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=io.BytesIO(body))

        with patch.object(chat_analyzer.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaisesRegex(ValueError, "Client-ID"):
                chat_analyzer._fetch_graphql_comments(
                    "123456",
                    job_id="job-http-400",
                    offset_seconds=0,
                    headers=chat_analyzer._twitch_headers(),
                    timeout_sec=1.0,
                )

    def test_download_chat_fetches_twitch_graphql_rechat_and_writes_normalized_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"
            seen_offsets: list[int] = []

            def fake_urlopen(req, **_kwargs):
                payload = json.loads(req.data.decode("utf-8"))
                seen_offsets.append(payload[0]["variables"]["contentOffsetSeconds"])
                if seen_offsets[-1] == 20:
                    return _FakeResponse(
                        [
                            {
                                "data": {
                                    "video": {
                                        "comments": {
                                            "edges": [
                                                {
                                                    "node": {
                                                        "id": "comment-2",
                                                        "contentOffsetSeconds": 22,
                                                        "commenter": {"displayName": "viewer2"},
                                                        "message": {
                                                            "fragments": [{"text": "no way", "emote": None}],
                                                        },
                                                    }
                                                }
                                            ],
                                            "pageInfo": {"hasNextPage": False},
                                        }
                                    }
                                }
                            }
                        ]
                    )
                return _FakeResponse(
                    [
                        {
                            "data": {
                                "video": {
                                    "comments": {
                                        "edges": [
                                            {
                                                "node": {
                                                    "id": "comment-1",
                                                    "contentOffsetSeconds": 3.25,
                                                    "commenter": {"displayName": "viewer1"},
                                                    "message": {
                                                        "fragments": [
                                                            {
                                                                "text": "KEKW",
                                                                "emote": {"emoteID": "KEKW"},
                                                            },
                                                            {"text": " that was insane", "emote": None},
                                                        ],
                                                    },
                                                }
                                            }
                                        ],
                                        "pageInfo": {"hasNextPage": True},
                                    }
                                }
                            }
                        }
                    ]
                )

            with patch.object(chat_analyzer.request, "urlopen", side_effect=fake_urlopen), \
                 patch.object(chat_analyzer.logger, "info") as info_log:
                path = chat_analyzer.download_chat(
                    "https://www.twitch.tv/videos/123456",
                    "job-1",
                    out_dir,
                    target_offsets=[0, 20],
                )
                rows = json.loads(path.read_text(encoding="utf-8")) if path is not None else []

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.name, "chat.twitch_rechat.json")
        self.assertEqual(seen_offsets, [0, 20])
        self.assertEqual(
            rows,
            [
                {
                    "t": 3.25,
                    "text": "KEKW that was insane",
                    "emotes": ["KEKW"],
                    "id": "comment-1",
                    "user": "viewer1",
                },
                {
                    "t": 22.0,
                    "text": "no way",
                    "emotes": [],
                    "id": "comment-2",
                    "user": "viewer2",
                },
            ],
        )
        info_messages = [call.args[0] for call in info_log.call_args_list]
        self.assertIn(
            "twitch_chat_download_start job_id=%s vod_id=%s url=%s out_dir=%s target_offsets=%s first_offset=%s last_offset=%s",
            info_messages,
        )
        self.assertIn(
            "twitch_graphql_rechat_request_start job_id=%s vod_id=%s offset=%s timeout_sec=%.1f client_id_source=web",
            info_messages,
        )
        self.assertIn(
            "twitch_rechat_fetch_page job_id=%s vod_id=%s page=%s offset=%s edges=%s normalized=%s new_messages=%s total_messages=%s first_t=%.3f last_t=%.3f has_next=%s mode=%s",
            info_messages,
        )
        self.assertIn(
            "twitch_rechat_fetch_done job_id=%s vod_id=%s status=%s pages=%s messages=%s first_t=%.3f last_t=%.3f mode=%s",
            info_messages,
        )
        self.assertIn(
            "twitch_chat_download_success job_id=%s vod_id=%s chat_path=%s messages=%s first_t=%.3f last_t=%.3f partial=%s",
            info_messages,
        )

    def test_download_chat_crawls_twitch_graphql_offsets_without_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"
            seen_offsets: list[int] = []

            def fake_urlopen(req, **_kwargs):
                payload = json.loads(req.data.decode("utf-8"))
                offset = payload[0]["variables"]["contentOffsetSeconds"]
                seen_offsets.append(offset)
                if offset == 0:
                    edges = [
                        {
                            "node": {
                                "id": "comment-1",
                                "contentOffsetSeconds": 3,
                                "commenter": {"displayName": "viewer1"},
                                "message": {"fragments": [{"text": "first", "emote": None}]},
                            }
                        }
                    ]
                    has_next = True
                else:
                    edges = [
                        {
                            "node": {
                                "id": "comment-2",
                                "contentOffsetSeconds": 8,
                                "commenter": {"displayName": "viewer2"},
                                "message": {"fragments": [{"text": "second", "emote": None}]},
                            }
                        }
                    ]
                    has_next = False
                return _FakeResponse(
                    [
                        {
                            "data": {
                                "video": {
                                    "comments": {
                                        "edges": edges,
                                        "pageInfo": {"hasNextPage": has_next},
                                    }
                                }
                            }
                        }
                    ]
                )

            with patch.object(chat_analyzer.request, "urlopen", side_effect=fake_urlopen):
                path = chat_analyzer.download_chat(
                    "https://www.twitch.tv/videos/123456",
                    "job-1b",
                    out_dir,
                )
                rows = json.loads(path.read_text(encoding="utf-8")) if path is not None else []

        self.assertEqual(seen_offsets, [0, 4])
        self.assertEqual([row["text"] for row in rows], ["first", "second"])

    def test_download_chat_treats_missing_twitch_graphql_video_as_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"

            def fake_urlopen(_req, **_kwargs):
                return _FakeResponse([{"data": {"video": None}}])

            with patch.object(chat_analyzer.request, "urlopen", side_effect=fake_urlopen), \
                 patch.object(chat_analyzer.logger, "info") as info_log, \
                 patch.object(chat_analyzer.logger, "warning") as warning_log:
                path = chat_analyzer.download_chat(
                    "https://www.twitch.tv/videos/123456",
                    "job-2",
                    out_dir,
                )

        self.assertIsNone(path)
        warning_log.assert_not_called()
        info_messages = [call.args[0] for call in info_log.call_args_list]
        self.assertIn(
            "twitch_graphql_rechat_unavailable job_id=%s vod_id=%s offset=%s status=%s reason=video_missing",
            info_messages,
        )
        self.assertIn(
            "twitch_chat_download_unavailable job_id=%s vod_id=%s reason=%s",
            info_messages,
        )

    def test_download_chat_treats_missing_twitch_rechat_as_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"

            def fake_urlopen(req, **_kwargs):
                raise error.HTTPError(req.full_url, 404, "Not Found", hdrs=None, fp=None)

            with patch.object(chat_analyzer.request, "urlopen", side_effect=fake_urlopen), \
                 patch.object(chat_analyzer.logger, "info") as info_log, \
                 patch.object(chat_analyzer.logger, "warning") as warning_log:
                path = chat_analyzer.download_chat(
                    "https://www.twitch.tv/videos/123456",
                    "job-2b",
                    out_dir,
                )

        self.assertIsNone(path)
        warning_log.assert_not_called()
        info_messages = [call.args[0] for call in info_log.call_args_list]
        self.assertIn(
            "twitch_graphql_rechat_unavailable job_id=%s vod_id=%s offset=%s status=%s reason=not_found elapsed_ms=%s",
            info_messages,
        )
        self.assertIn(
            "twitch_chat_download_unavailable job_id=%s vod_id=%s reason=%s",
            info_messages,
        )

    def test_download_chat_skips_unsupported_urls_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "chat"
            with patch.object(chat_analyzer.request, "urlopen") as urlopen_mock:
                path = chat_analyzer.download_chat(
                    "https://www.youtube.com/watch?v=abc",
                    "job-3",
                    out_dir,
                )

        self.assertIsNone(path)
        urlopen_mock.assert_not_called()

    def test_load_messages_accepts_normalized_direct_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.twitch_rechat.json"
            path.write_text(
                json.dumps([{"t": 1.5, "text": "KEKW", "emotes": ["KEKW"]}]),
                encoding="utf-8",
            )

            rows = chat_analyzer.load_messages(path)

        self.assertEqual(rows, [{"t": 1.5, "text": "KEKW", "emotes": ["KEKW"]}])


if __name__ == "__main__":
    unittest.main()
