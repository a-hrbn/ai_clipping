from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml.collect_twitch_clip_labels import TwitchClient, _get_clips, _get_user, _get_vods


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-verify Twitch API access and clip/VOD availability.")
    parser.add_argument("--broadcaster", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not settings.twitch_client_id or not settings.twitch_client_secret:
        print("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be configured.")
        return 1

    client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
    user = _get_user(client, args.broadcaster)
    vods = _get_vods(client, str(user["id"]), args.date_from, args.date_to)
    clips = _get_clips(client, str(user["id"]), args.date_from, args.date_to)

    linked_clips = [clip for clip in clips if str(clip.get("video_id") or "").strip()]
    with_offsets = [clip for clip in linked_clips if clip.get("vod_offset") not in (None, "")]
    summary = {
        "broadcaster": args.broadcaster,
        "user_id": str(user["id"]),
        "vod_count": len(vods),
        "clip_count": len(clips),
        "clips_with_video_id": len(linked_clips),
        "clips_with_vod_offset": len(with_offsets),
        "sample_vod_ids": [str(vod["id"]) for vod in vods[:5]],
        "sample_clips": [
            {
                "id": str(clip.get("id") or ""),
                "video_id": str(clip.get("video_id") or ""),
                "vod_offset": clip.get("vod_offset"),
                "view_count": int(clip.get("view_count") or 0),
            }
            for clip in clips[:5]
        ],
    }
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
