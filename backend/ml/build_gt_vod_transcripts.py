"""Download Jynxzi VODs, transcribe them, and write transcript + fingerprint caches.

Reads jynxzi_gt_vods.csv, then for each VOD:
  1. Downloads via downloader (durable Twitch cache)
  2. Extracts 16kHz mono audio
  3. Transcribes with Whisper (chunk-cached, idempotent)
  4. Writes tmp/vod_feature_cache/<vod_id>/transcript.json  (version=1)
  5. Pre-computes audio fingerprint → fingerprint.json  (for audio fallback)
  6. Writes jynxzi_gt_vod_manifest.jsonl for the matcher

Usage:
  python ml/build_gt_vod_transcripts.py --vods ml/jynxzi_gt_vods.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import shutil
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import gt_transcript_artifact
from ml import match_cross_platform_audio_fingerprints as audio_matcher
from services import audio_extractor, downloader, transcriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VOD_CSV = BACKEND_ROOT / "ml" / "jynxzi_gt_vods.csv"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
DEFAULT_MANIFEST_OUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gt_vod_manifest.jsonl"
_TRANSCRIPT_CACHE_VERSION = gt_transcript_artifact.TRANSCRIPT_CACHE_VERSION
_SERIALIZE_FIELDS = gt_transcript_artifact.SERIALIZE_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download VODs, transcribe, and build VOD transcript + fingerprint caches."
    )
    parser.add_argument("--vods", default=str(DEFAULT_VOD_CSV))
    parser.add_argument("--vod-cache-root", default=str(DEFAULT_VOD_CACHE_ROOT))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST_OUT))
    args = parser.parse_args()

    vod_rows = _read_vod_csv(pathlib.Path(args.vods))
    if not vod_rows:
        logger.error("No VOD rows in %s", args.vods)
        return 1

    vod_cache_root = pathlib.Path(args.vod_cache_root)
    manifest_out = pathlib.Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    fpcalc = shutil.which("fpcalc")
    if fpcalc:
        logger.info("fpcalc found at %s — audio fingerprints enabled", fpcalc)
    else:
        logger.warning("fpcalc not found — audio fingerprints disabled (transcript matching still works)")

    manifest_rows: list[dict] = []
    for row in vod_rows:
        vod_id = row["vod_id"]
        vod_url = f"https://www.twitch.tv/videos/{vod_id}"
        cache_dir = vod_cache_root / vod_id
        transcript_path = cache_dir / "transcript.json"
        fingerprint_path = cache_dir / "fingerprint.json"

        if transcript_path.exists():
            logger.info("VOD %s: transcript cache exists — skipping", vod_id)
        else:
            try:
                _process_vod(vod_id, vod_url, cache_dir, transcript_path, fpcalc)
            except Exception:
                logger.exception("VOD %s: processing failed — skipping", vod_id)
                continue

        manifest_row: dict = {"vod_id": vod_id, "vod_url": vod_url}
        if fingerprint_path.exists():
            manifest_row["fingerprint_path"] = str(fingerprint_path)
        manifest_rows.append(manifest_row)

    manifest_out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in manifest_rows),
        encoding="utf-8",
    )
    logger.info("Wrote VOD manifest: %d rows → %s", len(manifest_rows), manifest_out)
    return 0 if manifest_rows else 1


def _process_vod(
    vod_id: str,
    vod_url: str,
    cache_dir: pathlib.Path,
    transcript_path: pathlib.Path,
    fpcalc: str | None,
) -> None:
    job_id = f"gt_vod_{vod_id}"
    logger.info("VOD %s: downloading...", vod_id)
    video_path = downloader.download(vod_url, job_id)
    logger.info("VOD %s: downloaded → %s", vod_id, video_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / "audio.wav"

    logger.info("VOD %s: extracting audio...", vod_id)
    audio_extractor.extract(video_path, job_id, output_path=audio_path)

    logger.info("VOD %s: transcribing...", vod_id)
    segments = transcriber.transcribe_with_progress(
        audio_path,
        enable_diarization=False,
        cache_key=f"twitch_vod:{vod_id}",
    )
    logger.info("VOD %s: transcribed %d segments", vod_id, len(segments))

    _write_transcript_cache(transcript_path, segments)
    logger.info("VOD %s: transcript written → %s", vod_id, transcript_path)

    if fpcalc and video_path.exists():
        fingerprint_path = cache_dir / "fingerprint.json"
        try:
            audio_matcher.fingerprint_media_to_file(video_path, fingerprint_path, fpcalc)
            logger.info("VOD %s: fingerprint written → %s", vod_id, fingerprint_path)
        except Exception:
            logger.warning("VOD %s: fingerprint computation failed", vod_id, exc_info=True)

    try:
        audio_path.unlink(missing_ok=True)
    except Exception:
        pass


def _write_transcript_cache(path: pathlib.Path, segments: list) -> None:
    gt_transcript_artifact.write_transcript_cache(path, segments)


def _read_vod_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        logger.error("VOD CSV not found: %s", path)
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vod_id = (row.get("vod_id") or "").strip()
            if not vod_id or vod_id.startswith("#"):
                continue
            rows.append({"vod_id": vod_id, "vod_date": (row.get("vod_date") or "").strip()})
    return rows


if __name__ == "__main__":
    sys.exit(main())
