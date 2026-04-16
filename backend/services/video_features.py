"""
Frame-by-frame computer vision feature extraction.

Runs in parallel with audio_features.extract() and transcriber.transcribe().
Produces one VideoFeatureFrame per second of video, then merges into RichSegment
via enrich_segments() — identical pattern to audio_features.

Features extracted (CPU-only, no GPU required):
  scene_change      — HSV histogram diff detects hard cuts / overlay events
  motion_score      — Farneback optical flow at 25%-scale → physical reactions
  face_count        — MediaPipe face detection → number of visible faces
  face_size_ratio   — largest face bbox area / frame area → close-up vs. wide shot
  face_reaction_prob— DeepFace surprise+fear+disgust, sampled every 3s → visible shock
  visual_activity   — composite of motion + histogram variance

Performance budget for a 1-hour video on Mac CPU:
  scene change + histogram  ~4s
  optical flow (25% scale)  ~30s
  MediaPipe face detection  ~55s
  DeepFace (every 3s)       ~100s
  ─────────────────────────────
  Total                     ~3 min (finishes within Whisper's 10-15 min window)
"""

from __future__ import annotations

import os
import sys
import logging
import pathlib
from dataclasses import dataclass, field
from statistics import mean

import numpy as np

from config import settings
from services.timeline_types import Candidate
from services.transcriber import RichSegment

logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_VIDEO_RUNTIME_LOGGED = False
_MEDIAPIPE_STATUS = "unavailable"
_MEDIAPIPE_DETAIL = ""
_MTCNN_STATUS = "unavailable"
_MTCNN_DETAIL = ""
_DEEPFACE_STATUS = "unavailable"
_DEEPFACE_DETAIL = ""

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[video_features] WARNING: opencv-python-headless not installed — CV disabled", file=sys.stderr)

# MediaPipe face detector — preferred over Haar cascade (lower false-positive rate,
# better profile/low-light performance).  Falls back to Haar cascade if unavailable.
_MP_FACE_DETECTION = None
_MP_TASKS_FACE_DETECTOR = None
_MP_IMAGE = None
_MP_IMAGE_FORMAT = None


try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        _MP_FACE_DETECTION = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        _MEDIAPIPE_STATUS = "solutions_face_detection"
        _MEDIAPIPE_DETAIL = "model_selection=0"
    else:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        _MP_IMAGE = mp.Image
        _MP_IMAGE_FORMAT = mp.ImageFormat
        _MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "face_detection_short_range.tflite"
        if _MODEL_PATH.exists():
            _MP_TASKS_FACE_DETECTOR = FaceDetector.create_from_options(
                FaceDetectorOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(_MODEL_PATH),
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    min_detection_confidence=0.5,
                )
            )
            _MEDIAPIPE_STATUS = "tasks_face_detector"
            _MEDIAPIPE_DETAIL = f"model_asset_path={_MODEL_PATH.name}"
        else:
            raise FileNotFoundError(f"MediaPipe face model missing: {_MODEL_PATH}")
except Exception as e:
    _MEDIAPIPE_DETAIL = str(e)

_MTCNN_DETECTOR = None
try:
    from mtcnn import MTCNN
    _MTCNN_DETECTOR = MTCNN()
    _MTCNN_STATUS = "available"
    _MTCNN_DETAIL = "mtcnn"
except Exception as e:
    _MTCNN_DETAIL = str(e)

_FACE_CASCADE: "cv2.CascadeClassifier | None" = None
if _MP_FACE_DETECTION is None and _MP_TASKS_FACE_DETECTOR is None and _MTCNN_DETECTOR is None and _CV2_AVAILABLE:
    try:
        _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(_cascade_path)
        if _FACE_CASCADE.empty():
            _FACE_CASCADE = None
            print("[video_features] WARNING: Haar cascade not found — face detection disabled", file=sys.stderr)
    except Exception as e:
        print(f"[video_features] WARNING: face cascade init failed: {e}", file=sys.stderr)

try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
    _DEEPFACE_STATUS = "available"
    _DEEPFACE_DETAIL = "detector_backend=opencv"
except Exception:
    _DEEPFACE_AVAILABLE = False
    _DEEPFACE_DETAIL = "not installed or failed to load"


def runtime_summary() -> dict[str, object]:
    if _MP_FACE_DETECTION is not None:
        face_detector = "mediapipe"
    elif _MP_TASKS_FACE_DETECTOR is not None:
        face_detector = "mediapipe_tasks"
    elif _MTCNN_DETECTOR is not None:
        face_detector = "mtcnn"
    elif _FACE_CASCADE is not None:
        face_detector = "haar"
    else:
        face_detector = "none"
    return {
        "face_detector": face_detector,
        "mediapipe_status": _MEDIAPIPE_STATUS,
        "mediapipe_detail": _MEDIAPIPE_DETAIL,
        "mtcnn_status": _MTCNN_STATUS,
        "mtcnn_detail": _MTCNN_DETAIL,
        "deepface_enabled": _DEEPFACE_AVAILABLE,
        "deepface_status": _DEEPFACE_STATUS,
        "deepface_detail": _DEEPFACE_DETAIL,
        "expression_sample_sec": max(settings.video_expression_sample_sec, 1),
    }


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class VideoFeatureFrame:
    second: int
    scene_change: bool = False      # histogram diff vs. previous sampled frame > threshold
    motion_score: float = 0.0       # 0.0–1.0, optical flow magnitude normalised to video 95th pct
    face_count: int = 0             # number of faces detected by MediaPipe
    face_size_ratio: float = 0.0    # largest face bbox area / frame area (0.0–1.0)
    face_reaction_prob: float = 0.0 # DeepFace surprise+fear+disgust combined (every 3s, carried fwd)
    visual_activity: float = 0.0    # composite: motion_score * 0.6 + histogram_variance * 0.4


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract(video_path: pathlib.Path) -> list[VideoFeatureFrame]:
    """
    Extract per-second visual features from a video file.

    Returns an empty list (not an exception) when CV libraries are unavailable
    or the video cannot be opened, so the pipeline degrades gracefully.
    """
    global _VIDEO_RUNTIME_LOGGED
    if not _CV2_AVAILABLE:
        return []

    if not _VIDEO_RUNTIME_LOGGED:
        summary = runtime_summary()
        logger.info(
            "video_runtime face_detector=%s mediapipe=%s mtcnn=%s deepface=%s expression_sample_sec=%s",
            summary["face_detector"],
            summary["mediapipe_status"],
            summary["mtcnn_status"],
            summary["deepface_status"],
            summary["expression_sample_sec"],
        )
        _VIDEO_RUNTIME_LOGGED = True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[video_features] Could not open {video_path}", file=sys.stderr)
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps * 1000
        duration_sec = int(total_ms / 1000)

        # --- First pass: extract raw frames and raw metrics ----------------
        raw_frames: list[tuple[int, np.ndarray]] = []  # (second, bgr_frame)
        raw_motion: list[float] = []

        use_mediapipe = _MP_FACE_DETECTION is not None or _MP_TASKS_FACE_DETECTOR is not None
        use_mtcnn = _MTCNN_DETECTOR is not None
        face_detector = _FACE_CASCADE  # used only when MediaPipe unavailable
        if use_mediapipe:
            detector_name = "mediapipe"
        elif use_mtcnn:
            detector_name = "mtcnn"
        elif face_detector is not None:
            detector_name = "haar"
        else:
            detector_name = "none"
        logger.info("video_features face_detector=%s path=%s", detector_name, video_path)

        frames_data: list[VideoFeatureFrame] = []
        prev_gray_small: np.ndarray | None = None
        prev_hist: np.ndarray | None = None
        last_reaction: float = 0.0

        for sec in range(duration_sec):
            frame = _seek_frame(cap, sec)
            if frame is None:
                frames_data.append(VideoFeatureFrame(second=sec))
                raw_motion.append(0.0)
                continue

            vf = VideoFeatureFrame(second=sec)

            # Scene change
            curr_hist = _hsv_hist(frame)
            if prev_hist is not None:
                dist = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
                vf.scene_change = bool(dist > 0.35)
            prev_hist = curr_hist

            # Motion (optical flow on 25%-scale grayscale)
            try:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray_small is not None and prev_gray_small.shape == gray_small.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray_small, gray_small, None,
                        pyr_scale=0.5, levels=2, winsize=11,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
                    )
                    mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
                    vf.motion_score = mag  # raw, normalised later
                    raw_motion.append(mag)
                else:
                    raw_motion.append(0.0)
                prev_gray_small = gray_small
            except Exception:
                logger.debug("Motion extraction failed at second %s", sec, exc_info=True)
                raw_motion.append(0.0)

            # Face detection — prefer MediaPipe, fall back to Haar cascade
            try:
                if use_mediapipe:
                    count, size = _detect_faces_mediapipe(frame)
                elif use_mtcnn:
                    count, size = _detect_faces_mtcnn(frame)
                elif face_detector is not None:
                    count, size = _detect_faces(face_detector, frame)
                else:
                    count, size = 0, 0.0
                vf.face_count = count
                vf.face_size_ratio = size
            except Exception:
                logger.debug("Face detection failed at second %s", sec, exc_info=True)

            # Expression — sample sparsely; DeepFace dominates runtime on long VODs.
            if _DEEPFACE_AVAILABLE and sec % max(settings.video_expression_sample_sec, 1) == 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv",
                        silent=True,
                    )
                    if isinstance(result, list):
                        result = result[0]
                    emotions = result.get("emotion", {})
                    # Sum the three "reaction" emotions, cap at 1.0.
                    # Not averaged — a single strong surprise at 80% should score
                    # close to 1.0, not 0.27.
                    last_reaction = min(
                        float(
                            emotions.get("surprise", 0) / 100
                            + emotions.get("fear",     0) / 100
                            + emotions.get("disgust",  0) / 100
                        ),
                        1.0,
                    )
                except Exception:
                    logger.debug("DeepFace analysis failed at second %s", sec, exc_info=True)
            vf.face_reaction_prob = last_reaction

            frames_data.append(vf)

        cap.release()

        # --- Second pass: normalise motion to 0–1 using 95th percentile ---
        if raw_motion:
            p95 = float(np.percentile(raw_motion, 95)) if max(raw_motion) > 0 else 1.0
            p95 = max(p95, 1e-6)
            for vf in frames_data:
                vf.motion_score = min(vf.motion_score / p95, 1.0)

        # --- Compute visual_activity composite ----------------------------
        for vf in frames_data:
            hist_var = 0.3 if vf.scene_change else 0.0  # scene change implies high variance
            vf.visual_activity = round(
                vf.motion_score * 0.6 + hist_var * 0.4,
                3,
            )

        return frames_data

    except Exception as e:
        print(f"[video_features] Extraction failed: {e}", file=sys.stderr)
        cap.release()
        return []


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_segments(
    segments: list[RichSegment],
    frames: list[VideoFeatureFrame],
) -> list[RichSegment]:
    """
    Merge per-second VideoFeatureFrame data into RichSegment objects and
    append visual emotion tags.  No-op when frames is empty.
    """
    if not frames:
        return segments

    # Build second → frame lookup
    frame_map: dict[int, VideoFeatureFrame] = {f.second: f for f in frames}

    # Compute global motion threshold (sigma-based, same as audio_features)
    all_motion = [f.motion_score for f in frames]
    motion_threshold = _threshold(all_motion, sigma=1.3)

    for seg in segments:
        seg_frames = [
            frame_map[s]
            for s in range(int(seg.start), int(seg.end) + 1)
            if s in frame_map
        ]
        if not seg_frames:
            continue

        seg.has_scene_change = any(f.scene_change for f in seg_frames)
        seg.max_motion_score = max(f.motion_score for f in seg_frames)
        seg.max_face_count   = max(f.face_count   for f in seg_frames)
        seg.max_face_reaction= max(f.face_reaction_prob for f in seg_frames)
        seg.visual_activity  = max(f.visual_activity for f in seg_frames)

        face_sizes = [f.face_size_ratio for f in seg_frames if f.face_size_ratio > 0]
        seg.mean_face_size = mean(face_sizes) if face_sizes else 0.0

        # Emotion tags
        if seg.max_motion_score >= motion_threshold:
            seg.emotions.append("high_motion")
        if seg.max_face_reaction >= 0.35:
            seg.emotions.append("face_reaction")
        if seg.has_scene_change:
            seg.emotions.append("scene_change")
        if seg.max_face_count >= 2:
            seg.emotions.append("multi_face")
        if seg.mean_face_size >= 0.15:
            seg.emotions.append("prominent_face")

    return segments


def refine_face_reactions(
    segments: list[RichSegment],
    video_path: pathlib.Path,
    candidates: list[Candidate],
    sample_every_sec: int = 1,
) -> list[RichSegment]:
    """
    Re-run only the expensive face reaction step over shortlist candidate windows.

    Intended for a second pass after coarse ranking so long VODs avoid running
    DeepFace over the full duration.
    """
    if not segments or not candidates or not _CV2_AVAILABLE or not _DEEPFACE_AVAILABLE:
        return segments

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return segments

    try:
        target_seconds = _candidate_seconds(candidates, sample_every_sec=max(sample_every_sec, 1))
        if not target_seconds:
            return segments

        reaction_by_second: dict[int, float] = {}
        last_reaction = 0.0
        for sec in target_seconds:
            frame = _seek_frame(cap, sec)
            if frame is None:
                continue
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]
                emotions = result.get("emotion", {})
                last_reaction = min(
                    float(
                        emotions.get("surprise", 0) / 100
                        + emotions.get("fear", 0) / 100
                        + emotions.get("disgust", 0) / 100
                    ),
                    1.0,
                )
            except Exception:
                logger.debug("Shortlist DeepFace analysis failed at second %s", sec, exc_info=True)
            reaction_by_second[sec] = last_reaction

        if not reaction_by_second:
            return segments

        target_ranges = [(candidate.start, candidate.end) for candidate in candidates]
        for seg in segments:
            if not _segment_intersects_any(seg, target_ranges):
                continue
            sec_values = [
                reaction_by_second[sec]
                for sec in range(int(seg.start), int(seg.end) + 1)
                if sec in reaction_by_second
            ]
            if not sec_values:
                continue
            seg.max_face_reaction = max(seg.max_face_reaction, max(sec_values))
            _set_tag(seg, "face_reaction", seg.max_face_reaction >= 0.35)

        return segments
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seek_frame(cap: "cv2.VideoCapture", second: int) -> "np.ndarray | None":
    """Seek to `second` and decode one frame. Returns None on failure."""
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
        ret, frame = cap.read()
        return frame if ret else None
    except Exception:
        return None


def _candidate_seconds(
    candidates: list[Candidate],
    *,
    sample_every_sec: int,
) -> list[int]:
    seconds: set[int] = set()
    step = max(sample_every_sec, 1)
    for candidate in candidates:
        start = max(0, int(candidate.start))
        end = max(start, int(candidate.end))
        for sec in range(start, end + 1, step):
            seconds.add(sec)
        seconds.add(end)
    return sorted(seconds)


def _segment_intersects_any(seg: RichSegment, ranges: list[tuple[float, float]]) -> bool:
    for start, end in ranges:
        if seg.end > start and seg.start < end:
            return True
    return False


def _set_tag(seg: RichSegment, tag: str, enabled: bool) -> None:
    if enabled:
        if tag not in seg.emotions:
            seg.emotions.append(tag)
        return
    if tag in seg.emotions:
        seg.emotions = [item for item in seg.emotions if item != tag]


def _hsv_hist(frame: "np.ndarray") -> "np.ndarray":
    """Compute a compact HSV histogram for scene change detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [64, 64], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _detect_faces_mediapipe(frame: "np.ndarray") -> tuple[int, float]:
    """
    Detect faces using MediaPipe Face Detection.
    Returns (face_count, largest_face_size_ratio).
    Handles profile faces and low-light better than the Haar cascade.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if _MP_FACE_DETECTION is not None:
        results = _MP_FACE_DETECTION.process(rgb)
        if not results.detections:
            return 0, 0.0
        max_area = 0.0
        for detection in results.detections:
            bb = detection.location_data.relative_bounding_box
            area = bb.width * bb.height
            if area > max_area:
                max_area = area
        return len(results.detections), round(min(max_area, 1.0), 4)

    if _MP_TASKS_FACE_DETECTOR is None or _MP_IMAGE is None or _MP_IMAGE_FORMAT is None:
        return 0, 0.0

    mp_image = _MP_IMAGE(image_format=_MP_IMAGE_FORMAT.SRGB, data=rgb)
    result = _MP_TASKS_FACE_DETECTOR.detect(mp_image)
    detections = getattr(result, "detections", None) or []
    if not detections:
        return 0, 0.0

    h, w = frame.shape[:2]
    frame_area = max(float(h * w), 1.0)
    max_area = 0.0
    for detection in detections:
        bb = detection.bounding_box
        area = (float(bb.width) * float(bb.height)) / frame_area
        if area > max_area:
            max_area = area
    return len(detections), round(min(max_area, 1.0), 4)


def _detect_faces(cascade: "cv2.CascadeClassifier", frame: "np.ndarray") -> tuple[int, float]:
    """
    Detect frontal faces using OpenCV Haar cascade (built into cv2, no extra deps).
    Returns (face_count, largest_face_size_ratio).
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if not len(faces):
        return 0, 0.0

    frame_area = h * w
    max_area = 0.0
    for (_, _, fw, fh) in faces:
        area = (fw * fh) / frame_area
        if area > max_area:
            max_area = area

    return len(faces), round(min(max_area, 1.0), 4)


def _detect_faces_mtcnn(frame: "np.ndarray") -> tuple[int, float]:
    """
    Detect faces using MTCNN as a CPU-friendly fallback when MediaPipe is not
    usable on this runtime.
    """
    if _MTCNN_DETECTOR is None:
        return 0, 0.0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = _MTCNN_DETECTOR.detect_faces(rgb)
    if not detections:
        return 0, 0.0

    h, w = frame.shape[:2]
    frame_area = max(float(h * w), 1.0)
    max_area = 0.0
    for detection in detections:
        x, y, fw, fh = detection.get("box", [0, 0, 0, 0])
        area = (max(float(fw), 0.0) * max(float(fh), 0.0)) / frame_area
        if area > max_area:
            max_area = area
    return len(detections), round(min(max_area, 1.0), 4)


def _threshold(values: list[float], sigma: float = 1.5) -> float:
    """Return mean + sigma * std as an adaptive threshold (same as audio_features)."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean() + sigma * arr.std())
