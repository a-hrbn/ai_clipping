import pathlib
import ffmpeg


def extract(
    video_path: pathlib.Path,
    job_id: str,
    *,
    output_path: pathlib.Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> pathlib.Path:
    audio_path = output_path or (video_path.parent / "audio.wav")
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    input_kwargs: dict[str, float] = {}
    if start_sec is not None:
        input_kwargs["ss"] = max(float(start_sec), 0.0)

    output_kwargs: dict[str, object] = {
        "ac": 1,          # mono
        "ar": 16000,      # 16kHz — required by Whisper
        "acodec": "pcm_s16le",
    }
    if start_sec is not None and end_sec is not None:
        duration = max(float(end_sec) - max(float(start_sec), 0.0), 0.0)
        if duration > 0.0:
            output_kwargs["t"] = duration
    elif end_sec is not None:
        output_kwargs["to"] = max(float(end_sec), 0.0)

    (
        ffmpeg
        .input(str(video_path), **input_kwargs)
        .output(
            str(audio_path),
            **output_kwargs,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return audio_path
