import pathlib
import ffmpeg


def extract(video_path: pathlib.Path, job_id: str) -> pathlib.Path:
    audio_path = video_path.parent / "audio.wav"

    (
        ffmpeg
        .input(str(video_path))
        .output(
            str(audio_path),
            ac=1,          # mono
            ar=16000,      # 16kHz — required by Whisper
            acodec="pcm_s16le",
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return audio_path
