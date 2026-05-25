FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STORAGE_DIR=/app/backend/tmp/storage \
    XDG_CACHE_HOME=/app/backend/tmp/cache \
    HF_HOME=/app/backend/tmp/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/backend/tmp/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/backend/tmp/cache/huggingface/transformers \
    TORCH_HOME=/app/backend/tmp/cache/torch \
    MPLCONFIGDIR=/app/backend/tmp/matplotlib \
    DEEPFACE_HOME=/app/backend/tmp/cache/deepface \
    WHISPER_LANGUAGE=en \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel\;none

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libgles2 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.8.0" \
        "torchaudio>=2.8.0" \
        "torchvision>=0.23.0" \
    && python -m pip install ".[ml,vision]" \
    && mkdir -p \
        /app/backend/tmp/storage \
        /app/backend/tmp/cache \
        /app/backend/tmp/matplotlib \
        /app/backend/ml/artifacts

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
