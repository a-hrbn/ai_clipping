FROM python:3.12-bookworm

ARG NODE_MAJOR=20
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    WORKSPACE=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        less \
        libsndfile1 \
        openssh-server \
        pkg-config \
        procps \
        supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /tmp/build

COPY backend/requirements.txt backend/requirements.txt
COPY backend/requirements-base.txt backend/requirements-base.txt
COPY backend/requirements-ml.txt backend/requirements-ml.txt
COPY backend/requirements-dev.txt backend/requirements-dev.txt
COPY frontend/package.json frontend/package-lock.json frontend/

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "setuptools<81" wheel joblib \
    && python -m pip install --no-build-isolation --no-cache-dir \
        -r backend/requirements-base.txt \
        -r backend/requirements-ml.txt \
        -r backend/requirements-dev.txt \
    && graphify install \
    && cd frontend \
    && npm ci \
    && npm install -g @openai/codex \
    && npm cache clean --force

RUN useradd --create-home --shell /bin/bash dev \
    && mkdir -p /run/sshd /var/log/supervisor /workspace /home/dev/.ssh \
    && chown -R dev:dev /workspace /home/dev

COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/sshd_config /etc/ssh/sshd_config
COPY docker/ssh/authorized_keys.example /opt/dev/authorized_keys.example

RUN chmod 755 /usr/local/bin/docker-entrypoint.sh \
    && chmod 600 /opt/dev/authorized_keys.example

WORKDIR /workspace

EXPOSE 22 3000 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
