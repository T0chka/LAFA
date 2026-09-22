FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && uv sync --frozen --no-dev --no-install-project

COPY config.py ./
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p /data /embeddings /work /root/.cache/huggingface /root/.cache/torch

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-u", "-m", "scripts.lafa"]
