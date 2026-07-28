# ProcGrep explorer — Hugging Face Docker Space (CPU, free tier).
# Intent: serve the FastAPI backend (app.py) + the query frontend on port 7860.
# Design decision: install procgrep from git rather than vendoring it, so the
# Space and the library stay one source of truth (benefit); the image must be
# rebuilt to pick up library changes (price).
FROM python:3.11-slim

WORKDIR /app

# git is needed to pip-install procgrep from its repo (slim has none).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as a non-root user; keep all caches under writable /tmp.
ENV HF_HOME=/tmp/hf \
    HF_DATASETS_CACHE=/tmp/hf/datasets \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY static ./static

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
