# Build a small, reproducible image for the indexing pipeline
# - Python 3.11 (slim)
# - Tesseract OCR runtime + dev libs (so native wheels / plugins work)
# - Installs Python requirements from requirements.txt
# - Copies pipeline and sets entrypoint to main.py
FROM python:3.11-slim

# Noninteractive apt for deterministic builds
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Consolidated environment variables (defaults used if not provided at runtime)
# Keep them here as DOCUMENTED defaults; production values should be passed on `docker run` or in orchestration.
ENV S3_BUCKET="" \
    S3_CHUNKED_PREFIX="data/chunked/" \
    RAW_PREFIX="data/raw/" \
    CHUNKED_PREFIX="data/chunked/" \
    VECTOR_DB="pgvector" \
    EMBED_BATCH_SIZE="32" \
    EMBED_MODEL_ID="amazon.titan-embed-text-v2:0" \
    EMBED_DIM="1024" \
    AWS_REGION="" \
    PG_HOST="localhost" \
    PG_PORT="5432" \
    PG_USER="postgres" \
    PG_PASSWORD="postgres" \
    PG_DB="postgres" \
    PG_INDEX_NAME="documents" \
    OPENSEARCH_HOSTS="http://localhost:9200" \
    OPENSEARCH_INDEX="documents" \
    OPENSEARCH_USER="" \
    OPENSEARCH_PASSWORD="" \
    SEED_URLS="" \
    ALLOWED_DOMAINS="" \
    OCR_LANGUAGES="eng+tam+hin" \
    INDIC_OCR_SIZE="best"

# --- Install system packages required for Tesseract and common tooling
# Keep package list minimal and grouped to reduce image layers and size.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      unzip \
      git \
      build-essential \
      pkg-config \
      libleptonica-dev \
      libtesseract-dev \
      tesseract-ocr \
      tesseract-ocr-eng \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# If you need additional language packs preinstalled via apt, add them above (e.g. tesseract-ocr-hin).
# We intentionally avoid installing many language packs in apt; the bootstrap script can pull traineddata per-run.

# --- Create application directory and non-root user
WORKDIR /app
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home --shell /bin/bash app \
 && chown app:app /app

# --- Install Python dependencies
# Copy requirements first to leverage Docker cache when requirements unchanged.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip --no-cache-dir install -r /app/requirements.txt

# --- Copy application code
# Copy ownership to the unprivileged user
COPY . /app
RUN chown -R app:app /app

# --- Switch to non-root user
USER app
ENV PATH="/home/app/.local/bin:${PATH}"

# --- Entrypoint: run main.py by default.
# Use CMD (overridable by docker run).
CMD ["python", "main.py"]
