# indexing_pipeline/Dockerfile
# Build-time args:
#   OCR_LANGUAGES (comma-separated, default "eng")
#   INDIC_OCR_SIZE (best|standard|fast, default "best")
ARG OCR_LANGUAGES="eng"
ARG INDIC_OCR_SIZE="best"

FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Install packages required for Tesseract OCR + curl/unzip
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      unzip \
      build-essential \
      pkg-config \
      libleptonica-dev \
      libtesseract-dev \
      tesseract-ocr \
      tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install awscli so bootstrap/CI can use it at runtime if needed
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir awscli

# Copy python requirements and install early for caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Determine a safe tessdata dir (do NOT call tesseract --print-tessdata-dir during build).
# Pick the first existing path or default to /usr/share/tessdata
RUN set -eux; \
    CANDIDATES="/usr/share/tessdata /usr/share/tesseract-ocr/4.00/tessdata /usr/share/tesseract-ocr/tessdata /usr/local/share/tessdata"; \
    TESSDIR=""; \
    for d in $CANDIDATES; do \
      if [ -d \"$d\" ]; then TESSDIR=\"$d\"; break; fi; \
    done; \
    if [ -z \"$TESSDIR\" ]; then TESSDIR=\"/usr/share/tessdata\"; mkdir -p \"$TESSDIR\"; fi; \
    echo \"Using tessdata dir: $TESSDIR\" > /tmp/tessdir; \
    chmod 0755 /tmp/tessdir

# Prepare primary tessdata base based on INDIC_OCR_SIZE
ARG INDIC_OCR_SIZE
RUN set -eux; \
    case \"$INDIC_OCR_SIZE\" in \
      best) PRIMARY_BASE=\"https://github.com/tesseract-ocr/tessdata_best/raw/main\" ;; \
      standard) PRIMARY_BASE=\"https://github.com/tesseract-ocr/tessdata/raw/main\" ;; \
      fast) PRIMARY_BASE=\"https://github.com/tesseract-ocr/tessdata_fast/raw/main\" ;; \
      *) PRIMARY_BASE=\"https://github.com/tesseract-ocr/tessdata_best/raw/main\" ;; \
    esac; \
    echo \"PRIMARY_BASE=$PRIMARY_BASE\" > /tmp/primary_tessbase

# Download traineddata for requested languages at build time (CI recommended).
# Format: OCR_LANGUAGES="eng,tam,hin"
ARG OCR_LANGUAGES
RUN set -eux; \
    TESSDIR=$(cat /tmp/tessdir); \
    PRIMARY_BASE=$(cat /tmp/primary_tessbase | cut -d'=' -f2-); \
    INDIC_BASE="https://github.com/indic-ocr/tessdata/raw/main"; \
    IFS=','; \
    for lang in $OCR_LANGUAGES; do \
      lang="$(printf '%s' \"$lang\" | tr -d '[:space:]')"; \
      [ -z \"$lang\" ] && continue; \
      target=\"$TESSDIR/${lang}.traineddata\"; \
      if [ -f \"$target\" ]; then \
        echo \"SKIP: $lang already present\"; \
        continue; \
      fi; \
      # build candidate URLs in order
      urls=\"$INDIC_BASE/${lang}.traineddata $PRIMARY_BASE/${lang}.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/${lang}.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/${lang}.traineddata\"; \
      installed=0; \
      for u in $urls; do \
        echo \"Trying $u\"; \
        if curl -fsSL --retry 3 --retry-delay 2 \"$u\" -o \"/tmp/${lang}.traineddata\"; then \
          mv \"/tmp/${lang}.traineddata\" \"$target\"; chmod 0644 \"$target\"; echo \"Installed $target from $u\"; installed=1; break; \
        else \
          echo \"Failed to download $u\"; \
        fi; \
      done; \
      if [ \"$installed\" -ne 1 ]; then \
        echo \"WARNING: could not obtain traineddata for $lang; OCR for this language may fail\"; \
      fi; \
    done

# Create non-root runtime user and fix ownership
RUN useradd -m -u 1000 app || true && chown -R app:app /app

USER app
WORKDIR /app
ENV PATH="/home/app/.local/bin:$PATH"

# Default command executes main.py
CMD ["python", "main.py"]
