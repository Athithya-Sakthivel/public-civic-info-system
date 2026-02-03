# Build-time args:
#   OCR_LANGUAGES  - comma-separated list (default "eng,tam,hin")
#   INDIC_OCR_SIZE - best|standard|fast (default "best")
ARG OCR_LANGUAGES="eng,tam,hin"
ARG INDIC_OCR_SIZE="best"

FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

WORKDIR /app

# Install system packages required for Tesseract + curl/unzip + build tools
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

# Copy requirements and install Python deps (do not install awscli via pip here)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

# Expose OCR build args to environment for the helper script
ARG OCR_LANGUAGES
ARG INDIC_OCR_SIZE
ENV OCR_LANGUAGES="${OCR_LANGUAGES}"
ENV INDIC_OCR_SIZE="${INDIC_OCR_SIZE}"

# Write a robust script to detect tessdata dir and download traineddata
# Using a separate script avoids complex quoting in Dockerfile RUN lines.
RUN cat > /tmp/install_tessdata.sh <<'SH'
#!/bin/sh
set -euo pipefail

# Determine tessdata directory (first existing candidate or create default)
CANDIDATES="/usr/share/tessdata /usr/share/tesseract-ocr/4.00/tessdata /usr/share/tesseract-ocr/tessdata /usr/local/share/tessdata"
TESSDIR=""
for d in $CANDIDATES; do
  if [ -d "$d" ]; then
    TESSDIR="$d"
    break
  fi
done
if [ -z "$TESSDIR" ]; then
  TESSDIR="/usr/share/tessdata"
  mkdir -p "$TESSDIR"
fi
printf '%s\n' "$TESSDIR" > /tmp/tessdir
chmod 0644 /tmp/tessdir

# Choose primary tessdata base based on INDIC_OCR_SIZE
case "${INDIC_OCR_SIZE:-best}" in
  best) PRIMARY_BASE="https://github.com/tesseract-ocr/tessdata_best/raw/main" ;;
  standard) PRIMARY_BASE="https://github.com/tesseract-ocr/tessdata/raw/main" ;;
  fast) PRIMARY_BASE="https://github.com/tesseract-ocr/tessdata_fast/raw/main" ;;
  *) PRIMARY_BASE="https://github.com/tesseract-ocr/tessdata_best/raw/main" ;;
esac
printf '%s\n' "$PRIMARY_BASE" > /tmp/primary_tessbase
chmod 0644 /tmp/primary_tessbase

# If OCR_LANGUAGES is empty, do nothing
if [ -z "${OCR_LANGUAGES:-}" ]; then
  echo "No OCR_LANGUAGES specified; skipping tessdata downloads."
  exit 0
fi

INDIC_BASE="https://github.com/indic-ocr/tessdata/raw/main"

# Convert comma-separated to space-separated list
# Trim surrounding whitespace, then replace commas with spaces
langs=$(printf '%s' "${OCR_LANGUAGES}" | tr -d '[:space:]' | tr ',' ' ')
for lang in $langs; do
  [ -z "$lang" ] && continue
  target="$TESSDIR/${lang}.traineddata"
  if [ -f "$target" ]; then
    echo "SKIP: $lang already present at $target"
    continue
  fi

  # Candidate URLs in preferred order
  urls="$INDIC_BASE/${lang}.traineddata ${PRIMARY_BASE}/${lang}.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/${lang}.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/${lang}.traineddata"

  installed=0
  for u in $urls; do
    echo "Trying $u"
    if curl -fsSL --retry 3 --retry-delay 2 "$u" -o "/tmp/${lang}.traineddata"; then
      mv "/tmp/${lang}.traineddata" "$target"
      chmod 0644 "$target"
      echo "Installed $target from $u"
      installed=1
      break
    else
      echo "Failed to download $u"
    fi
  done

  if [ "$installed" -ne 1 ]; then
    echo "WARNING: could not obtain traineddata for $lang; OCR for this language may fail"
  fi
done
SH

# Make the script executable and run it
RUN chmod +x /tmp/install_tessdata.sh && /tmp/install_tessdata.sh

# Create non-root runtime user and chown app dir
RUN useradd -m -u 1000 app || true && chown -R app:app /app

USER app
WORKDIR /app
ENV PATH="/home/app/.local/bin:$PATH"

# Default command (override on docker run if necessary)
CMD ["python", "main.py"]
