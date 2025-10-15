#!/usr/bin/env sh
set -eu

# If GOOGLE_CREDENTIALS_JSON env var is provided, write it to a file and point GOOGLE_APPLICATION_CREDENTIALS to it
if [ -n "${GOOGLE_CREDENTIALS_JSON:-}" ]; then
  echo "Writing Google credentials JSON to /app/gcloud.json"
  # Use printf to avoid adding an extra newline
  printf "%s" "$GOOGLE_CREDENTIALS_JSON" > /app/gcloud.json
  export GOOGLE_APPLICATION_CREDENTIALS=/app/gcloud.json
fi

# Basic diagnostics (disable or remove in production if noisy)
echo "Starting FastAPI app with: PORT=${PORT:-8000} DB URL scheme=$(printf '%s' "${DATABASE_URL:-none}" | cut -d: -f1)"

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --timeout-keep-alive 30
