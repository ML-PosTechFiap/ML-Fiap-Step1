#!/bin/bash
set -e

if [ $# -gt 0 ]; then
  exec "$@"
fi

case "${SERVICE:-api}" in
  mlflow)
    exec mlflow server \
      --host 0.0.0.0 \
      --port "${PORT:-5000}" \
      --workers 1 \
      --allowed-hosts "*"
    ;;
  trainer)
    mkdir -p "${MODELS_PATH:-/models}"
    exec python /app/src/main.py
    ;;
  api)
    exec uvicorn api.main:app \
      --host 0.0.0.0 \
      --port "${PORT:-8000}" \
      --reload
    ;;
  *)
    echo "Unknown SERVICE='${SERVICE}'. Valid values: mlflow | trainer | api" >&2
    exit 1
    ;;
esac
