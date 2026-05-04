FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
# Override at runtime: mlflow | trainer | api
ENV SERVICE=api
ENV PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install -r pyproject.toml --no-sources

COPY src /app/src

# data/ must exist in the step4 directory before building the trainer image.
# deploy.ps1 copies ../data/Telco-Customer-Churn.csv here automatically.
COPY data /app/data

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000 8000

ENTRYPOINT ["/entrypoint.sh"]
