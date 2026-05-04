# Monitoring Plan — Telco Churn API

## Overview

The monitoring stack is: **Prometheus** (metrics collection) + **Grafana** (dashboards + alert rules).

Prometheus scrapes the `/metrics` endpoint exposed by `prometheus-fastapi-instrumentator` every 15 s.  
Grafana is provisioned automatically from the `monitoring/` folder on `docker compose up`.

---

## Metrics Collected

### Infrastructure metrics (auto-instrumented)

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total requests by method, handler, status code |
| `http_request_duration_seconds` | Histogram | Request latency (buckets: 5 ms to 10 s) |
| `http_request_size_bytes` | Histogram | Request payload size |
| `http_response_size_bytes` | Histogram | Response payload size |
| `up` | Gauge | 1 = API reachable by Prometheus, 0 = down |

### Derived metrics (computed in Grafana/PromQL)

| Metric | PromQL |
|--------|--------|
| Request rate (req/s) | `rate(http_requests_total[1m])` |
| p95 latency | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))` |
| Error rate % | `rate(http_requests_total{status_code=~"5.."}[1m]) / rate(http_requests_total[1m]) * 100` |
| Prediction throughput | `rate(http_requests_total{handler="/predict",status_code="200"}[1m])` |

---

## Grafana Dashboard

URL: `http://localhost:3000` (user: `admin`, password: `admin`)

Dashboard: **Churn API — Step 4** (auto-provisioned from `monitoring/grafana/dashboards/churn-api.json`)

### Panels

| Panel | Description |
|-------|-------------|
| Request Rate | Time-series of req/s split by handler and status |
| P95 Latency | Time-series of 95th percentile response time |
| Error Rate % | 5xx error percentage over time |
| Predictions (total) | Stat panel — cumulative successful `/predict` calls |
| API Uptime | Green/red indicator — `up` gauge |
| Request Duration Heatmap | Latency distribution heatmap for `/predict` |

---

## Alert Rules

> In a production environment these would be configured as Grafana Alerting rules or `rules.yml` for the Prometheus Alertmanager.

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| **API Down** | `up{job="churn-api"} == 0` for 1 min | Critical | Page on-call; restart container |
| **High Latency** | p95 latency > 500 ms for 5 min | Warning | Check resource limits; scale up |
| **High Error Rate** | 5xx > 1% for 5 min | Warning | Check logs; rollback model if recent deploy |
| **Model Not Loaded** | `/health` returns `model_status=not_loaded` | Critical | Trigger trainer; verify pkl file exists |
| **No Predictions** | Prediction rate = 0 for 15 min during business hours | Warning | Verify CRM integration; check API logs |

---

## Playbook

### P1: API Down (`up == 0`)

1. `docker compose ps` — identify which container is not running.
2. `docker compose logs api` — check for startup error.
3. If model file missing: `ls models/` — if empty, trigger training: `docker compose run trainer`.
4. Restart: `docker compose restart api`.
5. Verify: `curl http://localhost:8000/health`.

### P2: High Latency (p95 > 500 ms)

1. Check CPU/memory: `docker stats step4_api`.
2. Look for slow predictions in logs: `docker compose logs api | grep "processed in"`.
3. If all predictions slow: model might be too large. Check model size: `ls -lh models/tuned_model.pkl`.
4. Scale horizontally (if behind load balancer) or increase container resources.

### P3: High Error Rate (5xx > 1%)

1. `docker compose logs api --tail=100` — identify exception type.
2. If `ValueError: Model is not loaded` → model hot-reload failed → restart api.
3. If `500` on `/predict` with schema errors → validate feature engineering pipeline.
4. Roll back to previous model version: `cp models/tuned_model_backup.pkl models/tuned_model.pkl`.

### P4: Model Not Loaded

1. Verify file exists: `ls models/tuned_model.pkl`.
2. If missing, run trainer: `docker compose run --rm trainer`.
3. File watcher will pick up the new file within 5 s — no restart needed.
4. Confirm: `curl http://localhost:8000/health | jq .model_status`.

### P5: Concept Drift (manual detection)

Drift is not automatically detected in this setup. Manual checks:
1. Weekly: compare current week's churn rate predicted vs. actuals (if CRM feedback available).
2. If AUC-ROC (computed offline) drops > 3 pp from baseline → trigger full retraining.
3. Retrain: `docker compose run --rm trainer` (or CI pipeline).
4. New model auto-loaded by file watcher.

---

## Log Access

```bash
# API logs
docker compose logs -f api

# Training logs
docker compose logs trainer

# All services
docker compose logs -f

# Log file (inside container)
docker compose exec api cat /app/logs/pipeline.log
```

Logs are structured: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`  
No `print()` statements — all output goes through Python `logging`.

---

## Retraining Trigger Criteria

| Trigger | Threshold |
|---------|-----------|
| Scheduled | Every 90 days |
| Performance drift | AUC-ROC drops > 3 pp vs. baseline |
| Data volume change | New dataset with > 20% more records |
| Feature distribution shift | KS-test p-value < 0.05 on key features |
| Business rule change | CAC or retention cost updated |
