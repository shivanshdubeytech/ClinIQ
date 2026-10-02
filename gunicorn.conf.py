# Gunicorn configuration for ClinIQ.
#
# WORKERS = 1 IS DELIBERATE, NOT AN OVERSIGHT.
# Both the rate limiter (§3) and the SQLite connection handling (§7) hold
# state per process. More than one worker multiplies the effective rate
# limit and increases write contention. Single worker is correct for a
# small beta. See DEPLOYMENT_BLOCKERS.md for the multi-worker path.
#
# Do NOT add a --preload flag: the app loads SentenceTransformers lazily
# on first query, and preloading in the master process before fork risks
# holding model memory twice.

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DISABLE_GPU", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

port = os.environ.get("PORT", "8501")
host = os.environ.get("HOST", "0.0.0.0")
bind = f"{host}:{port}"
workers = 1
threads = 4
worker_class = "gthread"

# Timeout: derived from slowest observed report analysis and OCR extraction.
timeout = 180
graceful_timeout = 30
keepalive = 5

max_requests = 500         # recycle the worker to bound memory growth from
max_requests_jitter = 50   # the embedding model and OCR buffers

accesslog = "-"
errorlog = "-"
loglevel = "info"

# Trust the X-Forwarded-* headers from cloud proxies (Render, Nginx, etc.)
forwarded_allow_ips = "*"
