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

import multiprocessing  # noqa: F401  (kept for the future, see note)

bind = "127.0.0.1:8501"
workers = 1
threads = 4
worker_class = "gthread"

# Timeout: derived from slowest observed report analysis and OCR extraction.
# Provisional baseline set to 180 seconds to allow OCR and large multi-page PDF
# extraction without premature worker termination (must be at least 2x slowest operation).
timeout = 180
graceful_timeout = 30
keepalive = 5

max_requests = 500         # recycle the worker to bound memory growth from
max_requests_jitter = 50   # the embedding model and OCR buffers

accesslog = "-"
errorlog = "-"
loglevel = "info"
# Do NOT enable --capture-output: it routes app stdout into gunicorn's
# error log, which duplicates the redaction-audited output in §4.

# Trust the X-Forwarded-* headers from nginx. ProxyFix in app.py (§2b)
# does the actual IP extraction; this flag tells gunicorn's access log
# to record the forwarded address.
forwarded_allow_ips = "127.0.0.1"
