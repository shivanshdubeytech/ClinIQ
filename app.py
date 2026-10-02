"""ClinIQ - Intelligent Medical RAG System Flask Application.

Serves the responsive single-page Web application (templates/index.html) built with
Tailwind CSS and Three.js 3D background animations. Exposes REST API endpoints for
query processing, user session persistence, message history, and audio streaming.
"""

import os
import re
import sys
import tempfile
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-detect and include project venv site-packages if running under global/system python
_venv_site = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import site
_candidates = [
    site.getusersitepackages(),
    os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages"),
    os.path.expanduser(r"~\AppData\Roaming\Python\Python39\site-packages"),
]
for _p in _candidates:
    if _p and _p not in sys.path and Path(_p).exists():
        sys.path.append(str(_p))

# Ensure stdout handles UTF-8 unicode printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import config
from config import setup_logger
from src.memory import create_session, get_session_messages, get_user_sessions
from src.pipeline import handle_query

logger = setup_logger("flask_app")

_AUDIO_FILENAME_RE = re.compile(r"^audio_[A-Za-z0-9_\-]{1,80}\.mp3$")

# ------------------------------------------------------------------
# RATE LIMITING IS NOT ACCESS CONTROL.
# These limits slow abuse and protect provider quota. They do not
# authenticate anyone and do not substitute for per-record ownership.
# See DEPLOYMENT_BLOCKERS.md.
# ------------------------------------------------------------------

class SlidingWindowLimiter:
    """Per-key sliding-window rate limiter. Process-local.

    LIMITATIONS (document these; do not hide them):
    - State is per-process. With N gunicorn workers, the effective limit is
      N x the configured rate. §10 configures a single worker for beta.
    - Memory grows with the number of distinct keys seen in the window.
      The deque for each key is bounded, and idle keys are pruned on access.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max = max_requests
        self.window = window_seconds
        self._buckets: dict = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max:
                return False
            bucket.append(now)
            # Opportunistic pruning of empty buckets to bound memory.
            if len(self._buckets) > 10_000:
                for k in [k for k, v in self._buckets.items() if not v]:
                    del self._buckets[k]
            return True


# Limits are deliberately conservative for a small beta cohort.
# Calibrate upward only after observing real usage. See §13.6.
_QUERY_LIMITER = SlidingWindowLimiter(max_requests=20, window_seconds=60)
_UPLOAD_LIMITER = SlidingWindowLimiter(max_requests=5,  window_seconds=300)
_READ_LIMITER = SlidingWindowLimiter(max_requests=30,  window_seconds=60)


def _rate_limit_key() -> str:
    """Rate-limit key: client IP. See prior spec §8 for the read-endpoint
    limitation note — until authentication exists, IP is the best available
    key and is shared behind NAT."""
    return request.remote_addr or "unknown"


app = Flask(__name__, template_folder="templates")

# Reject oversized request bodies at the WSGI layer before they are buffered.
# This matches the frontend's 20 MB client-side check and the backend check
# in src/report_analyzer/extractor.py (prior spec §10d). All three must agree.
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB

# Bound multipart parsing memory; overflow spills to a temp file.
app.config["MAX_FORM_MEMORY_SIZE"] = 2 * 1024 * 1024  # 2 MB

# Trust exactly one proxy hop. This makes request.remote_addr reflect the
# client IP from X-Forwarded-For. If you later add a CDN in front of nginx,
# increase x_for to the number of trusted hops (2 for one CDN + one nginx).
# NEVER set this higher than the actual number of proxies you control —
# an attacker can then forge X-Forwarded-For and bypass rate limits.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1
)


@app.errorhandler(RequestEntityTooLarge)
def handle_oversized_request(err):
    """Returns JSON response when request entity exceeds MAX_CONTENT_LENGTH."""
    return jsonify({
        "error": "Uploaded file exceeds the 20 MB limit. "
                 "Please upload a smaller report or split it into pages."
    }), 413


@app.after_request
def add_security_headers(response):
    """Baseline security headers."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "img-src 'self' data: blob: https://images.unsplash.com https://*.unsplash.com https:; "
        "font-src 'self' data: https:; "
        "connect-src 'self' https: http: ws: wss:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'"
    )
    return response


# CORS wildcarded intentionally: this is a same-origin application and
# the API is not a public cross-origin service. If a cross-origin
# deployment is ever intended, replace "*" with an explicit allowlist
# and re-review CSRF implications.
@app.after_request
def add_cors_headers(response):
    """Appends CORS headers to all responses for smooth frontend-backend connection."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
    return response


@app.before_request
def handle_options():
    """Handles CORS preflight OPTIONS requests across all routes."""
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
        return response


@app.route("/")
@app.route("/index.html")
def index():
    """Renders the main single-page application."""
    return render_template("index.html", demo=request.args.get("demo"))


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint to verify backend connectivity."""
    return jsonify(
        {
            "status": "healthy",
            "service": "ClinIQ Medical RAG API",
            "model": config.LLM_MODEL_NAME,
        }
    )


@app.route("/api/query", methods=["POST"])
def api_query():
    """Processes user question through the unified ClinIQ RAG pipeline."""
    if not _QUERY_LIMITER.allow(_rate_limit_key()):
        return jsonify({
            "error": "Too many questions in a short period. "
                     "Please wait a moment and try again."
        }), 429

    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id", "default_user")
        session_id = data.get("session_id", "default_session")
        question = (data.get("question") or data.get("query") or "").strip()

        if not question:
            return jsonify({"error": "Question is required."}), 400

        if len(question) > 5000:
            return jsonify({"error": "Question too long"}), 400

        result = handle_query(user_id=user_id, question=question, session_id=session_id)

        # Build public audio URL if MP3 file was generated
        audio_url = None
        if result.get("audio_path") and os.path.exists(result["audio_path"]):
            filename = os.path.basename(result["audio_path"])
            audio_url = f"/api/audio/{filename}"

        return jsonify(
            {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "passed": result["passed"],
                "checks_ran": result.get("checks_ran", False),
                "audio_url": audio_url,
                "sources": result["sources"],
                "is_concerning": result.get("is_concerning", False),
                "disclaimer": result.get("disclaimer", config.STANDARD_MEDICAL_DISCLAIMER),
            }
        )

    except Exception as err:
        logger.error(f"Error processing query endpoint: {type(err).__name__}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


@app.route("/api/sessions/<user_id>", methods=["GET"])
def api_get_sessions(user_id):
    """Retrieves all registered sessions for a user."""
    if not _READ_LIMITER.allow(_rate_limit_key()):
        return jsonify({"error": "Too many requests. Please slow down."}), 429

    try:
        sessions = get_user_sessions(user_id)
        return jsonify(sessions)
    except Exception as err:
        logger.error(f"Error fetching sessions for user: {type(err).__name__}")
        return jsonify([]), 500


@app.route("/api/session/<session_id>/messages", methods=["GET"])
def api_get_session_messages(session_id):
    """Retrieves past messages for a specific session_id."""
    if not _READ_LIMITER.allow(_rate_limit_key()):
        return jsonify({"error": "Too many requests. Please slow down."}), 429

    try:
        messages = get_session_messages(session_id)
        # Convert internal audio paths to public URLs if present
        for msg in messages:
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                filename = os.path.basename(msg["audio_path"])
                msg["audio_url"] = f"/api/audio/{filename}"
        return jsonify(messages)
    except Exception as err:
        logger.error(f"Error fetching messages for session: {type(err).__name__}")
        return jsonify([]), 500


@app.route("/api/audio/<filename>")
def api_get_audio(filename):
    """Serves synthesized audio MP3 files from static/audio directory."""
    # Reject anything that isn't a plain generated audio filename
    if not _AUDIO_FILENAME_RE.fullmatch(filename):
        return jsonify({"error": "Audio file not found."}), 404
    target = config.AUDIO_DIR / filename
    if not target.is_file():
        return jsonify({"error": "Audio file not found."}), 404
    return send_from_directory(str(config.AUDIO_DIR), filename, mimetype="audio/mp3")


@app.route("/api/analyze-report", methods=["POST"])
def api_analyze_report():
    """Upload and analyze a medical report or prescription document (PDF or image)."""
    from src.report_analyzer import analyze_medical_document

    if not _UPLOAD_LIMITER.allow(_rate_limit_key()):
        return jsonify({
            "error": "Too many uploads in a short period. "
                     "Please wait before analysing another document."
        }), 429

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please attach a medical report or prescription."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Selected file has no filename."}), 400

    # Preserve the extension only; never trust or reuse the client-supplied
    # basename for the on-disk path. secure_filename() is applied and then
    # discarded in favour of a per-request UUID.
    original_suffix = Path(secure_filename(uploaded_file.filename) or "").suffix.lower()
    if original_suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp",
                               ".tiff", ".tif", ".txt", ".csv"}:
        return jsonify({
            "error": "Unsupported file format. Allowed: PDF, PNG, JPG, JPEG, "
                     "WEBP, BMP, TIFF, TXT."
        }), 400

    temp_dir = Path(tempfile.gettempdir()) / "cliniq_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(temp_dir, 0o700)
    except OSError:
        pass  # Best-effort; on Windows this is a no-op.

    # Per-request unique name; no collision possible.
    temp_path = temp_dir / f"{uuid.uuid4().hex}{original_suffix}"

    try:
        uploaded_file.save(str(temp_path))
        # Re-verify the actual size on disk. MAX_CONTENT_LENGTH (§2a) bounds the
        # request body, but this is defence in depth and covers the case where
        # a proxy rewrites Content-Length.
        actual_size = temp_path.stat().st_size
        if actual_size == 0:
            return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400
        if actual_size > 20 * 1024 * 1024:
            return jsonify({"error": "Uploaded file exceeds the 20 MB limit."}), 413

        analysis_result = analyze_medical_document(temp_path, auto_delete_temp=True)
        return jsonify(analysis_result.to_dict())
    except Exception as err:
        logger.error(f"Error in /api/analyze-report: {type(err).__name__}")
        return jsonify({"error": "Failed to analyze document."}), 500
    finally:
        # analyze_medical_document deletes on success; this covers the error
        # paths where it may not have been reached.
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to remove temp upload; will be cleaned by tmp reaper.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))

    # Startup audio directory pruning (§6)
    try:
        from src.tts import prune_audio_dir
        prune_audio_dir(config.AUDIO_DIR)
    except Exception as e:
        logger.warning(f"Initial audio pruning skipped: {type(e).__name__}")

    # Startup Model Existence Verification (§7 Pre-Beta Hardening)
    try:
        from groq import Groq
        from config import get_groq_api_key, LLM_MODEL_NAME
        api_key = get_groq_api_key()
        if api_key:
            client = Groq(api_key=api_key)
            models = [m.id for m in client.models.list().data]
            if LLM_MODEL_NAME not in models:
                logger.warning(
                    f"[STARTUP WARNING] LLM_MODEL_NAME '{LLM_MODEL_NAME}' not found in Groq available models. "
                    f"Available models sample: {models[:5]}..."
                )
            else:
                logger.info(f"[STARTUP OK] Verified model '{LLM_MODEL_NAME}' is active on Groq.")
    except Exception as e:
        logger.warning(f"[STARTUP NOTICE] Groq model verification skipped: {type(e).__name__}")

    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")

    if debug_mode and host not in ("127.0.0.1", "localhost"):
        raise RuntimeError(
            "Refusing to start: FLASK_DEBUG=1 with a non-loopback HOST exposes "
            "the Werkzeug debugger, which allows arbitrary code execution. "
            "Set HOST=127.0.0.1 or FLASK_DEBUG=0."
        )

    logger.info(f"Starting ClinIQ Flask Web Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug_mode)
