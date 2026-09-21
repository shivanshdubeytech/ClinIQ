"""ClinIQ - Intelligent Medical RAG System Flask Application.

Serves the responsive single-page Web application (templates/index.html) built with
Tailwind CSS and Three.js 3D background animations. Exposes REST API endpoints for
query processing, user session persistence, message history, and audio streaming.
"""

import os
import sys
from pathlib import Path

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

import importlib
import config
import src.retriever
import src.generator
import src.evaluator
import src.memory
import src.tts
import src.pipeline

importlib.reload(config)
importlib.reload(src.retriever)
importlib.reload(src.generator)
importlib.reload(src.evaluator)
importlib.reload(src.memory)
importlib.reload(src.tts)
importlib.reload(src.pipeline)

from src.memory import create_session, get_session_messages, get_user_sessions
from src.pipeline import handle_query
from config import setup_logger

logger = setup_logger("flask_app")

app = Flask(__name__, template_folder="templates")


@app.before_request
def handle_options():
    """Handles CORS preflight OPTIONS requests across all routes."""
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
        return response


@app.after_request
def add_cors_headers(response):
    """Appends CORS headers to all responses for smooth frontend-backend connection."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
    return response


@app.route("/")
@app.route("/index.html")
def index():
    """Renders the main single-page application."""
    return render_template("index.html")


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
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id", "default_user")
        session_id = data.get("session_id", "default_session")
        question = data.get("question") or data.get("query", "")

        if not question or not question.strip():
            return jsonify({"error": "Question is required."}), 400

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
                "audio_url": audio_url,
                "sources": result["sources"],
                "is_concerning": result.get("is_concerning", False),
                "disclaimer": result.get("disclaimer", config.STANDARD_MEDICAL_DISCLAIMER),
            }
        )

    except Exception as err:
        logger.error(f"Error processing query endpoint: {err}")
        return jsonify({"error": str(err)}), 500


@app.route("/api/sessions/<user_id>", methods=["GET"])
def api_get_sessions(user_id):
    """Retrieves all registered sessions for a user."""
    try:
        sessions = get_user_sessions(user_id)
        return jsonify(sessions)
    except Exception as err:
        logger.error(f"Error fetching sessions for user '{user_id}': {err}")
        return jsonify([]), 500


@app.route("/api/session/<session_id>/messages", methods=["GET"])
def api_get_session_messages(session_id):
    """Retrieves past messages for a specific session_id."""
    try:
        messages = get_session_messages(session_id)
        # Convert internal audio paths to public URLs if present
        for msg in messages:
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                filename = os.path.basename(msg["audio_path"])
                msg["audio_url"] = f"/api/audio/{filename}"
        return jsonify(messages)
    except Exception as err:
        logger.error(f"Error fetching messages for session '{session_id}': {err}")
        return jsonify([]), 500


@app.route("/api/audio/<filename>", methods=["GET"])
def api_get_audio(filename):
    """Serves synthesized audio MP3 files from static/audio directory."""
    try:
        if (config.AUDIO_DIR / filename).exists():
            return send_from_directory(str(config.AUDIO_DIR), filename, mimetype="audio/mp3")
        return send_from_directory(str(PROJECT_ROOT), filename, mimetype="audio/mp3")
    except Exception as err:
        logger.error(f"Error serving audio file '{filename}': {err}")
        return jsonify({"error": "Audio file not found."}), 404


@app.route("/api/analyze-report", methods=["POST"])
def api_analyze_report():
    """Upload and analyze a medical report or prescription document (PDF or image)."""
    import tempfile
    from werkzeug.utils import secure_filename
    from src.report_analyzer import analyze_medical_document

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Please attach a medical report or prescription."}), 400

        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            return jsonify({"error": "Selected file has no filename."}), 400

        # Create temporary storage with immediate auto-cleanup after processing (Rule 8)
        temp_dir = Path(tempfile.gettempdir()) / "cliniq_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = secure_filename(uploaded_file.filename) or "report_upload.pdf"
        temp_path = temp_dir / safe_name
        uploaded_file.save(str(temp_path))

        # Run Stage A-G pipeline and safely delete temporary file (Rule 8)
        analysis_result = analyze_medical_document(temp_path, auto_delete_temp=True)
        return jsonify(analysis_result.to_dict())

    except Exception as err:
        logger.error(f"Error in /api/analyze-report: {err}")
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    logger.info(f"Starting ClinIQ Flask Web Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
