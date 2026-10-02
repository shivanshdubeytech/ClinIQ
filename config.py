"""Centralized configuration module for the ClinIQ medical RAG system.

Loads environment variables using python-dotenv and defines system-wide
constants for AI models, vector database paths, chunking parameters,
and centralized logging/disclaimer policies.
Fails loudly at import time if required environment variables are missing.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, TypeVar

# Auto-detect and include project venv site-packages if running under global/system python
_PROJECT_ROOT = Path(__file__).resolve().parent
_venv_site = _PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

from dotenv import load_dotenv

# Load environment variables from a local .env file if available
load_dotenv()

# Memory optimization and telemetry suppression for production cloud environments
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

T = TypeVar("T")


def _get_required_env_var(var_name: str) -> str:
    """Retrieve a required environment variable or raise a RuntimeError.

    Args:
        var_name (str): The name of the environment variable to fetch.

    Returns:
        str: The trimmed string value of the environment variable.

    Raises:
        RuntimeError: If the environment variable is missing, empty, or unconfigured.
    """
    value = os.getenv(var_name)
    if not value or not value.strip() or value.strip() == "your_key_here":
        raise RuntimeError(
            f"Required environment variable '{var_name}' is missing or unconfigured. "
            "Please copy .env.example to .env and set a valid GROQ_API_KEY. "
            "You can obtain a free API key at https://console.groq.com"
        )
    return value.strip().strip(",").strip('"').strip("'").strip()


def _get_env_var(var_name: str, default: T, cast_type: Type[T] = str) -> T:
    """Retrieve an optional environment variable with type casting and fallback.

    Args:
        var_name (str): Name of the environment variable.
        default (T): Default value if variable is missing or invalid.
        cast_type (Type[T]): Data type function (int, float, str).

    Returns:
        T: Cast value of environment variable or default fallback.
    """
    value = os.getenv(var_name)
    if value is None or not value.strip():
        return default
    try:
        return cast_type(value.strip())
    except (ValueError, TypeError):
        return default


import threading

class KeyRotationManager:
    """Thread-safe API Key rotation and failover manager for Groq services."""

    def __init__(self, keys: List[str]):
        self._keys = [
            k.strip().strip(",").strip('"').strip("'").strip()
            for k in keys
            if k and k.strip()
        ]
        self._index = 0
        self._lock = threading.Lock()

    def get_current_key(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("No Groq API keys available for rotation.")
            return self._keys[self._index]

    def rotate_key(self, failed_key: Optional[str] = None) -> Optional[str]:
        with self._lock:
            if not self._keys:
                return None
            if len(self._keys) == 1:
                return self._keys[0]
            old_idx = self._index
            self._index = (self._index + 1) % len(self._keys)
            new_key = self._keys[self._index]
            masked_old = self._keys[old_idx][:8] + "..." + self._keys[old_idx][-4:]
            masked_new = new_key[:8] + "..." + new_key[-4:]
            print(f"[KEY ROTATION] Switched Groq API key from {masked_old} (index {old_idx}) to {masked_new} (index {self._index})")
            return new_key

    def get_all_keys(self) -> List[str]:
        with self._lock:
            return self._keys[self._index:] + self._keys[:self._index]


def _parse_api_keys() -> List[str]:
    """Parses single or comma-separated API keys from environment."""
    raw_keys = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
    parts = [p.strip().strip(",").strip('"').strip("'") for p in raw_keys.split(",") if p.strip()]
    if not parts or parts[0] == "your_key_here":
        return []
    return parts


# API Credentials & Key Rotation (lazy — resolved on first use).
_KEY_ROTATION_MANAGER: Optional[KeyRotationManager] = None

def _get_manager() -> KeyRotationManager:
    """Lazily construct the key manager. Resolves without crashing on empty environments."""
    global _KEY_ROTATION_MANAGER
    if _KEY_ROTATION_MANAGER is None:
        _KEY_ROTATION_MANAGER = KeyRotationManager(_parse_api_keys())
    return _KEY_ROTATION_MANAGER

def get_groq_api_key() -> Optional[str]:
    """Returns the currently active Groq API key (resolves lazily), or None if none configured."""
    try:
        return _get_manager().get_current_key()
    except Exception:
        return None

def rotate_groq_api_key(failed_key: Optional[str] = None) -> Optional[str]:
    """Rotates to the next configured Groq API key."""
    return _get_manager().rotate_key(failed_key)

def get_all_groq_api_keys() -> List[str]:
    """Returns keys in rotation order (currently active first)."""
    return _get_manager().get_all_keys()

# AI Model Configurations
EMBEDDING_MODEL_NAME: str = _get_env_var("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2", str)
LLM_MODEL_NAME: str = _get_env_var("LLM_MODEL_NAME", "llama-3.1-8b-instant", str)

# RAG & Document Processing Hyperparameters
CHUNK_SIZE: int = _get_env_var("CHUNK_SIZE", 250, int)
CHUNK_OVERLAP: int = _get_env_var("CHUNK_OVERLAP", 30, int)
CONFIDENCE_THRESHOLD: float = _get_env_var("CONFIDENCE_THRESHOLD", 0.6, float)

# Storage & Database Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
AUDIO_DIR: Path = Path(_get_env_var("AUDIO_DIR", str(PROJECT_ROOT / "static" / "audio"), str))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH: str = _get_env_var("CHROMA_DB_PATH", "./chroma_db", str)
SQLITE_DB_PATH: str = _get_env_var("SQLITE_DB_PATH", "./cliniq.db", str)

# System-Wide Centralized Policy Constants
STANDARD_FALLBACK_MESSAGE: str = (
    "I don't have enough reliable information to answer this question with sufficient confidence. "
    "Please seek professional medical help from a qualified healthcare provider."
)
STANDARD_MEDICAL_DISCLAIMER: str = (
    "Disclaimer: ClinIQ provides general consumer health information for educational "
    "purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment."
)
URGENT_MEDICAL_DISCLAIMER: str = (
    "Seek immediate medical help: If you or someone nearby is experiencing acute, severe symptoms, chest pain, "
    "breathing distress, signs of stroke, poisoning, or a medical emergency, please call emergency services "
    "(911 / 112) or go to the nearest emergency department immediately."
)

# Centralized Logging Setup
LOG_LEVEL_STR: str = _get_env_var("LOG_LEVEL", "INFO", str).upper()
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR, logging.INFO)


import hashlib
import secrets

def redact_for_log(text: Optional[str], max_len: int = 0) -> str:
    """Returns a non-reversible, non-reconstructable token for logging.

    Never log raw patient text. Use this for any string that originated
    from a user: questions, answers, filenames, extracted document text,
    drug names from prescriptions, and reference ranges from reports.

    Returns a short fingerprint so distinct inputs remain distinguishable
    in logs for debugging, without retaining the content.
    """
    if not text:
        return "<empty>"
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    if max_len <= 0:
        return f"<redacted:{digest}>"
    # Optionally keep a bounded prefix for operational debugging. Only use
    # this for NON-CLINICAL text such as error messages.
    return f"{text[:max_len]!r}...<redacted:{digest}>"


def new_audio_token() -> str:
    """Returns an unguessable token for an audio filename.

    base64url alphabet is [A-Za-z0-9_-], which matches the anchored
    regex in app.py (prior spec §1). 16 bytes = 128 bits of entropy.
    """
    return secrets.token_urlsafe(16)


# ------------------------------------------------------------------
# LOG REDACTION POLICY
# No string that originated from a user — question, answer, uploaded
# document text, extracted clinical values, or filename — may be logged
# at INFO or above. Use redact_for_log(). This is enforced by review,
# not by the type system; when adding a logger call, classify the
# interpolated values before committing.
# ------------------------------------------------------------------
def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a standardized logger for ClinIQ modules.

    Args:
        name (str): The module or component name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
