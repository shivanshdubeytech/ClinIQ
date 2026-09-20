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

    def rotate_key(self, failed_key: Optional[str] = None) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("No Groq API keys available for rotation.")
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
        raise RuntimeError(
            "Required environment variable 'GROQ_API_KEY' or 'GROQ_API_KEYS' is missing or unconfigured. "
            "Please configure your key in .env"
        )
    return parts


# API Credentials & Key Rotation
GROQ_API_KEYS: List[str] = _parse_api_keys()
KEY_ROTATION_MANAGER: KeyRotationManager = KeyRotationManager(GROQ_API_KEYS)
GROQ_API_KEY: str = KEY_ROTATION_MANAGER.get_current_key()


def get_groq_api_key() -> str:
    """Returns the currently active Groq API key."""
    return KEY_ROTATION_MANAGER.get_current_key()


def rotate_groq_api_key(failed_key: Optional[str] = None) -> str:
    """Rotates to the next configured Groq API key upon failure or rate limit."""
    new_key = KEY_ROTATION_MANAGER.rotate_key(failed_key)
    global GROQ_API_KEY
    GROQ_API_KEY = new_key
    return new_key

# AI Model Configurations
EMBEDDING_MODEL_NAME: str = _get_env_var("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2", str)
LLM_MODEL_NAME: str = _get_env_var("LLM_MODEL_NAME", "qwen/qwen3.8-27b", str)

# RAG & Document Processing Hyperparameters
CHUNK_SIZE: int = _get_env_var("CHUNK_SIZE", 250, int)
CHUNK_OVERLAP: int = _get_env_var("CHUNK_OVERLAP", 30, int)
CONFIDENCE_THRESHOLD: float = _get_env_var("CONFIDENCE_THRESHOLD", 0.6, float)

# Storage & Database Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
AUDIO_DIR: Path = PROJECT_ROOT / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH: str = _get_env_var("CHROMA_DB_PATH", "./chroma_db", str)
SQLITE_DB_PATH: str = _get_env_var("SQLITE_DB_PATH", "./cliniq.db", str)

# System-Wide Centralized Policy Constants
STANDARD_FALLBACK_MESSAGE: str = (
    "I don't have enough reliable information to answer this question with sufficient confidence."
)
STANDARD_MEDICAL_DISCLAIMER: str = (
    "Disclaimer: ClinIQ provides general consumer health information for educational "
    "purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment."
)

# Centralized Logging Setup
LOG_LEVEL_STR: str = _get_env_var("LOG_LEVEL", "INFO", str).upper()
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR, logging.INFO)


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
