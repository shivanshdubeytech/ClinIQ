"""Text-to-Speech (TTS) module for ClinIQ medical RAG system.

Converts medical answer text into audio MP3 files using Google Text-to-Speech (gTTS).
Supports text truncation to optimize API latency and includes error handling for network calls.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 unicode printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from gtts import gTTS
from config import setup_logger

logger = setup_logger("tts")


def synthesize(text: Optional[str], output_path: str = "output_audio.mp3", lang: str = "en") -> Optional[str]:
    """Converts a text string into an MP3 audio file using gTTS.

    Args:
        text (Optional[str]): The text string to convert to speech.
        output_path (str): Filepath where the MP3 audio file will be saved. Defaults to "output_audio.mp3".
        lang (str): Language code for TTS synthesis. Defaults to "en".

    Returns:
        Optional[str]: Absolute file path of saved MP3 audio if successful, or None on failure/empty text.
    """
    if not text or not text.strip():
        print("[TTS WARNING] Provided text is empty or None. Skipping synthesis.")
        return None

    cleaned_text = text.strip()

    # Truncate extremely long text at ~500 characters.
    # REASONING: gTTS relies on an HTTP request to Google Translate TTS servers.
    # Text longer than 500 characters requires chunking, increasing network latency
    # and resulting in slow audio generation for interactive RAG applications.
    MAX_TTS_CHARS = 500
    if len(cleaned_text) > MAX_TTS_CHARS:
        print(f"[TTS INFO] Truncating answer from {len(cleaned_text)} to {MAX_TTS_CHARS} characters for speech synthesis.")
        cleaned_text = cleaned_text[:MAX_TTS_CHARS] + "..."

    out_file = Path(output_path)
    if out_file.parent and not out_file.parent.exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[TTS INFO] Synthesizing speech using gTTS (Language: {lang})...")
        tts = gTTS(text=cleaned_text, lang=lang, slow=False)
        tts.save(str(out_file))

        abs_path = str(out_file.resolve())
        print(f"[TTS SUCCESS] Audio synthesized successfully: '{abs_path}'")
        return abs_path

    except Exception as err:
        print(f"[TTS ERROR] Failed to synthesize speech via gTTS network API: {err}")
        print("[TTS HINT] Ensure your device has an active internet connection to reach gTTS services.")
        return None


if __name__ == "__main__":
    sample_medical_answer = (
        "Early warning signs of type 2 diabetes include increased thirst, frequent urination, "
        "unexplained fatigue, and blurred vision. If you experience persistent symptoms, please "
        "consult a healthcare provider for an evaluation."
    )
    test_output = "sample_medical_answer.mp3"

    print("=" * 70)
    print("[TTS TEST] Testing Text-to-Speech Audio Synthesis")
    print("=" * 70)

    result_path = synthesize(text=sample_medical_answer, output_path=test_output)

    if result_path and os.path.exists(result_path):
        size_bytes = os.path.getsize(result_path)
        size_kb = size_bytes / 1024.0
        print(f"\n✅ Verification Successful:")
        print(f"  -> File Path : {result_path}")
        print(f"  -> File Size : {size_kb:.2f} KB ({size_bytes} bytes)")
    else:
        print("\n❌ Verification Failed: Audio file was not generated.")
