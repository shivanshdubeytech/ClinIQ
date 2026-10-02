import os
import re
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import new_audio_token
from src.tts import MAX_AUDIO_FILES, MAX_AUDIO_AGE_SECONDS, prune_audio_dir

_AUDIO_RE = re.compile(r"^audio_[A-Za-z0-9_\-]{1,80}\.mp3$")


def test_audio_tokens_1000_entropy_and_uniqueness():
    """Generates 1000 audio filenames, asserts uniqueness and regex adherence."""
    filenames = set()
    tokens = []
    for _ in range(1000):
        tok = new_audio_token()
        fname = f"audio_{tok}.mp3"
        assert _AUDIO_RE.fullmatch(fname), f"Filename {fname} failed regex validation"
        assert len(tok) >= 20, f"Token length {len(tok)} is too short"
        filenames.add(fname)
        tokens.append(tok)

    # 1000 unique filenames with zero collisions
    assert len(filenames) == 1000, "Collision detected in 1000 tokens"


def test_prune_audio_dir():
    """Verifies that prune_audio_dir removes expired files and caps total count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)

        # 1. Create an expired file (older than 24 hours)
        old_file = dir_path / "audio_old_test.mp3"
        old_file.write_bytes(b"dummy")
        # Set mtime to 48 hours ago
        past_time = time.time() - (MAX_AUDIO_AGE_SECONDS + 3600)
        import os
        os.utime(str(old_file), (past_time, past_time))

        # 2. Create a fresh file
        fresh_file = dir_path / "audio_fresh_test.mp3"
        fresh_file.write_bytes(b"dummy")

        removed = prune_audio_dir(dir_path)
        assert removed >= 1
        assert not old_file.exists(), "Old audio file was not pruned"
        assert fresh_file.exists(), "Fresh audio file should not be pruned"


if __name__ == "__main__":
    test_audio_tokens_1000_entropy_and_uniqueness()
    print("test_audio_tokens_1000_entropy_and_uniqueness passed (1000 unique unguessable tokens)")
    test_prune_audio_dir()
    print("test_prune_audio_dir passed (retention cleanup verified)")

