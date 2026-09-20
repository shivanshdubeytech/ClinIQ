"""Per-user conversation memory and persistence module for ClinIQ medical RAG system.

Manages SQLite database storage for user Q&A interactions and multi-session chat threads,
allowing persistent page refreshes, sidebar session navigation, and conversation history retrieval.
"""

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 unicode printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import SQLITE_DB_PATH, setup_logger

logger = setup_logger("memory")


def _get_db_path() -> str:
    """Ensures parent directory exists and returns the normalized SQLite database file path."""
    db_path = Path(SQLITE_DB_PATH)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def init_db() -> None:
    """Initializes the SQLite database, creates tables, and handles column migrations safely."""
    db_file = _get_db_path()

    create_sessions_sql = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_history_sql = """
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(create_sessions_sql)
        cursor.execute(create_history_sql)

        # Migration Check: Add session_id to history table if it was created in an earlier schema version
        cursor.execute("PRAGMA table_info(history);")
        columns = [col[1] for col in cursor.fetchall()]
        if "session_id" not in columns:
            logger.info("Migrating history table: adding 'session_id' column...")
            cursor.execute("ALTER TABLE history ADD COLUMN session_id TEXT;")

        conn.commit()


def create_session(user_id: str, session_id: str, title: Optional[str] = None) -> str:
    """Registers a new session thread for a user.

    Args:
        user_id (str): Unique user identifier.
        session_id (str): Unique session identifier.
        title (Optional[str]): Human-readable title for the session. Defaults to "New Conversation".

    Returns:
        str: The registered session_id.
    """
    if not user_id or not session_id:
        raise ValueError("user_id and session_id are required.")

    init_db()
    db_file = _get_db_path()
    session_title = title.strip() if title and title.strip() else "New Conversation"
    current_time = datetime.now().isoformat()

    insert_sql = """
    INSERT OR IGNORE INTO sessions (session_id, user_id, title, created_at)
    VALUES (?, ?, ?, ?);
    """
    with sqlite3.connect(db_file) as conn:
        conn.cursor().execute(insert_sql, (session_id, user_id, session_title, current_time))
        conn.commit()

    return session_id


def get_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    """Retrieves all sessions registered for a user, ordered most recent first.

    Args:
        user_id (str): Unique user identifier.

    Returns:
        List[Dict[str, Any]]: List of session dictionaries containing:
            - "session_id" (str)
            - "user_id" (str)
            - "title" (str)
            - "created_at" (str)
    """
    if not user_id:
        return []

    init_db()
    db_file = _get_db_path()
    query_sql = """
    SELECT session_id, user_id, title, created_at
    FROM sessions
    WHERE user_id = ?
    ORDER BY created_at DESC;
    """
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, (user_id,))
        rows = cursor.fetchall()

    return [
        {
            "session_id": r[0],
            "user_id": r[1],
            "title": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def save_interaction(
    user_id: str, session_id: str, question: str, answer: str, confidence: float
) -> None:
    """Saves a Q&A interaction bound to a specific user and session_id.

    Args:
        user_id (str): Unique identifier for the user.
        session_id (str): Unique identifier for the session thread.
        question (str): The user's input question.
        answer (str): The system's generated/validated answer.
        confidence (float): The composite confidence score of the answer.
    """
    if not user_id or not question:
        return

    # Fallback default session_id if none provided
    eff_session_id = session_id if session_id and session_id.strip() else f"sess_{user_id}_default"

    # Auto-register session if it doesn't exist yet, using truncated question as title
    title_snippet = question[:40] + "..." if len(question) > 40 else question
    create_session(user_id=user_id, session_id=eff_session_id, title=title_snippet)

    db_file = _get_db_path()
    insert_sql = """
    INSERT INTO history (user_id, session_id, question, answer, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    current_time = datetime.now().isoformat()

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            insert_sql,
            (user_id, eff_session_id, question, answer, float(confidence), current_time),
        )
        conn.commit()


def get_recent_history(
    user_id: str, session_id: Optional[str] = None, n: int = 3
) -> List[str]:
    """Retrieves the last n question texts for a user/session, ordered most recent first.

    Args:
        user_id (str): Unique identifier for the user.
        session_id (Optional[str]): Optional session_id filter.
        n (int): Maximum number of recent questions to retrieve. Defaults to 3.

    Returns:
        List[str]: List of recent question strings, most recent first.
    """
    if not user_id or n <= 0:
        return []

    init_db()
    db_file = _get_db_path()

    if session_id and session_id.strip():
        query_sql = """
        SELECT question FROM history
        WHERE user_id = ? AND session_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """
        params: tuple = (user_id, session_id, n)
    else:
        query_sql = """
        SELECT question FROM history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """
        params = (user_id, n)

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

    return [row[0] for row in rows]


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all past Q&A interaction records for a specific session_id.

    Args:
        session_id (str): Unique session identifier.

    Returns:
        List[Dict[str, Any]]: Chronological list of message dictionaries for UI rendering.
    """
    if not session_id:
        return []

    init_db()
    db_file = _get_db_path()
    query_sql = """
    SELECT question, answer, confidence, timestamp
    FROM history
    WHERE session_id = ?
    ORDER BY id ASC;
    """
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, (session_id,))
        rows = cursor.fetchall()

    messages: List[Dict[str, Any]] = []
    for q, a, conf, ts in rows:
        messages.append({"role": "user", "content": q, "timestamp": ts})
        messages.append(
            {
                "role": "assistant",
                "content": a,
                "confidence": conf,
                "passed": (conf >= 0.60),
                "timestamp": ts,
            }
        )
    return messages


if __name__ == "__main__":
    test_user = "test_multi_user"
    test_sess_1 = "sess_111"
    test_sess_2 = "sess_222"

    print("=" * 80)
    print(f"[MEMORY TEST] Testing Multi-Session Persistent Memory in '{SQLITE_DB_PATH}'")
    print("=" * 80)

    init_db()

    # Save interaction for Session 1
    save_interaction(
        user_id=test_user,
        session_id=test_sess_1,
        question="What causes high blood pressure?",
        answer="Narrowed blood vessels and lifestyle factors.",
        confidence=0.88,
    )

    # Save interaction for Session 2
    save_interaction(
        user_id=test_user,
        session_id=test_sess_2,
        question="What are early signs of diabetes?",
        answer="Increased thirst and frequent urination.",
        confidence=0.94,
    )

    # Retrieve user sessions
    user_sessions = get_user_sessions(test_user)
    print(f"\nUser Sessions Count: {len(user_sessions)}")
    for s in user_sessions:
        print(f"  -> Session ID: {s['session_id']} | Title: '{s['title']}' | Created: {s['created_at']}")

    # Retrieve messages for Session 1
    sess_1_msgs = get_session_messages(test_sess_1)
    print(f"\nSession 1 Messages Count: {len(sess_1_msgs)}")
    for m in sess_1_msgs:
        print(f"  [{m['role'].upper()}]: {m['content']}")

    assert len(user_sessions) >= 2, "Failed to retrieve multi-session list."
    assert len(sess_1_msgs) == 2, "Failed to retrieve session 1 Q&A messages."
    print("\n✅ VERIFICATION SUCCESSFUL: Multi-session memory persistence working properly!")
