"""Unified end-to-end RAG pipeline module for ClinIQ medical system.

Orchestrates conversation memory retrieval, semantic context search, grounded answer generation,
safety & faithfulness evaluation, memory persistence, and text-to-speech audio synthesis into a
single unified entry point function `handle_query()` for client UI integration.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 unicode printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from config import setup_logger
from src.evaluator import evaluate_and_respond
from src.generator import generate_answer
from src.memory import get_recent_history, save_interaction
from src.retriever import retrieve
from src.tts import synthesize

logger = setup_logger("pipeline")


def handle_query(user_id: str, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Orchestrates the end-to-end ClinIQ RAG pipeline for a user query.

    Flow:
    1. Fetch recent conversation history from memory.
    2. Retrieve top context chunks from ChromaDB vector store.
    3. Generate grounded LLM answer using context and history.
    4. Evaluate answer faithfulness and relevance, computing confidence score.
    5. Save interaction (question, final answer, confidence) to memory DB.
    6. Synthesize audio MP3 if response passed confidence evaluation (skip on fallback).

    Args:
        user_id (str): Unique identifier for the user.
        question (str): User's input medical question.
        session_id (Optional[str]): Optional session thread identifier.

    Returns:
        Dict[str, Any]: A result dictionary with keys:
            - "answer" (str): Final validated answer or fallback message.
            - "confidence" (float): Composite confidence score (0.0 to 1.0).
            - "passed" (bool): True if confidence >= CONFIDENCE_THRESHOLD.
            - "audio_path" (Optional[str]): Absolute path to synthesized MP3 audio if passed, else None.
            - "sources" (List[str]): List of unique topic sources used in context retrieval.
    """
    if not user_id or not question or not question.strip():
        return {
            "answer": "Please provide a valid question.",
            "confidence": 0.0,
            "passed": False,
            "audio_path": None,
            "sources": [],
        }

    # Custom Identity & Creator attribution handler
    q_norm = question.strip().lower()
    creator_keywords = [
        "father",
        "daddy",
        "dad",
        "papa",
        "creator",
        "maker",
        "who created you",
        "who created u",
        "who made you",
        "who made u",
        "who is your father",
        "who is your daddy",
        "who's your father",
        "who's your daddy",
        "who is ur father",
        "who is ur daddy",
        "who is your father or daddy",
        "your developer",
        "who developed you",
        "who developed u",
        "who programmed you",
    ]
    if any(kw in q_norm for kw in creator_keywords):
        creator_answer = "Shivansh Dubey is my creator and father."
        save_interaction(
            user_id=user_id,
            session_id=session_id or f"sess_{user_id}_default",
            question=question,
            answer=creator_answer,
            confidence=1.0,
        )
        audio_filename = f"audio_{user_id}_{abs(hash(question)) % 10000}.mp3"
        audio_target = str(config.AUDIO_DIR / audio_filename)
        audio_path = synthesize(text=creator_answer, output_path=audio_target)
        return {
            "answer": creator_answer,
            "confidence": 1.0,
            "passed": True,
            "audio_path": audio_path,
            "sources": ["ClinIQ System Profile"],
        }

    try:
        # Step 1: Retrieve recent conversation history for user and session
        history: List[str] = get_recent_history(user_id=user_id, session_id=session_id, n=3)

        # Step 2: Retrieve relevant context chunks from ChromaDB
        chunks: List[Dict[str, Any]] = retrieve(query=question, k=3)
        retrieval_score: float = chunks[0]["similarity_score"] if chunks else 0.0

        # Extract unique topics/sources from retrieved context chunks
        sources: List[str] = []
        for c in chunks:
            topic = c.get("topic")
            source = c.get("source")
            source_label = f"{topic} ({source})" if source else topic
            if source_label and source_label not in sources:
                sources.append(source_label)

        # Step 3: Generate grounded answer via Groq LLM
        raw_answer: str = generate_answer(question=question, context_chunks=chunks, history=history)

        # Step 4: Evaluate answer faithfulness and relevance
        eval_result: Dict[str, Any] = evaluate_and_respond(
            question=question,
            answer=raw_answer,
            context_chunks=chunks,
            retrieval_score=retrieval_score,
        )

        final_answer: str = eval_result["final_answer"]
        confidence: float = eval_result["confidence"]
        passed: bool = eval_result["passed"]

        # Step 5: Save interaction to memory DB
        save_interaction(
            user_id=user_id,
            session_id=session_id or f"sess_{user_id}_default",
            question=question,
            answer=final_answer,
            confidence=confidence,
        )

        # Step 6: Text-to-Speech audio synthesis (only if passed evaluation to save time/resources)
        audio_path: Optional[str] = None
        if passed:
            audio_filename = f"audio_{user_id}_{abs(hash(question)) % 10000}.mp3"
            audio_target = str(config.AUDIO_DIR / audio_filename)
            audio_path = synthesize(text=final_answer, output_path=audio_target)

        return {
            "answer": final_answer,
            "confidence": confidence,
            "passed": passed,
            "audio_path": audio_path,
            "sources": sources,
        }

    except Exception as err:
        # Fail-safe catch-all to prevent system crashes
        print(f"[PIPELINE ERROR] Critical error during query execution: {err}")
        return {
            "answer": "An unexpected error occurred while processing your request. Please try again later.",
            "confidence": 0.0,
            "passed": False,
            "audio_path": None,
            "sources": [],
            "error": str(err),
        }


if __name__ == "__main__":
    test_user = "pipeline_demo_user"
    test_questions = [
        ("Case 1: Answerable Question", "What are the early warning signs of type 2 diabetes?"),
        ("Case 2: Edge Case Question", "How much sleep do I need every night?"),
        ("Case 3: Irrelevant Question", "What is the capital of France?"),
    ]

    print("=" * 80)
    print("[PIPELINE TEST] Running Unified ClinIQ Pipeline Test Suite")
    print("=" * 80)

    for label, q in test_questions:
        print(f"\n================================================================================")
        print(f"[{label}]")
        print(f"Question: \"{q}\"")
        print(f"================================================================================")

        result = handle_query(user_id=test_user, question=q)

        print("\nPipeline Result Dict:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("--------------------------------------------------------------------------------")
