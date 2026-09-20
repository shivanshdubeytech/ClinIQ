"""Unified end-to-end RAG pipeline module for ClinIQ medical system.

Orchestrates conversation memory retrieval, semantic context search, grounded answer generation,
safety & faithfulness evaluation, memory persistence, and text-to-speech audio synthesis into a
single unified entry point function `handle_query()` for client UI integration.
"""

import json
import re
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

# Clinical Red-Flag Patterns for Acute, Emergency, or Concerning Inquiries
CONCERNING_PATTERNS = [
    # Cardiac / Acute Chest Distress
    r"\bchest\s+(pain|tightness|pressure|discomfort|heaviness)\b",
    r"\bheart\s+attack\b",
    r"\bmyocardial\s+infarction\b",
    r"\bcardiac\s+arrest\b",
    r"\bpain\s+radiating\s+to\s+(left\s+arm|jaw|neck|shoulder|back)\b",
    # Respiratory Distress & Airway Emergencies
    r"\b(shortness\s+of\s+breath|difficulty\s+breathing|cannot\s+breathe|can't\s+breathe|trouble\s+breathing)\b",
    r"\b(asthma\s+attack|acute\s+asthma|severe\s+asthma|gasping\s+for\s+(air|breath))\b",
    r"\b(choking|asphyxiat(ion|ing)|suffocat(ion|ing)|blue\s+(lips|face|fingers))\b",
    # Stroke / Neurological Crises
    r"\b(stroke|mini[- ]stroke|tia|transient\s+ischemic)\b",
    r"\b(face\s+droop(ing)?|facial\s+droop(ing)?)\b",
    r"\b(slurred\s+speech|speech\s+slurred|cannot\s+speak|unable\s+to\s+speak)\b",
    r"\b(sudden\s+(numbness|paralysis|weakness)\s+(on\s+one\s+side|in\s+arm|in\s+face|in\s+leg))\b",
    r"\b(thunderclap\s+headache|worst\s+headache\s+of\s+my\s+life)\b",
    r"\b(unconscious|loss\s+of\s+consciousness|passed\s+out|passing\s+out|fainted|fainting|unresponsive)\b",
    r"\b(seizure|seizures|convuls(ing|ion)|epilep(tic|sy)\s+fit)\b",
    # Severe Bleeding & Trauma
    r"\b(severe|uncontrolled|heavy|massive|profuse)\s+bleeding\b",
    r"\b(coughing|vomiting)\s+(up\s+)?blood\b",
    r"\b(blood\s+in\s+(vomit|stool|urine)|black\s+tarry\s+stool)\b",
    r"\b(hemorrhag(e|ing)|stab\s+wound|gunshot)\b",
    r"\b(severe\s+head\s+injury|skull\s+fracture|severe\s+concussion)\b",
    # Anaphylaxis & Severe Allergies
    r"\b(anaphylax(is|ic)|severe\s+allergic\s+reaction)\b",
    r"\b(throat\s+closing|throat\s+swelling|tongue\s+swelling|swollen\s+tongue)\b",
    # Poisoning & Overdose
    r"\b(poison(ing|ed)?|toxic\s+(ingestion|substance|fumes))\b",
    r"\b(overdose|drug\s+overdose|swallowed\s+(pills|battery|bleach|chemical))\b",
    # Mental Health Crisis & Self-Harm
    r"\b(suicid(e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die|self[- ]harm)\b",
    # Acute Medical Emergencies
    r"\b(high\s+fever\s+(with|and)\s+stiff\s+neck|meningitis)\b",
    r"\b(third\s+degree\s+burn|severe\s+burns)\b",
    r"\b(medical\s+emergency|life[- ]threatening\s+emergency)\b",
]


def check_concerning_query(question: str) -> bool:
    """Evaluates whether a user's question presents acute, red-flag, or emergency symptoms.

    Args:
        question (str): User's natural language input inquiry.

    Returns:
        bool: True if emergency or concerning symptom markers are identified.
    """
    if not question:
        return False
    q_norm = question.strip().lower()
    for pattern in CONCERNING_PATTERNS:
        if re.search(pattern, q_norm):
            return True
    return False


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
            "is_concerning": False,
            "disclaimer": config.STANDARD_MEDICAL_DISCLAIMER,
        }

    is_concerning: bool = check_concerning_query(question)
    if is_concerning:
        logger.warning(f"[CONCERNING QUERY DETECTED] Question: '{question}' flagged for urgent medical disclaimer.")

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
            "is_concerning": False,
            "disclaimer": config.STANDARD_MEDICAL_DISCLAIMER,
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
            "is_concerning": is_concerning,
            "disclaimer": config.URGENT_MEDICAL_DISCLAIMER if is_concerning else config.STANDARD_MEDICAL_DISCLAIMER,
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
            "is_concerning": False,
            "disclaimer": config.STANDARD_MEDICAL_DISCLAIMER,
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
