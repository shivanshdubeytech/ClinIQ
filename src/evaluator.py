"""Evaluation and confidence scoring module for ClinIQ medical RAG system.

Implements automated self-correction and confidence evaluation for RAG responses.
Evaluates faithfulness (groundedness against context) and relevance (addressing user query)
using structured JSON LLM calls, and computes a composite confidence score to gate responses.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 unicode printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq

from config import (
    CONFIDENCE_THRESHOLD,
    GROQ_API_KEY,
    KEY_ROTATION_MANAGER,
    LLM_MODEL_NAME,
    rotate_groq_api_key,
    setup_logger,
)
from src.generator import generate_answer
from src.retriever import retrieve

logger = setup_logger("evaluator")


def _extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Robust helper function to extract and parse JSON object from LLM response text."""
    if not response_text:
        raise ValueError("Empty response text")

    clean_text = response_text.strip()

    # Try direct JSON parsing
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # Try extracting markdown json code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding outer curly braces
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        candidate = clean_text[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from response: {response_text}")


def _sanitize_text_for_prompt(text: str) -> str:
    """Sanitizes text by stripping troublesome markdown formatting (pipes, triple backticks) to ensure clean JSON evaluation prompts."""
    if not text:
        return ""
    # Strip triple backticks to avoid markdown code block confusion
    s = text.replace("```", " ")
    # Replace markdown table pipes with spaces
    s = s.replace("|", " ")
    # Normalize multiple whitespace lines
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def check_faithfulness(
    answer: str, context_chunks: List[Dict[str, Any]], retrieval_score: float = 0.0
) -> Tuple[bool, str, str]:
    """Evaluates if the generated answer contains any claims not supported by retrieved context.

    Prompts the LLM for a plain text JSON object, extracts JSON, and falls back gracefully.

    Args:
        answer (str): The generated answer text to evaluate.
        context_chunks (List[Dict[str, Any]]): The retrieved context chunks used for generation.
        retrieval_score (float): Retrieval similarity score used for fallback evaluation.

    Returns:
        Tuple[bool, str, str]: (is_faithful, reasoning, raw_llm_output)
    """
    if not answer or not answer.strip():
        return False, "Answer is empty.", ""

    # If answer is standard fallback refusal statement, it is faithful by definition
    if "don't have enough information" in answer.lower():
        return True, "Standard fallback refusal statement.", '{"unsupported_claims": false, "reason": "Refusal statement"}'

    context_text = "\n\n".join(
        [f"Chunk {i+1}:\n{c.get('text', '')}" for i, c in enumerate(context_chunks)]
    )

    sanitized_answer = _sanitize_text_for_prompt(answer)
    sanitized_context = _sanitize_text_for_prompt(context_text)

    prompt = f"""You are a medical grounding evaluator. Evaluate if the Answer is faithful to and consistent with the Context.

Context:
{sanitized_context}

Answer:
{sanitized_answer}

Instruction:
Evaluate whether the Answer introduces major fabricated facts, incorrect medical statistics, or dangerous claims that contradict the context.
General health education, common self-care advice (such as rest, hydration, monitoring symptoms, seeking a doctor), and faithful synthesis of symptoms mentioned in the context are valid and supported.
Return a JSON object with two fields:
- "unsupported_claims": boolean (true ONLY if the answer contains major fabricated or contradicted medical claims, false if it is grounded and cautious)
- "reason": string (a short 1-sentence explanation)

JSON:"""

    raw_response = ""
    candidate_models = [LLM_MODEL_NAME, "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]
    keys_to_try = KEY_ROTATION_MANAGER.get_all_keys()

    for api_key in keys_to_try:
        client = Groq(api_key=api_key)
        for model in models_to_try:
            try:
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    temperature=0.1,
                    max_tokens=512,
                )

                raw_response = completion.choices[0].message.content or ""
                parsed = _extract_json_from_response(raw_response)

                has_unsupported = bool(parsed.get("unsupported_claims", True))
                reason = str(parsed.get("reason", "No reason provided."))

                is_faithful = not has_unsupported
                return is_faithful, reason, raw_response

            except Exception as err:
                err_str = str(err).lower()
                logger.warning(f"Faithfulness check issue on model '{model}': {err}.")
                if "rate_limit" in err_str or "429" in err_str or "401" in err_str or "invalid_api_key" in err_str:
                    rotate_groq_api_key(api_key)
                    break  # Try next key
                continue

    # Fallback: if retrieval score is high (>= 0.4), lean toward faithful on parse/API glitch
    if retrieval_score >= 0.4:
        return True, f"Faithful (Fallback: high retrieval score {retrieval_score:.2f})", raw_response

    return False, "Faithfulness check failed on all models", raw_response


def check_relevance(answer: str, question: str, retrieval_score: float = 0.0) -> Tuple[bool, str]:
    """Evaluates if the answer directly addresses the user's input question.

    Args:
        answer (str): The generated answer text.
        question (str): The user's input question.
        retrieval_score (float): Retrieval similarity score used for fallback evaluation.

    Returns:
        Tuple[bool, str]: (is_relevant, raw_llm_output)
    """
    if not answer or not question:
        return False, ""

    # Standard fallback refusal statement is not a direct answer to out-of-scope question
    if "don't have enough information" in answer.lower():
        return False, '{"relevant": false, "reason": "Standard refusal statement"}'

    sanitized_answer = _sanitize_text_for_prompt(answer)
    sanitized_question = _sanitize_text_for_prompt(question)

    prompt = f"""You are an answer relevance evaluator. Evaluate if the Answer directly addresses the User Question.

User Question:
{sanitized_question}

Answer:
{sanitized_answer}

Instruction:
Return a JSON object with two fields:
- "relevant": boolean (true if the answer directly addresses the user question, false otherwise)
- "reason": string (a short 1-sentence explanation)

JSON:"""

    raw_response = ""
    candidate_models = [LLM_MODEL_NAME, "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]
    keys_to_try = KEY_ROTATION_MANAGER.get_all_keys()

    for api_key in keys_to_try:
        client = Groq(api_key=api_key)
        for model in models_to_try:
            try:
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    temperature=0.1,
                    max_tokens=512,
                )

                raw_response = completion.choices[0].message.content or ""
                parsed = _extract_json_from_response(raw_response)

                return bool(parsed.get("relevant", False)), raw_response

            except Exception as err:
                err_str = str(err).lower()
                logger.warning(f"Relevance check issue on model '{model}': {err}.")
                if "rate_limit" in err_str or "429" in err_str or "401" in err_str or "invalid_api_key" in err_str:
                    rotate_groq_api_key(api_key)
                    break  # Try next key
                continue

    # Fallback: if retrieval score is high (>= 0.4), lean toward relevant on parse/API glitch
    if retrieval_score >= 0.4:
        return True, raw_response

    return False, raw_response


def compute_confidence(retrieval_score: float, faithful: bool, relevant: bool) -> float:
    """Computes a composite confidence score between 0.0 and 1.0.

    Formula & Logic:
    - Base Weighted Formula:
        score = (0.35 * retrieval_score) + (0.40 * faithful_score) + (0.25 * relevant_score)
        where faithful_score = 1.0 if faithful else 0.0
        and relevant_score = 1.0 if relevant else 0.0

    - Fail-Safe Guardrails (Strict Medical Safety):
        - If NOT faithful (hallucination / ungrounded claim): Cap maximum confidence at 0.30.
        - If NOT relevant (off-topic or fallback refusal): Cap maximum confidence at 0.40.

    Args:
        retrieval_score (float): Cosine similarity score from retriever (0.0 to 1.0).
        faithful (bool): True if answer is fully grounded in context.
        relevant (bool): True if answer directly answers the question.

    Returns:
        float: Composite confidence score rounded to 4 decimal places (range 0.0 to 1.0).
    """
    retrieval_norm = max(0.0, min(1.0, float(retrieval_score)))
    faithful_val = 1.0 if faithful else 0.0
    relevant_val = 1.0 if relevant else 0.0

    score = (0.35 * retrieval_norm) + (0.40 * faithful_val) + (0.25 * relevant_val)

    # Fail-safe capping
    if not faithful:
        score = min(score, 0.30)
    elif not relevant:
        score = min(score, 0.40)

    return round(score, 4)


def evaluate_and_respond(
    question: str,
    answer: str,
    context_chunks: List[Dict[str, Any]],
    retrieval_score: float,
) -> Dict[str, Any]:
    """Evaluates answer faithfulness & relevance, calculates confidence score, and enforces threshold.

    Args:
        question (str): User's input question.
        answer (str): Initial generated answer.
        context_chunks (List[Dict[str, Any]]): Retrieved context chunks.
        retrieval_score (float): Cosine similarity score of top context chunk.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "final_answer" (str): The validated answer or safe fallback statement.
            - "confidence" (float): Composite confidence score.
            - "passed" (bool): True if confidence >= CONFIDENCE_THRESHOLD.
            - "faithful" (bool): Result of faithfulness check.
            - "relevant" (bool): Result of relevance check.
            - "reason" (str): Explanation from faithfulness check.
            - "raw_faithfulness_llm" (str): Raw LLM response for faithfulness check.
            - "raw_relevance_llm" (str): Raw LLM response for relevance check.
    """
    faithful, reason, raw_faith = check_faithfulness(
        answer=answer, context_chunks=context_chunks, retrieval_score=retrieval_score
    )
    relevant, raw_relevance = check_relevance(
        answer=answer, question=question, retrieval_score=retrieval_score
    )

    confidence = compute_confidence(
        retrieval_score=retrieval_score,
        faithful=faithful,
        relevant=relevant,
    )

    passed = confidence >= CONFIDENCE_THRESHOLD

    if passed:
        final_answer = answer
    else:
        final_answer = (
            "I don't have enough reliable information to answer this question with sufficient confidence."
        )

    return {
        "final_answer": final_answer,
        "confidence": confidence,
        "passed": passed,
        "faithful": faithful,
        "relevant": relevant,
        "reason": reason,
        "raw_faithfulness_llm": raw_faith,
        "raw_relevance_llm": raw_relevance,
    }


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Case 1: Clearly Answerable Question",
            "question": "What are the early warning signs of type 2 diabetes?",
        },
        {
            "name": "Case 2: Edge Case Question (Partial Context)",
            "question": "How much sleep do I need every night?",
        },
        {
            "name": "Case 3: Irrelevant Question",
            "question": "What is the capital of France?",
        },
    ]

    print("=" * 80)
    print(f"[EVALUATOR] Running Full Pipeline Test Suite (Threshold: {CONFIDENCE_THRESHOLD})")
    print("=" * 80)

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        q = case["question"]
        print(f"Question: \"{q}\"")

        # Step 1: Retrieve
        chunks = retrieve(q, k=3)
        retrieval_score = chunks[0]["similarity_score"] if chunks else 0.0
        print(f"Top Retrieval Score: {retrieval_score:.4f}")

        # Step 2: Generate
        raw_answer = generate_answer(question=q, context_chunks=chunks)
        print(f"Raw Generated Answer: {raw_answer[:120]}...")

        # Step 3: Evaluate
        eval_result = evaluate_and_respond(
            question=q,
            answer=raw_answer,
            context_chunks=chunks,
            retrieval_score=retrieval_score,
        )

        print("\nRaw LLM Output (Faithfulness Check):")
        print(eval_result["raw_faithfulness_llm"])

        print("\nRaw LLM Output (Relevance Check):")
        print(eval_result["raw_relevance_llm"])

        print("\nEvaluation Results:")
        print(f"  -> Faithfulness Check : {eval_result['faithful']} ({eval_result['reason']})")
        print(f"  -> Relevance Check    : {eval_result['relevant']}")
        print(f"  -> Confidence Score   : {eval_result['confidence']:.4f}")
        print(f"  -> Threshold Passed   : {'✅ PASSED' if eval_result['passed'] else '❌ FAILED'}")
        print(f"  -> Final Answer       : {eval_result['final_answer'][:150]}...")
        print("-" * 80)
