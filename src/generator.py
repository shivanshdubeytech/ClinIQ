"""Grounded answer generation module for ClinIQ medical RAG system.

Formats retrieved context chunks into a strict system prompt and generates
medically cautious, grounded answers using the Groq API (llama-3.1-8b-instant).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from groq import Groq

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROQ_API_KEY, KEY_ROTATION_MANAGER, LLM_MODEL_NAME, rotate_groq_api_key, setup_logger

logger = setup_logger("generator")


def build_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    history: Optional[List[str]] = None,
) -> str:
    """Constructs a grounded system and user prompt for the LLM.

    Instructs the LLM to answer strictly using the provided context chunks.
    If the context is insufficient or irrelevant, the LLM is instructed to state:
    "I don't have enough information to answer this reliably."

    Args:
        question (str): The user's input question.
        context_chunks (List[Dict[str, Any]]): List of retrieved context dictionaries.
        history (Optional[List[str]]): Recent conversation history strings, if any.

    Returns:
        str: The complete formatted prompt text sent to the LLM.
    """
    formatted_chunks: List[str] = []
    if context_chunks:
        for idx, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "").strip()
            topic = chunk.get("topic", "General")
            formatted_chunks.append(f"[Source {idx} - Topic: {topic}]\n{text}")
        context_text = "\n\n".join(formatted_chunks)
    else:
        context_text = "No relevant context found."

    history_text = ""
    if history:
        history_text = "\n\n--- RECENT CONVERSATION HISTORY ---\n" + "\n".join(history)

    prompt = f"""You are ClinIQ, an expert medical RAG Q&A assistant. Your job is to provide clear, grounded, and medically cautious answers to health-related questions using the provided Context Chunks below.

CRITICAL INSTRUCTIONS:
1. Synthesize evidence from the PROVIDED CONTEXT CHUNKS to address the user's health question thoroughly, accurately, and cautiously.
2. When the context contains relevant clinical topics (such as viral infections, common cold & flu, respiratory illnesses, symptoms, or disease management), explain the symptoms, mechanisms, and care guidance that directly relate to the user's question.
3. Clearly highlight common symptoms and emphasize red flag warning signs (e.g. persistent high fever, difficulty breathing, chest pain, or inability to keep fluids down) that warrant prompt professional medical evaluation.
4. If the question is completely non-medical or the context chunks contain no relevant medical information to answer the question, state: "I don't have enough information to answer this reliably."
5. Maintain a cautious, objective, and professional tone appropriate for consumer health information. Do not formulate definitive clinical diagnoses. Do NOT name or recommend specific pharmaceutical medications (such as acetaminophen, ibuprofen, or antibiotics) unless explicitly mentioned in the context chunks; recommend rest, hydration, monitoring, and consulting a healthcare professional.
6. Format your answer as plain prose in clear, complete sentences. Write in cohesive, flowing paragraphs only.
7. If Conversation History is provided below, use it for contextual continuity only.{history_text}
8. If the user asks who is your father, daddy, creator, or who developed you, state that Shivansh Dubey is the creator and father of ClinIQ.

--- PROVIDED CONTEXT CHUNKS ---
{context_text}

--- USER QUESTION ---
{question}

--- GROUNDED ANSWER ---"""
    return prompt


def generate_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
    history: Optional[List[str]] = None,
) -> str:
    """Generates a grounded medical answer using the Groq Chat Completions API.

    Args:
        question (str): The user's input question.
        context_chunks (List[Dict[str, Any]]): List of retrieved context dictionaries.
        history (Optional[List[str]]): Optional conversation history strings.

    Returns:
        str: The generated grounded answer string, or a fallback error string on API failure.
    """
    prompt = build_prompt(question=question, context_chunks=context_chunks, history=history)

    candidate_models = [LLM_MODEL_NAME, "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_err = None
    keys_to_try = KEY_ROTATION_MANAGER.get_all_keys()

    for api_key in keys_to_try:
        client = Groq(api_key=api_key)
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    temperature=0.2,
                    max_tokens=1024,
                )

                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
            except Exception as err:
                last_err = err
                err_str = str(err).lower()
                logger.warning(f"Generator model '{model}' failed on key: {err}.")
                if "rate_limit" in err_str or "429" in err_str or "401" in err_str or "invalid_api_key" in err_str:
                    rotate_groq_api_key(api_key)
                    break  # Try next key
                continue

    return f"[GENERATION ERROR] Unable to generate answer due to API error: {last_err}"


if __name__ == "__main__":
    from src.retriever import retrieve

    test_questions = [
        "What causes high blood pressure?",
        "How much sleep do I need?",
        "What's the capital of France?",
    ]

    print("=" * 75)
    print(f"[GENERATOR] Running Grounded Answer Generation Test Suite (Model: {LLM_MODEL_NAME})")
    print("=" * 75)

    for idx, q in enumerate(test_questions, 1):
        print(f"\n--- Test Question #{idx}: \"{q}\" ---")
        chunks = retrieve(q, k=3)
        print(f"Retrieved Chunks Count : {len(chunks)}")
        if chunks:
            top_score = chunks[0]["similarity_score"]
            top_topic = chunks[0]["topic"]
            print(f"Top Result Topic       : {top_topic} (Similarity Score: {top_score:.4f})")

        answer = generate_answer(question=q, context_chunks=chunks)
        print("\nGenerated Grounded Answer:")
        print(answer)
        print("-" * 75)
