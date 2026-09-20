"""Document retrieval module for ClinIQ medical RAG system.

Performs semantic vector search against ChromaDB medical knowledge base using
sentence-transformers embeddings, followed by a keyword-overlap re-ranking step
to prioritize on-topic medical content over vague semantic overlaps.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-detect and include project venv site-packages if running under global/system python
_venv_site = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

# Ensure stdout handles UTF-8 unicode printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, setup_logger

logger = setup_logger("retriever")

# Global cache for embedder instance to avoid re-loading weights on every query
_EMBEDDER: Any = None

# Common English and query stop words to exclude from keyword extraction
STOPWORDS: Set[str] = {
    "a", "an", "the", "what", "is", "are", "was", "were", "causes", "cause",
    "how", "why", "do", "does", "did", "to", "in", "of", "and", "or", "for",
    "with", "on", "at", "by", "from", "about", "can", "could", "should", "would",
    "my", "your", "his", "her", "their", "its", "tell", "me", "about"
}

# Medical key phrase aliases to ensure synonym boosting
MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    "hypertension": ["high blood pressure", "blood pressure", "hypertension", "bp"],
    "blood pressure": ["hypertension", "blood pressure", "high blood pressure"],
    "high blood pressure": ["hypertension", "blood pressure", "high blood pressure"],
    "diabetes": ["diabetes", "diabetic", "blood sugar", "glucose"],
    "asthma": ["asthma", "wheezing", "airways", "respiratory"],
    "migraine": ["migraine", "headache", "aura", "tension headache"],
    "sleep": ["sleep", "insomnia", "apnea", "sleep deprivation"],
    "fever": ["fever", "viral fever", "temperature", "pyrexia", "flu", "chills", "infection", "cold"],
    "viral": ["virus", "viral", "infection", "viral fever", "influenza", "flu"],
    "flu": ["influenza", "flu", "cold", "viral fever", "fever", "cough", "infection"],
    "cold": ["common cold", "rhinovirus", "cold", "flu", "sore throat", "sneezing", "congestion"],
    "infection": ["infection", "infectious", "viral", "bacterial", "pathogen", "fever"],
    "cough": ["cough", "bronchitis", "respiratory", "phlegm", "cold", "throat"],
    "pain": ["pain", "ache", "soreness", "discomfort", "headache"],
    "heart": ["cardiac", "heart", "cardiovascular", "coronary", "arrhythmia", "chest pain"],
    "allergy": ["allergies", "allergic", "histamine", "sneezing", "rash", "hay fever"],
    "stomach": ["gastrointestinal", "digestive", "stomach", "nausea", "abdomen", "acid reflux", "gerd"],
    "cancer": ["oncology", "cancer", "tumor", "carcinoma", "malignant", "leukemia", "lymphoma"],
    "fatigue": ["tired", "exhaustion", "fatigue", "lethargy", "weakness"],
}


def _get_embedder() -> SentenceTransformer:
    """Lazy-loads and caches the SentenceTransformer embedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def _extract_query_keywords(query: str) -> List[str]:
    """Extracts key medical terms and synomym aliases from input query."""
    words = re.findall(r"\w+", query.lower())
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]

    query_lower = query.lower()
    for key_phrase, aliases in MEDICAL_SYNONYMS.items():
        if key_phrase in query_lower:
            keywords.extend(aliases)

    return list(set(keywords))


def _calculate_rerank_boost(
    query: str, query_keywords: List[str], topic: str, source_question: str, text: str
) -> float:
    """Calculates keyword-overlap and medical condition topic re-ranking boost score."""
    if not query:
        return 0.0

    q_lower = query.lower()
    topic_lower = topic.lower()
    sq_lower = source_question.lower()
    text_lower = text.lower()

    boost = 0.0

    # 1. Dynamic Topic & Query Overlap Boost
    query_words = [w for w in re.findall(r"\w+", q_lower) if w not in STOPWORDS and len(w) > 2]
    for w in query_words:
        if w in topic_lower:
            boost += 0.25
        if w in sq_lower:
            boost += 0.15

    # 2. Medical Keyword & Synonym Overlap Boost
    for kw in query_keywords:
        if len(kw) > 2 and kw in topic_lower:
            boost += 0.15
        if len(kw) > 3 and kw in sq_lower:
            boost += 0.10
        if len(kw) > 3 and kw in text_lower:
            boost += 0.03

    # 3. Specific Topic Enhancements
    if ("fever" in q_lower or "viral" in q_lower or "flu" in q_lower) and (
        "cold" in topic_lower or "flu" in topic_lower or "fever" in topic_lower or "infection" in topic_lower
    ):
        boost += 0.30
    elif ("high blood pressure" in q_lower or "hypertension" in q_lower) and (
        "hypertension" in topic_lower or "high blood pressure" in topic_lower
    ):
        boost += 0.35
    elif ("diabetes" in q_lower or "diabetic" in q_lower) and (
        "diabetes" in topic_lower or "diabetic" in topic_lower
    ):
        boost += 0.35
    elif "asthma" in q_lower and "asthma" in topic_lower:
        boost += 0.35
    elif ("migraine" in q_lower or "headache" in q_lower) and (
        "migraine" in topic_lower or "headache" in topic_lower
    ):
        boost += 0.35
    elif ("sleep" in q_lower or "insomnia" in q_lower) and (
        "sleep" in topic_lower or "insomnia" in topic_lower
    ):
        boost += 0.35

    # 4. Off-Topic Distractor Penalty:
    # Penalize chunks whose topic belongs to a major unrelated condition not mentioned in the query
    if not any(d in q_lower for d in ["diabetes", "diabetic"]) and any(
        d in topic_lower for d in ["diabetes", "diabetic"]
    ):
        boost -= 0.15

    if not any(d in q_lower for d in ["hypertension", "blood pressure"]) and any(
        d in topic_lower for d in ["hypertension", "blood pressure"]
    ):
        boost -= 0.15

    return boost



def retrieve(query: str, k: int = 3, pool_k: int = 20) -> List[Dict[str, Any]]:
    """Retrieves top-k relevant text chunks with hybrid vector search and keyword re-ranking.

    Args:
        query (str): The search query or user question.
        k (int): Number of final top results to return. Defaults to 3.
        pool_k (int): Number of initial candidate chunks to pull from vector store. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: List of top-k retrieved chunk dictionaries containing:
            - "text" (str): Retrieved chunk text.
            - "topic" (str): Topic category.
            - "source" (str): Provenance source.
            - "source_question" (str): Original question text.
            - "raw_score" (float): Original vector cosine similarity score.
            - "similarity_score" (float): Effective score after re-ranking boost.

    Raises:
        RuntimeError: If database or collection is missing/empty.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        raise RuntimeError(
            f"[RETRIEVER ERROR] ChromaDB directory '{CHROMA_DB_PATH}' was not found. "
            "Please run 'python src/ingest.py' first to build the knowledge base."
        )

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = [c.name for c in client.list_collections()]

    if "medical_knowledge" not in collections:
        raise RuntimeError(
            "[RETRIEVER ERROR] Collection 'medical_knowledge' does not exist in ChromaDB. "
            "Please run 'python src/ingest.py' first to build the knowledge base."
        )

    collection = client.get_collection(name="medical_knowledge")
    if collection.count() == 0:
        raise RuntimeError(
            "[RETRIEVER ERROR] Collection 'medical_knowledge' is empty. "
            "Please run 'python src/ingest.py' first to populate the knowledge base."
        )

    # Generate query embedding
    embedder = _get_embedder()
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

    # Query initial pool of candidates (pool_k=10) from vector store
    fetch_k = max(k, pool_k)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    query_keywords = _extract_query_keywords(query)
    candidate_list: List[Dict[str, Any]] = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance metric: Cosine Similarity = 1.0 - distance
        raw_similarity = round(max(0.0, 1.0 - float(dist)), 4)
        topic = meta.get("topic", "Unknown")
        source = meta.get("source", "Medical Knowledge")
        source_question = meta.get("source_question", "Unknown")

        # Calculate keyword & topic overlap boost
        boost = _calculate_rerank_boost(
            query=query,
            query_keywords=query_keywords,
            topic=topic,
            source_question=source_question,
            text=doc,
        )
        effective_score = round(min(1.0, max(0.0, raw_similarity + boost)), 4)

        candidate_list.append(
            {
                "text": doc,
                "topic": topic,
                "source": source,
                "source_question": source_question,
                "raw_score": raw_similarity,
                "boost": round(boost, 4),
                "similarity_score": effective_score,
            }
        )

    # Sort candidates by re-ranked effective similarity score descending
    candidate_list.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_k_results = candidate_list[:k]

    # Log full retrieval score breakdown
    logger.info(f"Retrieval Breakdown for Query: '{query}' (Keywords: {query_keywords})")
    for idx, item in enumerate(top_k_results, 1):
        logger.info(
            f"  Rank #{idx} | Topic: '{item['topic']}' | "
            f"Raw Sim: {item['raw_score']:.4f} | Boost: {item['boost']:+.4f} | "
            f"Effective Score: {item['similarity_score']:.4f}"
        )

    return top_k_results


if __name__ == "__main__":
    test_queries = [
        "What causes high blood pressure?",
        "What causes insomnia and sleep deprivation?",
        "What is the capital of France?",
    ]

    print("=" * 80)
    print("[RETRIEVER] Running Hybrid Retrieval & Re-ranking Test Suite")
    print("=" * 80)

    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        print("-" * 60)
        res = retrieve(q, k=3, pool_k=10)
        for idx, item in enumerate(res, 1):
            print(
                f"  [{idx}] Topic: {item['topic']:<35} | "
                f"Raw: {item['raw_score']:.4f} | "
                f"Boost: +{item['boost']:.4f} | "
                f"Effective: {item['similarity_score']:.4f}"
            )
