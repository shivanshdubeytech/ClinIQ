"""Document ingestion module for ClinIQ medical RAG system.

Processes medical Q&A data (combining seed data and MedQuAD dataset) into overlapping
text chunks, computes dense vector embeddings using sentence-transformers, and persists
them to a ChromaDB collection.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, setup_logger
from data.seed_data import MEDICAL_QA

logger = setup_logger("ingest")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits a body of text into overlapping word-based chunks.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): Maximum number of words per chunk. Defaults to CHUNK_SIZE from config.
        overlap (int): Number of overlapping words between consecutive chunks. Defaults to CHUNK_OVERLAP from config.

    Returns:
        List[str]: A list of text chunk strings.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    step = chunk_size - overlap
    if step <= 0:
        step = 1

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks


def load_all_dataset_entries() -> List[Dict[str, str]]:
    """Loads and merges medical QA pairs from seed_data.py and medquad_subset.json."""
    entries: List[Dict[str, str]] = []

    # 1. Load verified seed data
    for item in MEDICAL_QA:
        entries.append(
            {
                "topic": item.get("topic", "General Health"),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "source": item.get("source", "ClinIQ Seed Data"),
            }
        )

    # 2. Load MedQuAD subset JSON if available
    medquad_path = PROJECT_ROOT / "data" / "medquad_subset.json"
    if medquad_path.exists():
        try:
            with open(medquad_path, "r", encoding="utf-8") as f:
                medquad_entries = json.load(f)
                if isinstance(medquad_entries, list):
                    entries.extend(medquad_entries)
                    print(f"[INGEST] Loaded {len(medquad_entries)} entries from '{medquad_path.name}'.")
        except Exception as e:
            print(f"[INGEST] Warning: Failed to load '{medquad_path}': {e}")
    else:
        print(
            f"[INGEST] Notice: '{medquad_path.name}' not found. Run 'python data/load_medquad.py' to generate MedQuAD dataset."
        )

    print(f"[INGEST] Total merged dataset size: {len(entries)} Q&A pairs.")
    return entries


def build_knowledge_base(force: bool = False, batch_size: int = 64) -> int:
    """Builds and persists the ChromaDB medical knowledge base vector store.

    Combines question and answer pairs from seed_data and MedQuAD, generates text chunks,
    computes embeddings in batches via SentenceTransformer, and stores them in ChromaDB.

    Args:
        force (bool): If True, overwrites and rebuilds the knowledge base even if it exists.
            Defaults to False.
        batch_size (int): Size of embedding calculation batches. Defaults to 64.

    Returns:
        int: Total number of text chunks stored in the ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_name = "medical_knowledge"

    # Check existing collections and document counts to prevent duplicate ingestion
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
        existing_count = collection.count()
        if existing_count > 0 and not force:
            print(
                f"[INGEST] Collection '{collection_name}' already contains {existing_count} chunks. "
                "Skipping ingestion (pass force=True or run with --force to overwrite)."
            )
            return existing_count
        elif force:
            client.delete_collection(name=collection_name)
            collection = client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
    else:
        collection = client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    all_entries = load_all_dataset_entries()

    # Process dataset into documents, metadatas, and IDs
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    for item_idx, entry in enumerate(all_entries):
        question = entry["question"]
        answer = entry["answer"]
        topic = entry.get("topic", "General Health")
        source = entry.get("source", "Medical Knowledge")

        full_text = f"Question: {question}\nAnswer: {answer}"
        chunks = chunk_text(full_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{item_idx}_chunk_{chunk_idx}"
            documents.append(chunk)
            metadatas.append(
                {
                    "topic": topic,
                    "source": source,
                    "source_question": question,
                }
            )
            ids.append(chunk_id)

    if not documents:
        print("[INGEST] No documents found to ingest.")
        return 0

    # Compute dense embeddings explicitly using SentenceTransformer model (batched)
    print(f"[INGEST] Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[INGEST] Generating embeddings for {len(documents)} chunks (batch size: {batch_size})...")
    embeddings = embedder.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    # Add all chunks, embeddings, and metadatas to ChromaDB collection
    print(f"[INGEST] Adding {len(documents)} chunks to ChromaDB collection...")
    # ChromaDB accepts adding in batches
    db_batch_size = 250
    for i in range(0, len(documents), db_batch_size):
        end_idx = min(i + db_batch_size, len(documents))
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )

    stored_count = collection.count()
    print(f"[INGEST] Successfully stored {stored_count} chunks into ChromaDB collection '{collection_name}'.")
    return stored_count


if __name__ == "__main__":
    force_flag = "--force" in sys.argv or "-f" in sys.argv
    stored_chunks = build_knowledge_base(force=force_flag)
