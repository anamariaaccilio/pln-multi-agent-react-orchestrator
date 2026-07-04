"""
Data & Retrieval Layer — WikiQA + ChromaDB vector store.

This module provides `retrieve_context(question, top_k)` following the contract
defined in `src/retriever/interface.py`.

Usage:
    from src.retriever.retrieval_pipeline import retrieve_context
    results = retrieve_context("When was the Eiffel Tower built?", top_k=5)

The vector store is built lazily on first call (or explicitly via `initialize()`).
Building takes ~10 min on CPU the first time; subsequent runs reuse the persisted
ChromaDB directory.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wikiqa_documents"
MIN_ANSWER_LEN = 3

DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")

# ---------------------------------------------------------------------------
# Module-level state (lazily initialized)
# ---------------------------------------------------------------------------

_collection: Optional[chromadb.Collection] = None
_initialized: bool = False


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------


def _set_seeds() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def _load_wikiqa() -> pd.DataFrame:
    """Load all WikiQA splits and return a single cleaned DataFrame."""
    print("[retrieval_pipeline] Loading WikiQA dataset...")
    wikiqa = load_dataset("microsoft/wiki_qa")

    frames = []
    for split in ("train", "validation", "test"):
        df = wikiqa[split].to_pandas()
        df["split"] = split
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows (all splits): {len(full_df)}")

    # Filter out empty / too-short answers
    before = len(full_df)
    full_df = full_df[full_df["answer"].str.strip().str.len() >= MIN_ANSWER_LEN].copy()
    print(f"  Rows removed (answer too short): {before - len(full_df)}")

    # Normalize whitespace
    full_df["answer"] = full_df["answer"].str.strip()
    full_df["question"] = full_df["question"].str.strip()
    full_df["document_title"] = full_df["document_title"].str.strip()

    return full_df


def _build_document_groups(full_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct documents by grouping unique answers per title."""
    deduped = full_df.drop_duplicates(subset=["document_title", "answer"])

    doc_groups = (
        deduped.groupby("document_title")["answer"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index()
        .rename(columns={"answer": "document_text"})
    )
    doc_groups["chunk_id"] = ["doc_" + str(i) for i in range(len(doc_groups))]
    doc_groups["n_sentences"] = deduped.groupby("document_title").size().values

    print(f"  Reconstructed documents (unique titles): {len(doc_groups)}")
    print(f"  Avg document length (chars): {doc_groups['document_text'].str.len().mean():.0f}")

    return doc_groups


# ---------------------------------------------------------------------------
# ChromaDB indexing
# ---------------------------------------------------------------------------


def _get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device="cpu",
    )


def _index_collection(
    client: chromadb.PersistentClient,
    doc_groups: pd.DataFrame,
    embedding_fn: SentenceTransformerEmbeddingFunction,
    batch_size: int = 256,
) -> chromadb.Collection:
    """Create (or reuse) the ChromaDB collection with progress bar."""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # If collection already has all documents, skip re-indexing
    existing_count = collection.count()
    target_count = len(doc_groups)
    if existing_count >= target_count:
        print(f"[retrieval_pipeline] Collection '{COLLECTION_NAME}' already has "
              f"{existing_count} docs — skipping indexing.")
        return collection

    print(f"[retrieval_pipeline] Indexing {target_count} documents into ChromaDB...")

    ids = doc_groups["chunk_id"].tolist()
    documents = doc_groups["document_text"].tolist()
    metadatas = doc_groups[["document_title", "n_sentences"]].to_dict("records")

    n = len(ids)
    with tqdm(total=n, desc="Indexing", unit="doc") as pbar:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            pbar.update(end - start)

    print(f"[retrieval_pipeline] Done. Collection count = {collection.count()}")
    return collection


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize(force_rebuild: bool = False) -> None:
    """
    Load data, build embeddings, and index into ChromaDB.

    This is called automatically on the first `retrieve_context()` call, but
    you can call it explicitly if you want to control when the ~6 min indexing
    happens (e.g. in a setup script).

    Args:
        force_rebuild: If True, delete and recreate the collection even if it
                       already exists with the correct count.
    """
    global _collection, _initialized

    if _initialized and not force_rebuild:
        return

    _set_seeds()
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    # 1. Load & preprocess
    full_df = _load_wikiqa()
    doc_groups = _build_document_groups(full_df)

    # 2. Embedding function
    print("[retrieval_pipeline] Loading embedding model...")
    embedding_fn = _get_embedding_function()

    # 3. ChromaDB client + indexing
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[retrieval_pipeline] Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    _collection = _index_collection(client, doc_groups, embedding_fn)
    _initialized = True
    print("[retrieval_pipeline] Initialization complete ✓")


# ---------------------------------------------------------------------------
# Public API — matches the contract in src/retriever/interface.py
# ---------------------------------------------------------------------------


def retrieve_context(question: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a given question.

    Args:
        question: Natural language query (e.g. user question or sub-question
                  generated by the Researcher Agent).
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: "content", "source", "score" — as required
        by the interface contract in `src/retriever/interface.py`.
    """
    global _collection

    # Lazy initialization
    if not _initialized:
        initialize()

    assert _collection is not None, "Collection not initialized"

    res = _collection.query(query_texts=[question], n_results=top_k)

    results = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        results.append({
            "content": doc,
            "source": meta.get("document_title", "unknown"),
            "score": float(round(1 - dist, 4)),  # cosine distance → similarity
        })

    return results
