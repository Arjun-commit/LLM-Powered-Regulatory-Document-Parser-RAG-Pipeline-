"""
retrieval.py

Two stage retrieval over the local ChromaDB index:

  1. Embed the query with BAAI/bge-small-en-v1.5 and pull the top N candidate
     chunks from Chroma.
  2. Score each candidate against the query with the BAAI/bge-reranker-base
     cross encoder, squash the logits with sigmoid so the score lives in
     [0, 1], and drop anything below the configured confidence floor.

If nothing survives the rerank threshold the function returns an empty list.
The calling layer (llm_client.py) is responsible for the safe fallback reply,
not this module. That keeps the retrieval logic pure.
"""

from __future__ import annotations

import os

# Tell transformers to skip its TensorFlow probe. A partial TF install on the
# host can break the import otherwise. PyTorch is the only backend we need.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ----------------------------- config ---------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = "BAAI/bge-reranker-base"

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "reg_docs"

TOP_K = 10            # vector candidates pulled from chroma
RERANK_THRESHOLD = 0.60  # minimum sigmoid score to keep a chunk

# BGE asks for an instruction prefix on the query side only, never on the
# passages. Skipping it on documents was a deliberate choice in ingestion.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


log = logging.getLogger("retrieval")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ----------------------------- data shape -----------------------------------

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float           # rerank score, sigmoid space, 0 to 1
    vector_distance: float  # raw cosine distance from chroma, for debugging

    def as_citation(self) -> str:
        return f"[Source: {self.source}, Page: {self.page}]"

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------- model loaders --------------------------------

# Loading the cross encoder takes a few seconds and a chunk of RAM, so we keep
# it in module level globals and reuse it across queries. The Streamlit app
# also caches them at the UI layer, but the belt and braces does not hurt.

_embedder: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info("Loading embedding model %s", EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info("Loading reranker %s", RERANKER_NAME)
        _reranker = CrossEncoder(RERANKER_NAME, max_length=512)
    return _reranker


def get_collection(
    persist_dir: str = CHROMA_DIR,
    name: str = COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ----------------------------- helpers --------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable form
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------- main api -------------------------------------

def retrieve(
    query: str,
    top_k: int = TOP_K,
    threshold: float = RERANK_THRESHOLD,
    collection=None,
    embedder: Optional[SentenceTransformer] = None,
    reranker: Optional[CrossEncoder] = None,
) -> List[RetrievedChunk]:
    """
    Run the two stage retrieval and return reranked chunks that clear the
    confidence floor. Result is sorted by score, highest first.
    """
    query = (query or "").strip()
    if not query:
        return []

    if collection is None:
        collection = get_collection()
    if embedder is None:
        embedder = get_embedder()
    if reranker is None:
        reranker = get_reranker()

    # ---- stage 1: dense recall ---------------------------------------------
    query_vec = embedder.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    try:
        raw = collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.error("Chroma query failed: %s", exc)
        return []

    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    if not docs:
        log.info("No candidates returned by Chroma for query: %r", query)
        return []

    # ---- stage 2: cross encoder rerank -------------------------------------
    pairs = [(query, doc) for doc in docs]
    try:
        logits = reranker.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        log.error("Reranker failed, falling back to vector order: %s", exc)
        # graceful fallback: keep the chroma order but mark scores as unknown
        # so the threshold filter still applies and probably drops them.
        logits = np.full(len(docs), -10.0)

    scores = _sigmoid(np.asarray(logits, dtype=np.float64))

    results: List[RetrievedChunk] = []
    for doc, meta, dist, score in zip(docs, metas, dists, scores):
        meta = meta or {}
        results.append(
            RetrievedChunk(
                text=doc,
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 0) or 0),
                score=float(score),
                vector_distance=float(dist) if dist is not None else float("nan"),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)

    if not results:
        return []

    top_score = results[0].score
    log.info(
        "Query %r: %d candidates, top rerank score %.3f, threshold %.2f",
        query, len(results), top_score, threshold,
    )

    if top_score < threshold:
        # nothing trustworthy enough to feed the LLM
        return []

    filtered = [r for r in results if r.score >= threshold]
    return filtered


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Stitch reranked chunks into a single context block that the LLM prompt
    can quote from. Each block is prefixed with its citation tag so the model
    can copy it verbatim into the answer.
    """
    if not chunks:
        return ""
    parts: List[str] = []
    for c in chunks:
        parts.append(f"{c.as_citation()}\n{c.text}")
    return "\n\n".join(parts)


# ----------------------------- cli for sanity checks -----------------------

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ad hoc retrieval probe.")
    parser.add_argument("query", help="The question to test.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=RERANK_THRESHOLD)
    args = parser.parse_args()

    hits = retrieve(args.query, top_k=args.top_k, threshold=args.threshold)
    if not hits:
        print("No chunks cleared the threshold.")
        return

    for h in hits:
        print(json.dumps(h.to_dict(), indent=2))
        print("-" * 60)


if __name__ == "__main__":
    _cli()
