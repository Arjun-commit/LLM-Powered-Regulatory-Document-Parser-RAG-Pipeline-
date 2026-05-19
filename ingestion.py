"""
ingestion.py

Reads a PDF page by page with PyMuPDF, splits the text into overlapping
character chunks, tags every chunk with its source filename and page number,
and writes the result to a persistent ChromaDB collection on disk.

No cloud calls. Embeddings are produced locally with BAAI/bge-small-en-v1.5.
"""

from __future__ import annotations

import os

# Force the transformers library to skip its TensorFlow probe. Some Windows
# installs have a partial tensorflow package that crashes the probe on import.
# We only use PyTorch under the hood, so disabling TF here is safe and must
# happen before sentence_transformers is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer


# ----------------------------- config ---------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION = "reg_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# embed in reasonable batches so we do not hold the whole pdf in one big tensor
EMBED_BATCH = 64


log = logging.getLogger("ingestion")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ----------------------------- pdf reading ----------------------------------

def read_pdf_pages(pdf_path) -> List[Tuple[int, str]]:
    """
    Open a PDF and return a list of (page_number, text) tuples.

    Page numbers are 1 based so they line up with what a human sees in a
    viewer. Empty / whitespace only pages are skipped because they only add
    noise to the index (typical for scan separators or blank back covers).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Tuple[int, str]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        # PyMuPDF raises a fairly generic error for corrupt files; surface it
        # with the path so the caller actually knows which file failed.
        raise RuntimeError(f"Could not open PDF {pdf_path}: {exc}") from exc

    try:
        for idx, page in enumerate(doc, start=1):
            try:
                raw = page.get_text("text") or ""
            except Exception as exc:
                # one bad page should not kill the whole ingest
                log.warning("Page %s of %s failed to extract: %s", idx, pdf_path.name, exc)
                continue

            cleaned = _normalise_whitespace(raw)
            if not cleaned:
                # skip blank pages, common in scanned filings
                continue
            pages.append((idx, cleaned))
    finally:
        doc.close()

    if not pages:
        log.warning("No extractable text found in %s. Is this a scanned PDF?", pdf_path.name)
    return pages


def _normalise_whitespace(text: str) -> str:
    # collapse runs of spaces and tabs but keep paragraph breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------- chunking -------------------------------------

# Recursive character splitter. Tries the biggest separator first and falls
# back to smaller ones until the piece fits inside CHUNK_SIZE. Overlap is
# stitched on at the end by carrying the tail of chunk N into chunk N+1.

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces = _recursive_split(text, size, _SEPARATORS)
    if overlap <= 0 or len(pieces) <= 1:
        return [p.strip() for p in pieces if p.strip()]

    out: List[str] = [pieces[0].strip()]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        merged = (prev_tail + " " + pieces[i]).strip()
        out.append(merged)
    return [c for c in out if c]


def _recursive_split(text: str, size: int, seps: List[str]) -> List[str]:
    # pick the first separator that actually shows up in the text
    sep = seps[-1]
    for s in seps:
        if s == "" or s in text:
            sep = s
            break

    if sep == "":
        # last resort: hard cut by character
        return [text[i:i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    chunks: List[str] = []
    buf = ""

    for part in parts:
        joiner = sep if buf else ""
        candidate = buf + joiner + part
        if len(candidate) <= size:
            buf = candidate
            continue

        if buf:
            chunks.append(buf)
            buf = ""

        if len(part) > size:
            # part on its own is still too big, recurse with smaller separators
            next_seps = seps[seps.index(sep) + 1:] if sep in seps else [""]
            chunks.extend(_recursive_split(part, size, next_seps))
        else:
            buf = part

    if buf:
        chunks.append(buf)
    return chunks


# ----------------------------- chroma ---------------------------------------

def get_collection(
    persist_dir: str = CHROMA_DIR,
    name: str = DEFAULT_COLLECTION,
):
    """Return a persistent ChromaDB collection, creating it if needed."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    # cosine distance pairs well with BGE embeddings, which are L2 normalised
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _stable_id(filename: str, page: int, chunk_idx: int, text: str) -> str:
    # short hash of the chunk content prevents accidental duplicates if the
    # same file is ingested twice. file + page + index keeps it readable.
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{filename}::p{page}::c{chunk_idx}::{digest}"


def ingest_pdf(
    pdf_path,
    collection=None,
    embedder=None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    progress_cb=None,
) -> int:
    """
    Ingest a single PDF into ChromaDB.

    Returns the number of chunks written. ``progress_cb`` is an optional
    callback ``fn(done_pages, total_pages)`` used by the Streamlit app to
    drive a progress bar.
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    if collection is None:
        collection = get_collection()
    if embedder is None:
        log.info("Loading embedding model %s ...", EMBED_MODEL_NAME)
        embedder = SentenceTransformer(EMBED_MODEL_NAME)

    pages = read_pdf_pages(pdf_path)
    total_pages = len(pages)
    if total_pages == 0:
        return 0

    all_texts: List[str] = []
    all_ids: List[str] = []
    all_meta: List[dict] = []

    for page_no, page_text in pages:
        page_chunks = chunk_text(page_text, size=chunk_size, overlap=chunk_overlap)
        for idx, chunk in enumerate(page_chunks):
            chunk_id = _stable_id(filename, page_no, idx, chunk)
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_meta.append({
                "source": filename,
                "page": page_no,
                "chunk_index": idx,
            })

    if not all_texts:
        log.warning("Nothing to index for %s", filename)
        return 0

    log.info("Embedding %d chunks from %s", len(all_texts), filename)

    # embed in batches and push to chroma in batches too, so memory stays sane
    written = 0
    for batch_start in range(0, len(all_texts), EMBED_BATCH):
        batch_end = min(batch_start + EMBED_BATCH, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]
        batch_ids = all_ids[batch_start:batch_end]
        batch_meta = all_meta[batch_start:batch_end]

        vectors = embedder.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # upsert so a re ingest of the same file just refreshes the rows
        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=vectors,
            metadatas=batch_meta,
        )
        written += len(batch_ids)

        if progress_cb is not None:
            try:
                progress_cb(min(batch_end, len(all_texts)), len(all_texts))
            except Exception:
                # never let a UI callback break ingestion
                pass

    log.info("Wrote %d chunks for %s", written, filename)
    return written


def ingest_folder(folder) -> dict:
    """Ingest every PDF inside a folder. Handy for batch loading."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found under %s", folder)
        return {}

    collection = get_collection()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    results: dict = {}
    for pdf in pdfs:
        try:
            n = ingest_pdf(pdf, collection=collection, embedder=embedder)
            results[pdf.name] = n
        except Exception as exc:
            log.error("Failed on %s: %s", pdf.name, exc)
            results[pdf.name] = 0
    return results


# ----------------------------- cli ------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Index a PDF or a folder of PDFs into the local ChromaDB store.",
    )
    parser.add_argument("path", help="Path to a PDF file or a folder of PDFs.")
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="ChromaDB collection name (default: %(default)s).",
    )
    parser.add_argument(
        "--persist-dir",
        default=CHROMA_DIR,
        help="Directory where ChromaDB stores its data (default: %(default)s).",
    )
    args = parser.parse_args()

    target = Path(args.path)
    collection = get_collection(persist_dir=args.persist_dir, name=args.collection)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    if target.is_dir():
        summary = ingest_folder(target)
        for name, count in summary.items():
            print(f"{name}: {count} chunks")
    else:
        count = ingest_pdf(target, collection=collection, embedder=embedder)
        print(f"{target.name}: {count} chunks")


if __name__ == "__main__":
    _cli()
