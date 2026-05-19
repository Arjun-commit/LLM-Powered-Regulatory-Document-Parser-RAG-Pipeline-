"""
app.py

Streamlit front end for the local RAG pipeline.

Run with:
    streamlit run app.py

The app does three things:
  1. Lets the user upload one or more PDFs and ingest them into ChromaDB,
     with a real progress bar driven by the ingestion callback.
  2. Shows a chat box for asking questions about the indexed documents,
     streaming the answer token by token from local Ollama.
  3. Shows the exact reranked chunks that backed each answer, so the user
     can verify the citations against the source.
"""

from __future__ import annotations

import os

# Disable the TensorFlow probe inside transformers before any ML import. A
# partial tensorflow install on Windows can otherwise crash the import chain
# at sentence_transformers load time. We only need PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import logging
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingestion import (
    CHROMA_DIR,
    DEFAULT_COLLECTION,
    EMBED_MODEL_NAME,
    get_collection,
    ingest_pdf,
)
from retrieval import (
    RERANK_THRESHOLD,
    RERANKER_NAME,
    TOP_K,
)
from llm_client import (
    OLLAMA_MODEL,
    SAFE_FALLBACK,
    health_check,
    retrieve_then_stream,
)


# ----------------------------- page setup -----------------------------------

st.set_page_config(
    page_title="Local Regulatory RAG",
    page_icon=":books:",
    layout="wide",
)


# ----------------------------- cached heavy objects -------------------------

# These three are the expensive things in the system. Streamlit reruns the
# script on every interaction, so caching is not optional, it is the only way
# the app stays usable.

@st.cache_resource(show_spinner="Loading embedding model ...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Loading reranker ...")
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_NAME, max_length=512)


@st.cache_resource
def load_collection():
    return get_collection(persist_dir=CHROMA_DIR, name=DEFAULT_COLLECTION)


# ----------------------------- session state --------------------------------

def ensure_state() -> None:
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("ingested_files", [])
    st.session_state.setdefault("model", OLLAMA_MODEL)
    st.session_state.setdefault("top_k", TOP_K)
    st.session_state.setdefault("threshold", RERANK_THRESHOLD)


# ----------------------------- ingestion ------------------------------------

def ingest_uploads(uploaded_files: list) -> None:
    """Drive ingest_pdf for each uploaded file and paint a progress bar."""
    embedder = load_embedder()
    collection = load_collection()

    for uf in uploaded_files:
        st.write(f"Ingesting **{uf.name}** ...")
        progress = st.progress(0, text="Reading and chunking ...")

        # PyMuPDF needs a file path, and we also want the original filename
        # to land in chunk metadata, so write the upload into a temp dir
        # using its real name rather than the random NamedTemporaryFile name.
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / uf.name
            target.write_bytes(uf.getbuffer())

            def _cb(done: int, total: int) -> None:
                if total <= 0:
                    return
                pct = min(int(100 * done / total), 100)
                progress.progress(pct, text=f"Embedding chunks ... {done} of {total}")

            try:
                n = ingest_pdf(
                    target,
                    collection=collection,
                    embedder=embedder,
                    progress_cb=_cb,
                )
                progress.progress(100, text=f"Done. {n} chunks indexed.")
                if uf.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uf.name)
                st.success(f"Indexed {uf.name}: {n} chunks.")
            except Exception as exc:
                progress.empty()
                st.error(f"Failed to ingest {uf.name}: {exc}")


# ----------------------------- sidebar --------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Local RAG controls")

        # ---- ollama health -------------------------------------------------
        with st.expander("Ollama status", expanded=True):
            model = st.text_input(
                "Model",
                value=st.session_state["model"],
                help="Any model pulled in your local Ollama, for example llama3 or phi3.",
            )
            st.session_state["model"] = model

            h = health_check(model)
            if h["ok"]:
                st.success(h["detail"])
            else:
                st.error(h["detail"])
                st.caption("Start Ollama and pull the model:")
                st.code(f"ollama serve\nollama pull {model}", language="bash")

            if h["models"]:
                st.caption("Models present locally:")
                st.write(", ".join(h["models"]))

        # ---- retrieval knobs ----------------------------------------------
        with st.expander("Retrieval settings"):
            st.session_state["top_k"] = st.slider(
                "Top K candidates from ChromaDB",
                min_value=3, max_value=20,
                value=int(st.session_state["top_k"]),
            )
            st.session_state["threshold"] = st.slider(
                "Rerank confidence threshold",
                min_value=0.10, max_value=0.95,
                value=float(st.session_state["threshold"]),
                step=0.05,
                help=(
                    "If the top reranked chunk scores below this value, the "
                    "LLM is bypassed and a safe fallback is returned."
                ),
            )

        st.markdown("---")

        # ---- upload --------------------------------------------------------
        st.markdown("## Add documents")
        uploaded = st.file_uploader(
            "Drop a PDF here",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded:
            if st.button("Ingest selected PDFs", type="primary", use_container_width=True):
                ingest_uploads(uploaded)

        # ---- index status --------------------------------------------------
        st.markdown("---")
        col = load_collection()
        try:
            count = col.count()
        except Exception:
            count = 0
        st.metric("Chunks in index", count)

        if st.session_state.ingested_files:
            st.markdown("**Indexed this session:**")
            for f in st.session_state.ingested_files:
                st.write(f"- {f}")

        # ---- chat controls -------------------------------------------------
        st.markdown("---")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()


# ----------------------------- chat -----------------------------------------

def _render_citations(citations: List[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources used ({len(citations)})"):
        for c in citations:
            score = c.get("score", 0.0)
            st.markdown(
                f"**{c.get('source', 'unknown')}** "
                f"page {c.get('page', 0)} "
                f"(rerank score {score:.2f})"
            )
            st.write(c.get("text", ""))
            st.markdown("---")


def render_chat() -> None:
    st.markdown("## Ask the documents")

    # replay chat history first
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_citations(msg.get("citations", []))

    question = st.chat_input("Ask a question about the indexed documents ...")
    if not question:
        return

    # paint the user message right away
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # warm caches so the spinner reflects reality
    load_embedder()
    load_reranker()
    load_collection()

    model = st.session_state["model"]
    top_k = st.session_state["top_k"]
    threshold = st.session_state["threshold"]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reranking ..."):
            chunks, gen = retrieve_then_stream(
                question,
                top_k=top_k,
                threshold=threshold,
                model=model,
            )

        placeholder = st.empty()
        buffer: List[str] = []
        for token in gen:
            buffer.append(token)
            placeholder.markdown("".join(buffer))

        final_text = ("".join(buffer)).strip() or SAFE_FALLBACK
        citations = [c.to_dict() for c in chunks]
        _render_citations(citations)

    st.session_state.chat.append({
        "role": "assistant",
        "content": final_text,
        "citations": citations,
    })


# ----------------------------- main -----------------------------------------

def main() -> None:
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    ensure_state()

    st.title("Local Regulatory Document Q and A")
    st.caption(
        "Runs end to end on your machine. PyMuPDF for parsing, "
        "BAAI/bge-small-en-v1.5 for embeddings, BAAI/bge-reranker-base for "
        "reranking, ChromaDB for the vector store, local Ollama for the LLM."
    )

    render_sidebar()

    # if there is nothing in the index yet, nudge the user toward the uploader
    col = load_collection()
    try:
        idx_size = col.count()
    except Exception:
        idx_size = 0
    if idx_size == 0:
        st.info("No documents indexed yet. Upload a PDF from the sidebar to get started.")

    render_chat()


if __name__ == "__main__":
    main()
