"""
llm_client.py

Talks to a local Ollama daemon (default http://localhost:11434), feeds it the
reranked context produced by retrieval.py, and returns a grounded answer with
inline citations of the form [Source: <filename>, Page: <n>].

Behaviour rules baked into this module:

  * If retrieval returns no chunks above the rerank threshold, the LLM is
    NOT called. The function returns the safe fallback string directly.
  * The system prompt forbids outside knowledge and forces citations.
  * Connection errors and missing models are surfaced as plain strings so
    the Streamlit layer never has to handle network exceptions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Generator, List, Optional, Tuple

import requests

from retrieval import (
    RERANK_THRESHOLD,
    TOP_K,
    RetrievedChunk,
    format_context,
    retrieve,
)


# ----------------------------- config ---------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# (connect timeout, read timeout) in seconds. Local generation on a CPU can
# easily take a minute on the first run, so the read side is generous.
REQUEST_TIMEOUT: Tuple[int, int] = (5, 180)

DEFAULT_TEMPERATURE = 0.1   # we want the model boring and faithful, not creative
DEFAULT_NUM_CTX = 4096      # context window. llama3 can take more, this is safe.

SAFE_FALLBACK = (
    "I cannot find the relevant information in the provided document "
    "to answer this question safely."
)

log = logging.getLogger("llm_client")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ----------------------------- prompts --------------------------------------

# Kept as plain strings so it is easy to skim and tweak. The system prompt is
# deliberately blunt; regulatory questions are not the place to be polite to
# the model.

SYSTEM_PROMPT = (
    "You are a careful research assistant for regulatory and financial "
    "documents. Follow every rule below without exception.\n\n"
    "RULES:\n"
    "1. Use ONLY the text inside the CONTEXT section to answer. Do not "
    "use outside knowledge, prior training data, or guesses.\n"
    "2. After every factual statement, append an inline citation copied "
    "verbatim from a citation tag in the context, formatted exactly as "
    "[Source: <filename>, Page: <page>].\n"
    "3. If the context does not contain the answer, reply with this exact "
    "sentence and nothing else:\n"
    f"   {SAFE_FALLBACK}\n"
    "4. Quote the document verbatim when stating numbers, dates, deadlines, "
    "thresholds, monetary amounts, or formal definitions. Do not paraphrase "
    "those.\n"
    "5. Do not invent citations. Do not refer to pages or sources that do "
    "not appear in the context.\n"
    "6. Keep the answer concise and direct. No preamble, no apologies, no "
    "filler phrases such as 'Based on the context'."
)


USER_TEMPLATE = (
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION:\n"
    "{question}\n\n"
    "Answer using only the context above. Place a "
    "[Source: <filename>, Page: <page>] tag after each fact you state."
)


# ----------------------------- response shape -------------------------------

@dataclass
class Answer:
    text: str
    used_fallback: bool
    chunks: List[RetrievedChunk] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "used_fallback": self.used_fallback,
            "model": self.model,
            "chunks": [c.to_dict() for c in self.chunks],
        }


class OllamaUnavailable(RuntimeError):
    """Raised internally when we cannot complete a request to Ollama."""


# ----------------------------- transport ------------------------------------

def _build_messages(question: str, context: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            context=context, question=question,
        )},
    ]


def _post_chat(
    messages: List[dict],
    model: str,
    stream: bool,
    temperature: float,
    num_ctx: int,
) -> requests.Response:
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=stream,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailable(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            "Start it with `ollama serve` and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailable(
            f"Ollama timed out after {REQUEST_TIMEOUT[1]}s while generating. "
            "Try a smaller model (for example phi3) or a shorter question."
        ) from exc

    if resp.status_code == 404:
        # Ollama returns 404 if the model name does not exist locally
        raise OllamaUnavailable(
            f"Model {model!r} is not pulled locally. "
            f"Run `ollama pull {model}` and try again."
        )
    if resp.status_code >= 400:
        # surface the body so the user sees what Ollama actually complained about
        snippet = resp.text[:300] if resp.text else ""
        raise OllamaUnavailable(
            f"Ollama returned HTTP {resp.status_code}: {snippet}"
        )

    return resp


# ----------------------------- public api -----------------------------------

def answer(
    question: str,
    top_k: int = TOP_K,
    threshold: float = RERANK_THRESHOLD,
    model: str = OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Answer:
    """
    Retrieve, then generate. Returns a fully formed Answer object.

    If no retrieved chunk clears the rerank threshold we short circuit and
    return the safe fallback string without ever calling the LLM, which is
    the whole point of the threshold.
    """
    question = (question or "").strip()
    if not question:
        return Answer(text=SAFE_FALLBACK, used_fallback=True, model="")

    chunks = retrieve(question, top_k=top_k, threshold=threshold)
    if not chunks:
        return Answer(text=SAFE_FALLBACK, used_fallback=True, model="")

    context = format_context(chunks)
    messages = _build_messages(question, context)

    try:
        resp = _post_chat(
            messages, model,
            stream=False,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        data = resp.json()
    except OllamaUnavailable as exc:
        log.error(str(exc))
        return Answer(
            text=f"LLM error: {exc}",
            used_fallback=True,
            chunks=chunks,
            model=model,
        )
    except ValueError as exc:
        # JSON decode failure, very rare but worth catching cleanly
        log.error("Could not parse Ollama response: %s", exc)
        return Answer(
            text="LLM error: malformed response from Ollama.",
            used_fallback=True,
            chunks=chunks,
            model=model,
        )

    text = ((data or {}).get("message") or {}).get("content", "").strip()
    if not text:
        # model returned nothing useful; fall back rather than show a blank
        return Answer(text=SAFE_FALLBACK, used_fallback=True, chunks=chunks, model=model)

    return Answer(text=text, used_fallback=False, chunks=chunks, model=model)


def stream_answer(
    question: str,
    top_k: int = TOP_K,
    threshold: float = RERANK_THRESHOLD,
    model: str = OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Generator[str, None, None]:
    """
    Yield response tokens as they arrive from Ollama. Useful for the chat UI.

    The first yielded value is always a plain text chunk; on retrieval miss
    we yield SAFE_FALLBACK once and stop. Citation rendering is the UI's job
    and is best done by calling `retrieve()` (or `answer()`) separately.
    """
    question = (question or "").strip()
    if not question:
        yield SAFE_FALLBACK
        return

    chunks = retrieve(question, top_k=top_k, threshold=threshold)
    if not chunks:
        yield SAFE_FALLBACK
        return

    context = format_context(chunks)
    messages = _build_messages(question, context)

    try:
        resp = _post_chat(
            messages, model,
            stream=True,
            temperature=temperature,
            num_ctx=num_ctx,
        )
    except OllamaUnavailable as exc:
        yield f"LLM error: {exc}"
        return

    # Ollama streams newline delimited JSON, one object per line.
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line.decode("utf-8"))
        except json.JSONDecodeError:
            # skip lines we can't parse; usually a keep alive or partial flush
            continue

        piece = ((obj or {}).get("message") or {}).get("content", "")
        if piece:
            yield piece
        if obj.get("done"):
            break


def retrieve_then_stream(
    question: str,
    top_k: int = TOP_K,
    threshold: float = RERANK_THRESHOLD,
    model: str = OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Tuple[List[RetrievedChunk], Generator[str, None, None]]:
    """
    Convenience for the Streamlit chat: returns (chunks, generator) so the UI
    can paint the citation panel immediately and stream the answer next to it.
    """
    question = (question or "").strip()
    chunks = retrieve(question, top_k=top_k, threshold=threshold) if question else []

    def _gen() -> Generator[str, None, None]:
        if not chunks:
            yield SAFE_FALLBACK
            return
        context = format_context(chunks)
        messages = _build_messages(question, context)
        try:
            resp = _post_chat(
                messages, model,
                stream=True,
                temperature=temperature,
                num_ctx=num_ctx,
            )
        except OllamaUnavailable as exc:
            yield f"LLM error: {exc}"
            return
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            piece = ((obj or {}).get("message") or {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                break

    return chunks, _gen()


# ----------------------------- health check ---------------------------------

def health_check(model: str = OLLAMA_MODEL) -> dict:
    """
    Quick probe used by the Streamlit sidebar. Returns a dict with:
      ok      : bool, true if Ollama is up and the model is pulled
      detail  : human readable explanation
      models  : list of model names Ollama is currently serving
    """
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        tags = r.json().get("models", []) or []
        names = [t.get("name", "") for t in tags]
        # tags look like "llama3:latest"; allow loose prefix match
        present = any(n == model or n.startswith(model + ":") for n in names)
        if present:
            detail = f"Connected. Model {model!r} is available."
        else:
            detail = (
                f"Connected, but model {model!r} is not pulled. "
                f"Run `ollama pull {model}`."
            )
        return {"ok": present, "detail": detail, "models": names}
    except Exception as exc:
        return {
            "ok": False,
            "detail": f"Cannot reach Ollama at {OLLAMA_URL}: {exc}",
            "models": [],
        }


# ----------------------------- cli ------------------------------------------

def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Ask the local RAG pipeline a question.",
    )
    p.add_argument("question", help="The question to ask.")
    p.add_argument("--model", default=OLLAMA_MODEL)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--threshold", type=float, default=RERANK_THRESHOLD)
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout.")
    args = p.parse_args()

    if args.stream:
        chunks, gen = retrieve_then_stream(
            args.question, top_k=args.top_k, threshold=args.threshold, model=args.model,
        )
        for tok in gen:
            print(tok, end="", flush=True)
        print()
        if chunks:
            print("\n--- citations used ---")
            for c in chunks:
                print(f"  {c.as_citation()}  score={c.score:.3f}")
        return

    a = answer(
        args.question,
        top_k=args.top_k,
        threshold=args.threshold,
        model=args.model,
    )
    print(a.text)
    if a.chunks:
        print("\n--- citations used ---")
        for c in a.chunks:
            print(f"  {c.as_citation()}  score={c.score:.3f}")


if __name__ == "__main__":
    _cli()
