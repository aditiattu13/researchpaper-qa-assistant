import os
import re
import unicodedata
from functools import lru_cache

import faiss
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

NUMBERED_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\-, ]{2,60})\s*$"
)
NAMED_RE = re.compile(
    r"^\s*(Abstract|References|Acknowledg\w+|Appendix.*)\s*$"
)


@lru_cache(maxsize=1)
def get_embedder():
    """Load the embedding model once per Space process."""
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm_client():
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing LLM_API_KEY. Add it under your Hugging Face Space Settings > Secrets."
        )

    base_url = os.environ.get(
        "LLM_BASE_URL", "https://api.groq.com/openai/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_name():
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    if not model:
        raise RuntimeError("LLM_MODEL is empty. Set it in your Space Variables.")
    return model


def _add_chunks(words, page, section, out, chunk_words, overlap):
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < 30:
            continue
        out.append({
            "text": " ".join(piece),
            "page": page,
            "section": section
        })


def parse_pdf(path, chunk_words=200, overlap=40):
    """Extract text and return chunks containing text, page and section."""
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be non-negative and smaller than chunk_words")

    chunks = []
    section = "Front matter"
    done = False

    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            words = []

            page_text = unicodedata.normalize("NFKC", page.get_text())
            for line in page_text.split("\n"):
                match = NUMBERED_RE.match(line) or NAMED_RE.match(line)

                if match:
                    _add_chunks(
                        words, page_num, section, chunks, chunk_words, overlap
                    )
                    words = []
                    section = match.group(0).strip()

                    if section.startswith("References"):
                        done = True
                        break
                    continue

                words.extend(line.split())

            _add_chunks(words, page_num, section, chunks, chunk_words, overlap)

            if done:
                break

    return chunks


def build_index(chunks):
    if not chunks:
        raise ValueError("No text chunks were extracted from this PDF.")

    vectors = get_embedder().encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True
    )
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query, index, chunks, k=5):
    if not query or not query.strip():
        raise ValueError("Please enter a question.")

    if index.ntotal == 0 or not chunks:
        return []

    query_vector = get_embedder().encode(
        ["Represent this sentence for searching relevant passages: " + query],
        normalize_embeddings=True
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    k = min(k, len(chunks), index.ntotal)
    scores, ids = index.search(query_vector, k)

    results = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id < 0:
            continue
        results.append(dict(chunks[int(chunk_id)], score=float(score)))

    return results


def answer(query, index, chunks, k=5):
    hits = retrieve(query, index, chunks, k)

    if not hits:
        return "The paper does not appear to address this.", []

    context = "\n\n".join(
        f"[{i + 1}] (Section: {hit['section']}, Page {hit['page']})\n"
        f"{hit['text']}"
        for i, hit in enumerate(hits)
    )

    system_prompt = (
        "You answer questions about a scientific paper using ONLY the numbered "
        "sources provided. Cite sources inline like [1] or [2][3] after each claim. "
        "If the sources do not contain the answer, say exactly: "
        "'The paper does not appear to address this.' Do not use outside knowledge."
    )

    response = get_llm_client().chat.completions.create(
        model=get_model_name(),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {query}"
            }
        ],
    )

    response_text = response.choices[0].message.content or ""
    response_text = re.sub(r"【(\d+)[^】]*】", r"[\1]", response_text)
    return response_text, hits


def summarize(index, chunks):
    if not chunks:
        return "No paper has been indexed yet."

    queries = [
        "What problem does this paper address and why does it matter?",
        "What methods and data were used?",
        "What are the main results and findings?",
        "What are the limitations and conclusions?"
    ]

    seen = set()
    picked = []

    for query in queries:
        for hit in retrieve(query, index, chunks, k=3):
            key = (hit["page"], hit["text"][:50])
            if key not in seen:
                seen.add(key)
                picked.append(hit)

    context = "\n\n".join(
        f"(Section: {hit['section']}, Page {hit['page']})\n{hit['text']}"
        for hit in picked
    )

    response = get_llm_client().chat.completions.create(
        model=get_model_name(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the paper from the excerpts only. Output: a 3-4 "
                    "sentence summary, then 'Key findings' as 3-5 bullets. "
                    "Do not invent numbers."
                )
            },
            {"role": "user", "content": context}
        ],
    )

    return response.choices[0].message.content or "No summary was generated."
