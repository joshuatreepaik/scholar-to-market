"""Retrieval-augmented Q&A over indexed works, with inline citations."""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from . import config, index

SYSTEM_PROMPT = (
    "You are a research-commercialization analyst. Answer the question using ONLY "
    "the provided CONTEXT of research papers. Cite the works you use inline by their "
    "id in square brackets, e.g. [W2153344788]. If the context does not contain the "
    "answer, say 'The indexed papers don't cover that.' Be concise and specific."
)


def _client() -> OpenAI:
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def retrieve(query: str, k: int | None = None) -> list[dict[str, Any]]:
    k = k or config.TOP_K
    collection = index.get_collection()
    res = collection.query(query_texts=[query], n_results=k)
    hits: list[dict[str, Any]] = []
    if res and res.get("ids") and res["ids"][0]:
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i]
            hits.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "title": meta.get("title", ""),
                "year": meta.get("year"),
                "cited_by_count": meta.get("cited_by_count", 0),
                "companies": meta.get("companies", ""),
            })
    return hits


def _format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for h in hits:
        header = f"[{h['id']}] {h['title']} ({h['year']}, cited {h['cited_by_count']}x"
        if h["companies"]:
            header += f", industry: {h['companies']}"
        header += ")"
        blocks.append(f"{header}\n{h['text'][:1200]}")
    return "\n\n---\n\n".join(blocks)


def answer(query: str, k: int | None = None) -> dict[str, Any]:
    """Retrieve relevant works and synthesize a cited answer."""
    hits = retrieve(query, k=k)
    if not hits:
        return {"answer": "No indexed papers matched. Try ingesting/indexing first.",
                "sources": []}

    context = _format_context(hits)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nCITED ANSWER:"
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return {
        "answer": resp.choices[0].message.content.strip(),
        "sources": [{"id": h["id"], "title": h["title"], "year": h["year"]} for h in hits],
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Which companies are commercializing CRISPR therapies?"
    out = answer(q)
    print(out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print(f"  [{s['id']}] {s['title']} ({s['year']})")
