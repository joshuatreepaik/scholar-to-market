"""
Multithreaded ingestion of research works from the OpenAlex API.

OpenAlex is a free, open catalog of ~250M scholarly works (no API key needed).
This module fetches works matching a query and normalizes each into a compact
record with the fields we need for RAG and commercialization analytics.

Design notes
------------
- OpenAlex basic paging allows ``page * per_page <= 10_000``. Because pages are
  independent, we fetch the *first* page synchronously (to learn the total
  count), then pull the remaining pages **concurrently** with a
  ``ThreadPoolExecutor`` -- an I/O-bound workload where threading gives a large
  wall-clock win over sequential requests.
- Every request goes through a ``requests.Session`` with automatic retry and
  exponential backoff, so transient 429/5xx responses don't abort a long run.
- Results are de-duplicated by OpenAlex id and streamed to a JSONL file.

For datasets larger than 10k records, OpenAlex publishes a full ~400 GB data
snapshot (S3); the same per-record ``normalize_work`` logic applies when
streaming snapshot partitions. This module targets the live API slice used by
the demo.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .. import config

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
PER_PAGE = 200                # OpenAlex maximum
MAX_PAGEABLE = 10_000         # OpenAlex basic-paging ceiling

# Only the fields we actually use -- keeps payloads small and fast.
SELECT_FIELDS = ",".join([
    "id", "doi", "title", "publication_year", "cited_by_count",
    "authorships", "primary_topic", "concepts",
    "funders", "referenced_works_count", "abstract_inverted_index",
    "open_access",
])


def _make_session() -> requests.Session:
    """A session that retries on transient errors with exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,                       # 1s, 2s, 4s, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    return session


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild plain text."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def normalize_work(w: dict[str, Any]) -> dict[str, Any]:
    """Flatten a raw OpenAlex work into the fields the pipeline consumes."""
    authorships = w.get("authorships") or []
    institutions: list[dict[str, str]] = []
    companies: list[str] = []
    for a in authorships:
        for inst in a.get("institutions") or []:
            name = inst.get("display_name")
            itype = inst.get("type")
            if name:
                institutions.append({"name": name, "type": itype or "unknown"})
                if itype == "company":
                    companies.append(name)

    topic = (w.get("primary_topic") or {}).get("display_name")
    concepts = [c.get("display_name") for c in (w.get("concepts") or [])[:5] if c.get("display_name")]
    funders = [f.get("display_name") for f in (w.get("funders") or []) if f.get("display_name")]

    return {
        "id": w.get("id"),
        "doi": w.get("doi"),
        "title": w.get("title") or "",
        "year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count", 0),
        "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
        "authors": [a.get("author", {}).get("display_name") for a in authorships if a.get("author")],
        "institutions": institutions,
        "companies": sorted(set(companies)),
        "topic": topic,
        "concepts": concepts,
        "funders": sorted(set(funders)),
        "referenced_works_count": w.get("referenced_works_count", 0),
        "is_oa": bool((w.get("open_access") or {}).get("is_oa")),
    }


def _fetch_page(session: requests.Session, params: dict[str, Any], page: int) -> list[dict[str, Any]]:
    q = dict(params, page=page, **{"per-page": PER_PAGE})
    resp = session.get(OPENALEX_WORKS_URL, params=q, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_works(
    query: str,
    max_records: int = 1000,
    workers: int | None = None,
    filters: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch and normalize up to ``max_records`` works matching ``query``.

    Pages after the first are fetched concurrently across ``workers`` threads.
    """
    workers = workers or config.INGEST_WORKERS
    session = _make_session()

    base_params: dict[str, Any] = {"search": query, "select": SELECT_FIELDS}
    if filters:
        base_params["filter"] = filters
    if config.OPENALEX_MAILTO:
        base_params["mailto"] = config.OPENALEX_MAILTO

    # First page (synchronous) also tells us the total count.
    first = session.get(
        OPENALEX_WORKS_URL,
        params=dict(base_params, page=1, **{"per-page": PER_PAGE}),
        timeout=60,
    )
    first.raise_for_status()
    payload = first.json()
    total = payload["meta"]["count"]

    target = min(max_records, total, MAX_PAGEABLE)
    n_pages = math.ceil(target / PER_PAGE)

    raw: list[dict[str, Any]] = list(payload.get("results", []))

    if n_pages > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_page, session, base_params, page): page
                for page in range(2, n_pages + 1)
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"OpenAlex '{query}' ({workers} threads)"):
                try:
                    raw.extend(fut.result())
                except Exception as e:  # noqa: BLE001 - keep the run alive
                    print(f"[WARN] page {futures[fut]} failed: {e}")

    # De-duplicate by id, cap at target, normalize.
    seen: set[str] = set()
    works: list[dict[str, Any]] = []
    for w in raw:
        wid = w.get("id")
        if wid and wid not in seen:
            seen.add(wid)
            works.append(normalize_work(w))
        if len(works) >= target:
            break
    return works


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def ingest(query: str, max_records: int = 1000, workers: int | None = None) -> Path:
    """Fetch works for ``query`` and persist them to the canonical JSONL path."""
    works = fetch_works(query, max_records=max_records, workers=workers)
    out = config.works_path()
    n = write_jsonl(works, out)
    print(f"Wrote {n} works to {out}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ingest OpenAlex works")
    ap.add_argument("query", help="search query, e.g. 'CRISPR gene editing'")
    ap.add_argument("-n", "--max-records", type=int, default=1000)
    ap.add_argument("-w", "--workers", type=int, default=None)
    args = ap.parse_args()
    ingest(args.query, max_records=args.max_records, workers=args.workers)
