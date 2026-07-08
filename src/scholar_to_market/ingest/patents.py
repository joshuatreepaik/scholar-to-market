"""
Patent ingestion via the PatentsView Search API, with an offline fallback.

The PatentsView Search API (https://search.patentsview.org) is free but requires
an API key (request at https://patentsview.org/apis/keyrequest). When
``PATENTSVIEW_API_KEY`` is set we query it live; otherwise we fall back to a small
bundled sample of real CRISPR patents so the paper-to-patent linkage demo still
runs end-to-end with no credentials.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from .. import config

SEARCH_URL = "https://search.patentsview.org/api/v1/patent/"
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "samples" / "patents_crispr.jsonl"


def _load_sample() -> list[dict[str, Any]]:
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_api(p: dict[str, Any]) -> dict[str, Any]:
    assignees = p.get("assignees") or []
    assignee = assignees[0].get("assignee_organization") if assignees else ""
    date = p.get("patent_date") or ""
    return {
        "id": p.get("patent_id", ""),
        "title": p.get("patent_title", ""),
        "assignee": assignee or "Unknown",
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "abstract": p.get("patent_abstract", "") or "",
    }


def fetch_patents(query: str = "CRISPR", max_records: int = 100) -> list[dict[str, Any]]:
    """Fetch patents matching ``query``. Live if a key is set, else sample data."""
    if not config.PATENTSVIEW_API_KEY:
        print("[INFO] No PATENTSVIEW_API_KEY set - using bundled sample patents.")
        return _load_sample()

    headers = {"X-Api-Key": config.PATENTSVIEW_API_KEY}
    body = {
        "q": {"_text_any": {"patent_title": query}},
        "f": ["patent_id", "patent_title", "patent_date", "patent_abstract",
              "assignees.assignee_organization"],
        "o": {"size": min(max_records, 1000)},
    }
    try:
        resp = requests.post(SEARCH_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        patents = resp.json().get("patents", []) or []
        return [_normalize_api(p) for p in patents]
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] PatentsView request failed ({e}); using bundled sample.")
        return _load_sample()


def ingest(query: str = "CRISPR", max_records: int = 100) -> Path:
    patents = fetch_patents(query, max_records=max_records)
    out = config.patents_path()
    with open(out, "w", encoding="utf-8") as f:
        for p in patents:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(patents)} patents to {out}")
    return out


if __name__ == "__main__":
    ingest()
