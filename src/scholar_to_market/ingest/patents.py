"""
Patent ingestion.

The legacy PatentsView *Search* API (search.patentsview.org/api) is being retired
as USPTO migrates to the Open Data Portal (ODP). Previously-issued keys won't
carry over and there is no firm relaunch date for the hosted search endpoints.
Patent **bulk datasets**, however, remain freely downloadable with **no key**
(via ODP's Bulk Datasets API / PatentsView downloads).

This module therefore supports two keyless sources, in priority order:

  1. A downloaded PatentsView/ODP **bulk TSV** — set ``PATENTSVIEW_BULK_TSV`` to
     the file path. It is streamed line-by-line, so it scales to the full
     multi-GB ``g_patent.tsv`` without loading it into memory.
  2. A small **bundled sample** of real CRISPR patents (default) so the
     paper-to-patent linkage runs with no downloads or credentials.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .. import config

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "samples" / "patents_crispr.jsonl"

# Increase the TSV field-size limit; bulk patent abstracts can be long.
csv.field_size_limit(10_000_000)


def active_source() -> tuple[str, str]:
    """Which patent source will be used, as (key, human-readable label)."""
    bulk = config.PATENTSVIEW_BULK_TSV
    if bulk and Path(bulk).exists():
        return "bulk", f"PatentsView bulk file ({Path(bulk).name})"
    return "sample", "bundled CRISPR sample patents"


def _load_sample() -> list[dict[str, Any]]:
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pick(row: dict[str, str], *names: str) -> str:
    for n in names:
        if row.get(n):
            return row[n].strip()
    return ""


def _normalize_row(row: dict[str, str]) -> dict[str, Any]:
    """Map a bulk-TSV row to our patent schema (column names vary by export)."""
    date = _pick(row, "patent_date", "date")
    return {
        "id": _pick(row, "patent_id", "id", "patent_number"),
        "title": _pick(row, "patent_title", "title"),
        "assignee": _pick(row, "assignee", "assignee_organization",
                          "disambig_assignee_organization") or "Unknown",
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "abstract": _pick(row, "patent_abstract", "abstract"),
    }


def load_from_bulk(path: str, query: str, max_records: int = 100,
                   max_scan: int = 3_000_000) -> list[dict[str, Any]]:
    """Stream a bulk patent TSV and keep rows matching ``query``.

    Matches rows whose title/abstract contains any meaningful query token.
    Scans at most ``max_scan`` rows so a run against the full file stays bounded.
    """
    tokens = [t.lower() for t in query.split() if len(t) > 3]
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= max_scan:
                break
            rec = _normalize_row(row)
            if not rec["id"] or not rec["title"]:
                continue
            haystack = f"{rec['title']} {rec['abstract']}".lower()
            if not tokens or any(t in haystack for t in tokens):
                out.append(rec)
                if len(out) >= max_records:
                    break
    return out


def fetch_patents(query: str = "CRISPR", max_records: int = 100) -> list[dict[str, Any]]:
    """Fetch patents matching ``query`` from the best available keyless source."""
    source, _ = active_source()
    if source == "bulk":
        recs = load_from_bulk(config.PATENTSVIEW_BULK_TSV, query, max_records=max_records)
        if recs:
            return recs
        print("[WARN] No matches in bulk file; falling back to bundled sample.")
    else:
        print("[INFO] Using bundled CRISPR sample patents "
              "(set PATENTSVIEW_BULK_TSV for topic-matched patents).")
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
