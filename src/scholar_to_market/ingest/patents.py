"""
Patent ingestion, with three keyless-or-keyed sources tried in priority order.

USPTO retired the legacy PatentsView *Search* API during its Open Data Portal
(ODP) migration. The sources this module supports now:

  1. **USPTO ODP search API** (live, any topic) — set ``USPTO_ODP_API_KEY``.
     A free key from https://data.uspto.gov/apis/getting-started queries
     ``POST https://api.uspto.gov/api/v1/patent/applications/search`` by invention
     title and returns granted patents matching the loaded topic.
  2. **Bulk TSV** (keyless, any topic) — set ``PATENTSVIEW_BULK_TSV`` to a
     downloaded ``g_patent.tsv``; it is streamed line-by-line (scales to GBs).
  3. **Curated reference set** (default) — a bundled file of real granted patents
     spanning the trending topics, so linkage runs with no setup or credentials.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import requests

from .. import config

REFERENCE_FILE = Path(__file__).resolve().parent.parent / "samples" / "patents_reference.jsonl"
ODP_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"

# Bulk patent abstracts can be long; raise the TSV field-size limit.
csv.field_size_limit(10_000_000)


def active_source() -> tuple[str, str]:
    """Which patent source will be used, as (key, human-readable label)."""
    if config.USPTO_ODP_API_KEY:
        return "odp", "USPTO Open Data Portal (live)"
    bulk = config.PATENTSVIEW_BULK_TSV
    if bulk and Path(bulk).exists():
        return "bulk", f"PatentsView bulk file ({Path(bulk).name})"
    return "reference", "curated multi-topic reference patents"


# --------------------------------------------------------------------------- #
# 1) USPTO Open Data Portal (live search)
# --------------------------------------------------------------------------- #
def _odp_query(query: str) -> str:
    """Build an ODP query string matching the topic against invention titles."""
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 2][:4]
    if not tokens:
        return f"applicationMetaData.inventionTitle:{query}"
    return " AND ".join(f"applicationMetaData.inventionTitle:{t}" for t in tokens)


def fetch_from_odp(query: str, max_records: int, api_key: str) -> list[dict[str, Any]]:
    """Query the USPTO ODP search API and return granted patents on the topic."""
    body = {
        "q": _odp_query(query),
        "pagination": {"offset": 0, "limit": min(max(max_records * 3, 25), 100)},
    }
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = requests.post(ODP_SEARCH_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    bag = resp.json().get("patentFileWrapperDataBag", []) or []

    out: list[dict[str, Any]] = []
    for item in bag:
        md = item.get("applicationMetaData", {}) or {}
        pnum = md.get("patentNumber")
        if not pnum:  # keep granted patents only (applications have no number yet)
            continue
        grant = md.get("grantDate", "") or ""
        applicants = md.get("applicantBag") or []
        assignee = md.get("firstApplicantName") or (
            applicants[0].get("applicantNameText") if applicants else ""
        )
        out.append({
            "id": f"US{pnum}",
            "title": md.get("inventionTitle", "") or "",
            "assignee": assignee or "Unknown",
            "year": int(grant[:4]) if grant[:4].isdigit() else None,
            "abstract": "",  # not returned by the file-wrapper search
        })
        if len(out) >= max_records:
            break
    return out


# --------------------------------------------------------------------------- #
# 2) Bulk TSV (streamed)
# --------------------------------------------------------------------------- #
def _pick(row: dict[str, str], *names: str) -> str:
    for n in names:
        if row.get(n):
            return row[n].strip()
    return ""


def _normalize_row(row: dict[str, str]) -> dict[str, Any]:
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
    """Stream a bulk patent TSV and keep rows whose text matches ``query``."""
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


# --------------------------------------------------------------------------- #
# 3) Curated multi-topic reference set (bundled)
# --------------------------------------------------------------------------- #
def _load_reference() -> list[dict[str, Any]]:
    """Real granted patents spanning the trending topics, for a keyless demo."""
    with open(REFERENCE_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_patents(query: str = "CRISPR", max_records: int = 100) -> list[dict[str, Any]]:
    """Fetch patents matching ``query`` from the best available source."""
    source, _ = active_source()
    if source == "odp":
        try:
            recs = fetch_from_odp(query, max_records, config.USPTO_ODP_API_KEY)
            if recs:
                return recs
            print("[WARN] ODP returned no granted patents; falling back to reference set.")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] ODP request failed ({e}); falling back to reference set.")
    elif source == "bulk":
        recs = load_from_bulk(config.PATENTSVIEW_BULK_TSV, query, max_records=max_records)
        if recs:
            return recs
        print("[WARN] No matches in bulk file; falling back to reference set.")
    else:
        print("[INFO] Using curated multi-topic reference patents "
              "(set USPTO_ODP_API_KEY for full live coverage).")
    return _load_reference()


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
