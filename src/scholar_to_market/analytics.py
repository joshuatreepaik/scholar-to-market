"""
Commercialization analytics over ingested works, plus paper->patent linkage.

The signal we care about: which research areas / institutions / companies show
momentum toward commercialization. We proxy that from OpenAlex fields:
  - publication volume & recency (is the field growing?)
  - citation intensity (is the work influential?)
  - industry involvement (are *companies* co-authoring?)
  - patent linkage (does adjacent IP exist?)
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from . import index


def load_works_df(path=None) -> pd.DataFrame:
    return pd.DataFrame(index.load_works(path))


def pubs_per_year(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.dropna(subset=["year"])
             .groupby("year")
             .size()
             .reset_index(name="publications")
             .sort_values("year"))
    out["year"] = out["year"].astype(int)
    return out


def _explode_counter(df: pd.DataFrame, col: str, key: str | None = None) -> Counter:
    c: Counter = Counter()
    for val in df[col]:
        for item in val or []:
            name = item.get(key) if (key and isinstance(item, dict)) else item
            if name:
                c[name] += 1
    return c


def top_institutions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    c = _explode_counter(df, "institutions", key="name")
    return pd.DataFrame(c.most_common(n), columns=["institution", "papers"])


def top_companies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    c = _explode_counter(df, "companies")
    return pd.DataFrame(c.most_common(n), columns=["company", "papers"])


def top_funders(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    c = _explode_counter(df, "funders")
    return pd.DataFrame(c.most_common(n), columns=["funder", "papers"])


def top_topics(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    c = Counter(t for t in df["topic"] if t)
    return pd.DataFrame(c.most_common(n), columns=["topic", "papers"])


def summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    industry = int(df["companies"].apply(lambda x: bool(x)).sum())
    years = df["year"].dropna()
    return {
        "works": n,
        "with_abstract": int((df["abstract"].str.len() > 0).sum()) if n else 0,
        "industry_authored": industry,
        "industry_share_pct": round(100 * industry / n, 1) if n else 0.0,
        "total_citations": int(df["cited_by_count"].sum()) if n else 0,
        "median_citations": float(df["cited_by_count"].median()) if n else 0.0,
        "year_range": (int(years.min()), int(years.max())) if len(years) else (None, None),
    }


def commercialization_readiness(df: pd.DataFrame, recent_years: int = 5) -> dict[str, Any]:
    """A transparent 0-100 score blending recency, industry, and citation signals.

    Not a scientific index -- an illustrative composite that shows how one would
    turn raw dataset fields into a decision-support metric.
    """
    if df.empty:
        return {"score": 0.0, "components": {}}

    years = df["year"].dropna()
    latest = int(years.max()) if len(years) else 0
    recent_share = float((years >= latest - recent_years).mean()) if len(years) else 0.0
    industry_share = float(df["companies"].apply(bool).mean())
    # Citation momentum: fraction of works above the corpus median.
    med = df["cited_by_count"].median()
    high_impact_share = float((df["cited_by_count"] > med).mean())

    components = {
        "recent_activity": round(recent_share, 3),
        "industry_involvement": round(industry_share, 3),
        "citation_momentum": round(high_impact_share, 3),
    }
    score = round(100 * (0.4 * recent_share + 0.4 * industry_share + 0.2 * high_impact_share), 1)
    return {"score": score, "components": components}


def link_patents_to_papers(patents: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    """For each patent, find the most semantically similar indexed papers.

    Reuses the RAG vector store: we embed each patent's title+abstract and query
    the works collection. This surfaces the research literature adjacent to a
    piece of IP -- the paper<->patent bridge at the heart of tech scouting.
    """
    collection = index.get_collection()
    links: list[dict[str, Any]] = []
    for p in patents:
        query = f"{p.get('title', '')} {p.get('abstract', '')}".strip()
        if not query:
            continue
        res = collection.query(query_texts=[query], n_results=k)
        papers = []
        if res and res.get("ids") and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                meta = res["metadatas"][0][i]
                dist = res["distances"][0][i] if res.get("distances") else None
                papers.append({
                    "paper_id": res["ids"][0][i],
                    "title": meta.get("title", ""),
                    "year": meta.get("year"),
                    "similarity": round(1 - dist, 3) if dist is not None else None,
                })
        links.append({
            "patent_id": p.get("id"),
            "patent_title": p.get("title"),
            "assignee": p.get("assignee"),
            "linked_papers": papers,
        })
    return links
