"""Command-line entry point: ingest -> index -> ask / report."""
from __future__ import annotations

import argparse
import json

from . import analytics, config, rag
from .ingest import openalex, patents


def _corpus_query() -> str | None:
    """The search query used to build the current corpus (recorded at ingest)."""
    p = config.corpus_meta_path()
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8")).get("query")
        except Exception:  # noqa: BLE001
            return None
    return None


def cmd_ingest(args: argparse.Namespace) -> None:
    openalex.ingest(args.query, max_records=args.max_records, workers=args.workers)
    if args.patents:
        patents.ingest(args.query)


def cmd_index(args: argparse.Namespace) -> None:
    from . import index
    index.build_index()


def cmd_ask(args: argparse.Namespace) -> None:
    out = rag.answer(" ".join(args.question), k=args.k)
    print(out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print(f"  [{s['id']}] {s['title'][:70]} ({s['year']})")


def cmd_report(args: argparse.Namespace) -> None:
    df = analytics.load_works_df()
    print("=== Summary ===")
    print(json.dumps(analytics.summary_metrics(df), indent=2))
    print("\n=== Commercialization readiness ===")
    print(json.dumps(analytics.commercialization_readiness(df), indent=2))
    print("\n=== Top companies (industry involvement) ===")
    print(analytics.top_companies(df).to_string(index=False))
    print("\n=== Top institutions ===")
    print(analytics.top_institutions(df).to_string(index=False))
    print("\n=== Paper<->patent linkage ===")
    query = args.query or _corpus_query() or "CRISPR"
    pats = patents.fetch_patents(query)
    for link in analytics.link_patents_to_papers(pats, k=2):
        print(f"\n[{link['patent_id']}] {link['patent_title'][:60]} - {link['assignee']}")
        for lp in link["linked_papers"]:
            print(f"    ~ {lp['similarity']}  [{lp['paper_id']}] {lp['title'][:55]} ({lp['year']})")


def main() -> None:
    ap = argparse.ArgumentParser(prog="s2m", description="Scholar-to-Market pipeline")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("ingest", help="fetch works (and patents) from the APIs")
    p.add_argument("query")
    p.add_argument("-n", "--max-records", type=int, default=1000)
    p.add_argument("-w", "--workers", type=int, default=None)
    p.add_argument("--patents", action="store_true", help="also ingest patents")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("index", help="embed works into the vector store")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask a cited question over the corpus")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("report", help="print commercialization analytics")
    p.add_argument("--query", default=None,
                   help="patent query for linkage (defaults to the loaded corpus's topic)")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
