"""Unit tests for the offline, deterministic parts of the pipeline (no network)."""
import pandas as pd

from scholar_to_market.ingest import openalex, patents
from scholar_to_market import analytics


def test_reconstruct_abstract_orders_words():
    inv = {"Gene": [0], "editing": [1], "works": [2]}
    assert openalex._reconstruct_abstract(inv) == "Gene editing works"


def test_reconstruct_abstract_handles_empty():
    assert openalex._reconstruct_abstract(None) == ""
    assert openalex._reconstruct_abstract({}) == ""


def test_normalize_work_extracts_company_and_topic():
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Editing genes",
        "publication_year": 2021,
        "cited_by_count": 42,
        "authorships": [
            {"author": {"display_name": "A. Researcher"},
             "institutions": [{"display_name": "Editas Medicine", "type": "company"}]},
            {"author": {"display_name": "B. Scholar"},
             "institutions": [{"display_name": "MIT", "type": "education"}]},
        ],
        "primary_topic": {"display_name": "Gene Editing"},
        "funders": [{"display_name": "NIH"}],
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
    }
    w = openalex.normalize_work(raw)
    assert w["companies"] == ["Editas Medicine"]
    assert w["topic"] == "Gene Editing"
    assert w["funders"] == ["NIH"]
    assert w["abstract"] == "Hello world"
    assert {"name": "MIT", "type": "education"} in w["institutions"]


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "W1", "title": "a", "abstract": "x", "year": 2024, "cited_by_count": 100,
         "companies": ["Intellia Therapeutics"], "institutions": [{"name": "MIT", "type": "education"}],
         "funders": ["NIH"], "topic": "Gene Editing"},
        {"id": "W2", "title": "b", "abstract": "", "year": 2020, "cited_by_count": 5,
         "companies": [], "institutions": [{"name": "MIT", "type": "education"}],
         "funders": ["NSF"], "topic": "Gene Editing"},
        {"id": "W3", "title": "c", "abstract": "y", "year": 2015, "cited_by_count": 300,
         "companies": ["Editas Medicine"], "institutions": [{"name": "Broad", "type": "facility"}],
         "funders": ["NIH"], "topic": "Diagnostics"},
    ])


def test_summary_metrics():
    m = analytics.summary_metrics(_fixture_df())
    assert m["works"] == 3
    assert m["industry_authored"] == 2
    assert m["industry_share_pct"] == 66.7
    assert m["year_range"] == (2015, 2024)


def test_top_companies_and_institutions():
    df = _fixture_df()
    comps = analytics.top_companies(df)
    assert set(comps["company"]) == {"Intellia Therapeutics", "Editas Medicine"}
    insts = analytics.top_institutions(df)
    assert insts.iloc[0]["institution"] == "MIT"  # appears twice


def test_readiness_score_in_range():
    r = analytics.commercialization_readiness(_fixture_df())
    assert 0 <= r["score"] <= 100
    assert set(r["components"]) == {"recent_activity", "industry_involvement", "citation_momentum"}


def test_patents_sample_fallback_loads():
    # With no bulk file configured, fetch_patents returns the bundled sample.
    pats = patents.fetch_patents("CRISPR")
    assert len(pats) >= 5
    assert all({"id", "title", "assignee"} <= set(p) for p in pats)


def test_active_source_defaults_to_sample():
    key, label = patents.active_source()
    assert key == "sample"
    assert "sample" in label.lower()


def test_active_source_prefers_odp_key(monkeypatch):
    monkeypatch.setattr(patents.config, "USPTO_ODP_API_KEY", "test-key")
    key, _ = patents.active_source()
    assert key == "odp"


def test_odp_query_builds_title_clause():
    q = patents._odp_query("GLP-1 obesity drugs")
    assert "applicationMetaData.inventionTitle:obesity" in q
    assert " AND " in q                       # multiple tokens AND-ed
    assert "GLP" in q or "obesity" in q


def test_load_from_bulk_filters_and_normalizes(tmp_path):
    tsv = tmp_path / "g_patent.tsv"
    tsv.write_text(
        "patent_id\tpatent_title\tpatent_date\tpatent_abstract\tassignee\n"
        "111\tGLP-1 receptor agonist formulation\t2021-05-01\tA peptide for obesity.\tNovo Nordisk\n"
        "222\tWidget for gardening\t2019-03-02\tUnrelated tool.\tAcme\n"
        "333\tGLP-1 analog delivery device\t2023-01-10\tInjection pen.\tEli Lilly\n",
        encoding="utf-8",
    )
    recs = patents.load_from_bulk(str(tsv), "GLP-1 obesity", max_records=10)
    ids = {r["id"] for r in recs}
    assert ids == {"111", "333"}          # gardening widget filtered out
    r = next(r for r in recs if r["id"] == "111")
    assert r["assignee"] == "Novo Nordisk"
    assert r["year"] == 2021
