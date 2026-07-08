"""
Streamlit dashboard for Scholar-to-Market.

Run with:
    streamlit run src/scholar_to_market/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from scholar_to_market import analytics, config
from scholar_to_market.ingest import patents

st.set_page_config(page_title="Scholar-to-Market", page_icon="🔬", layout="wide")


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return analytics.load_works_df()


st.title("🔬 Scholar-to-Market")
st.caption("Linking academic research to commercialization signals — OpenAlex + patents.")

try:
    df = _load()
except FileNotFoundError:
    st.warning(f"No data yet. Run `s2m ingest \"CRISPR gene editing\" -n 600` then "
               f"`s2m index` to populate {config.works_path()}.")
    st.stop()

# --- Headline metrics ---
m = analytics.summary_metrics(df)
readiness = analytics.commercialization_readiness(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Works", f"{m['works']:,}")
c2.metric("Industry-authored", f"{m['industry_authored']:,}", f"{m['industry_share_pct']}%")
c3.metric("Total citations", f"{m['total_citations']:,}")
c4.metric("Year range", f"{m['year_range'][0]}–{m['year_range'][1]}")
c5.metric("Readiness score", readiness["score"])

with st.expander("How is the readiness score computed?"):
    st.json(readiness["components"])
    st.write("Composite = 0.4·recent_activity + 0.4·industry_involvement + 0.2·citation_momentum "
             "(illustrative, on a 0–100 scale).")

# --- Trends & players ---
left, right = st.columns(2)
with left:
    st.subheader("Publications per year")
    ppy = analytics.pubs_per_year(df)
    st.altair_chart(
        alt.Chart(ppy).mark_area(line=True, opacity=0.5).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("publications:Q", title="Publications"),
        ).properties(height=280),
        use_container_width=True,
    )
with right:
    st.subheader("Top industry players")
    tc = analytics.top_companies(df, 10)
    if tc.empty:
        st.info("No company-affiliated works in this slice.")
    else:
        st.altair_chart(
            alt.Chart(tc).mark_bar().encode(
                x=alt.X("papers:Q", title="Papers"),
                y=alt.Y("company:N", sort="-x", title=None),
                tooltip=["company", "papers"],
            ).properties(height=280),
            use_container_width=True,
        )

left2, right2 = st.columns(2)
with left2:
    st.subheader("Top institutions")
    st.dataframe(analytics.top_institutions(df, 10), use_container_width=True, hide_index=True)
with right2:
    st.subheader("Top funders")
    st.dataframe(analytics.top_funders(df, 10), use_container_width=True, hide_index=True)

# --- Paper <-> patent linkage ---
st.subheader("Paper ↔ patent linkage")
st.caption("Each patent is matched to the most semantically similar indexed papers (vector search).")
try:
    pats = patents.fetch_patents("CRISPR")
    rows = []
    for link in analytics.link_patents_to_papers(pats, k=2):
        for lp in link["linked_papers"]:
            rows.append({
                "patent": link["patent_id"],
                "assignee": link["assignee"],
                "similarity": lp["similarity"],
                "linked paper": lp["title"],
                "paper year": lp["year"],
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
except Exception as e:  # noqa: BLE001
    st.info(f"Linkage needs an index. Run `s2m index` first. ({e})")

# --- RAG ask box ---
st.subheader("Ask the corpus")
q = st.text_input("Question", "Which companies are developing in vivo CRISPR therapies?")
if st.button("Ask") and q:
    from scholar_to_market import rag
    with st.spinner("Retrieving + synthesizing…"):
        out = rag.answer(q)
    st.markdown(out["answer"])
    st.caption("Sources: " + ", ".join(f"[{s['id']}] {s['title'][:40]}" for s in out["sources"]))
