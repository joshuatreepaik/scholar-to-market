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

from scholar_to_market import analytics, config, index
from scholar_to_market.ingest import openalex, patents

st.set_page_config(page_title="Scholar-to-Market", layout="wide")

# --- Visual design -----------------------------------------------------------
INK = "#222b36"       # primary text
MUTED = "#6b7683"     # secondary text
ACCENT = "#33566e"    # single restrained accent (steel navy)
RULE = "#e4e7ea"      # hairline borders

CSS = f"""
<style>
  /* Constrain width and calm the default Streamlit chrome. */
  header[data-testid="stHeader"] {{ background: transparent; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ max-width: 1160px; padding-top: 2.4rem; padding-bottom: 4rem; }}

  html, body, [class*="css"] {{ color: {INK}; }}

  /* Masthead */
  .s2m-kicker {{
    font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: {MUTED}; font-weight: 600; margin-bottom: 0.35rem;
  }}
  .s2m-title {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.35rem; line-height: 1.1; font-weight: 600; color: {INK};
    margin: 0 0 0.4rem 0;
  }}
  .s2m-sub {{ color: {MUTED}; font-size: 1rem; max-width: 60ch; margin: 0; }}
  .s2m-rule {{ border: none; border-top: 1px solid {RULE}; margin: 1.6rem 0 1.9rem; }}

  /* Section eyebrows */
  .s2m-section {{ margin: 2.1rem 0 0.9rem; }}
  .s2m-section .num {{
    font-variant-numeric: tabular-nums; color: {ACCENT}; font-weight: 700;
    font-size: 0.8rem; margin-right: 0.5rem;
  }}
  .s2m-section .label {{
    font-size: 0.95rem; font-weight: 600; letter-spacing: 0.01em; color: {INK};
  }}
  .s2m-section .desc {{ color: {MUTED}; font-size: 0.82rem; margin-top: 0.15rem; }}

  /* Metric cards: hairline, monochrome, no default color arrows. */
  [data-testid="stMetric"] {{
    background: #fff; border: 1px solid {RULE}; border-radius: 6px;
    padding: 0.9rem 1rem 0.8rem;
  }}
  [data-testid="stMetricLabel"] p {{
    font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase;
    color: {MUTED} !important; font-weight: 600;
  }}
  [data-testid="stMetricValue"] {{
    font-size: 1.55rem; font-weight: 600; color: {INK};
    font-variant-numeric: tabular-nums;
  }}
  [data-testid="stMetricDelta"] {{ color: {MUTED} !important; }}

  /* Buttons / inputs */
  .stButton > button {{
    background: {ACCENT}; color: #fff; border: none; border-radius: 5px;
    font-weight: 600; padding: 0.4rem 1.1rem;
  }}
  .stButton > button:hover {{ background: #294353; color: #fff; }}
  [data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 6px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def section(num: str, label: str, desc: str = "") -> None:
    st.markdown(
        f'<div class="s2m-section"><span class="num">{num}</span>'
        f'<span class="label">{label}</span>'
        + (f'<div class="desc">{desc}</div>' if desc else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def _bar(df: pd.DataFrame, value: str, label: str, height: int = 300):
    return (
        alt.Chart(df)
        .mark_bar(color=ACCENT, size=16)
        .encode(
            x=alt.X(f"{value}:Q", title=None),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            tooltip=[label, value],
        )
        .properties(height=height)
        .configure_axis(grid=False, labelColor=MUTED, titleColor=MUTED,
                        domainColor=RULE, tickColor=RULE, labelFontSize=12)
        .configure_view(strokeWidth=0)
    )


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return analytics.load_works_df()


# --- Sidebar: build / switch the corpus without leaving the browser ---------
with st.sidebar:
    st.markdown("#### Corpus")
    st.caption("Fetch a new research area from OpenAlex and rebuild the index.")
    query = st.text_input("Topic or query", st.session_state.get("query", "CRISPR gene editing"))
    n = st.slider("Max works", min_value=100, max_value=2000, value=600, step=100)
    if st.button("Ingest & reindex", use_container_width=True):
        try:
            with st.spinner(f"Ingesting “{query}” from OpenAlex…"):
                openalex.ingest(query, max_records=n)
            with st.spinner("Embedding + building the vector index…"):
                index.build_index()
            st.session_state["query"] = query
            _load.clear()
            st.success("Corpus rebuilt.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Ingest failed: {e}")
    st.divider()
    st.caption("Patent linkage uses the PatentsView API when `PATENTSVIEW_API_KEY` "
               "is set, otherwise a bundled CRISPR sample.")


# --- Masthead ---------------------------------------------------------------
st.markdown(
    '<div class="s2m-kicker">Research Commercialization Analytics</div>'
    '<div class="s2m-title">Scholar-to-Market</div>'
    '<p class="s2m-sub">Linking academic research to commercialization signals '
    'across OpenAlex publications and patents.</p>'
    '<hr class="s2m-rule">',
    unsafe_allow_html=True,
)

try:
    df = _load()
except FileNotFoundError:
    st.warning(f'No data yet. Run  `s2m ingest "CRISPR gene editing" -n 600`  then  '
               f'`s2m index`  to populate {config.works_path()}.')
    st.stop()

# What's currently loaded, so questions/linkage are read in the right context.
_topics = analytics.top_topics(df, 1)
LOADED_TOPIC = _topics.iloc[0]["topic"] if not _topics.empty else "research"
st.markdown(
    f'<p style="color:{MUTED};font-size:0.85rem;margin:-0.6rem 0 0.4rem;">'
    f'Loaded corpus: <b style="color:{INK}">{len(df):,} works</b> · '
    f'dominant topic <b style="color:{INK}">{LOADED_TOPIC}</b></p>',
    unsafe_allow_html=True,
)

# --- Overview ---------------------------------------------------------------
m = analytics.summary_metrics(df)
readiness = analytics.commercialization_readiness(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Works", f"{m['works']:,}")
c2.metric("Industry-authored", f"{m['industry_authored']:,}",
          f"{m['industry_share_pct']}% of corpus", delta_color="off")
c3.metric("Total citations", f"{m['total_citations']:,}")
c4.metric("Years", f"{m['year_range'][0]}–{m['year_range'][1]}")
c5.metric("Readiness", readiness["score"])

with st.expander("How the readiness score is derived"):
    comp = readiness["components"]
    st.markdown(
        f"A transparent 0–100 composite: "
        f"**0.4 × recent activity** ({comp['recent_activity']}) + "
        f"**0.4 × industry involvement** ({comp['industry_involvement']}) + "
        f"**0.2 × citation momentum** ({comp['citation_momentum']}). "
        f"Illustrative — it shows how raw dataset fields become a decision metric, "
        f"not a validated index."
    )

# --- Trends & players -------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    section("01", "Publication volume by year", "Is the field growing?")
    ppy = analytics.pubs_per_year(df)
    area = (
        alt.Chart(ppy)
        .mark_area(color=ACCENT, opacity=0.12, line={"color": ACCENT, "strokeWidth": 2})
        .encode(x=alt.X("year:O", title=None), y=alt.Y("publications:Q", title=None))
        .properties(height=300)
        .configure_axis(grid=False, labelColor=MUTED, domainColor=RULE, tickColor=RULE,
                        labelFontSize=12)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(area, use_container_width=True)
with right:
    section("02", "Industry involvement", "Which companies co-author this research?")
    tc = analytics.top_companies(df, 10)
    if tc.empty:
        st.info("No company-affiliated works in this slice.")
    else:
        st.altair_chart(_bar(tc, "papers", "company"), use_container_width=True)

left2, right2 = st.columns(2, gap="large")
with left2:
    section("03", "Leading institutions")
    st.dataframe(analytics.top_institutions(df, 10), use_container_width=True, hide_index=True)
with right2:
    section("04", "Funding sources")
    st.dataframe(analytics.top_funders(df, 10), use_container_width=True, hide_index=True)

# --- Linkage ----------------------------------------------------------------
section("05", "Research-to-patent linkage",
        "Each patent matched to its nearest papers by embedding similarity.")
try:
    _pquery = st.session_state.get("query", "CRISPR").split()[0]
    pats = patents.fetch_patents(_pquery)
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

# --- Ask --------------------------------------------------------------------
section("06", "Ask the corpus",
        f"Retrieval-augmented answers with inline citations, grounded in the "
        f"{len(df):,} loaded works.")
q = st.text_input(
    "Question",
    "What are the leading commercial applications in this research area, "
    "and which companies are involved?",
    label_visibility="collapsed",
)
if st.button("Ask") and q:
    from scholar_to_market import rag
    with st.spinner("Retrieving and synthesizing…"):
        out = rag.answer(q)
    st.markdown(out["answer"])
    st.caption("Sources: " + ", ".join(f"[{s['id']}] {s['title'][:40]}" for s in out["sources"]))
