# 🔬 Scholar-to-Market

**Linking academic research to commercialization signals.** A data pipeline that
ingests scholarly works from **OpenAlex** (and patents from **PatentsView**),
indexes them for retrieval-augmented Q&A, and computes commercialization
analytics — which labs, companies, and technologies are moving discoveries
toward market.

Built as a tech-scouting tool for research-commercialization: *given a research
area, who is publishing, which companies are involved, what patents exist, and
what does the literature actually say?* The demo focuses on **CRISPR / gene
editing**, a canonical academia-to-startup pipeline (Editas, Intellia, Caribou,
Beam, Prime Medicine).

---

## Why this exists / what it demonstrates

| Capability | Where it lives |
| --- | --- |
| **Multithreaded ingestion** of a large public dataset | [`ingest/openalex.py`](src/scholar_to_market/ingest/openalex.py) — `ThreadPoolExecutor` fan-out over API pages with retry/backoff |
| **Working with named innovation datasets** | OpenAlex (250M+ works) + PatentsView (USPTO patents) |
| **LLM-based tooling** | Retrieval-augmented, **cited** Q&A over the corpus ([`rag.py`](src/scholar_to_market/rag.py)) |
| **Turning raw fields into decision metrics** | Commercialization-readiness score, industry involvement, citation momentum ([`analytics.py`](src/scholar_to_market/analytics.py)) |
| **Entity linkage** | Paper ↔ patent matching via semantic search |
| **Reporting / dashboards** | Streamlit dashboard + CLI report |
| **Software rigor** | `src/` package, typed code, `pytest` suite, GitHub Actions CI |

> OpenAlex also publishes a full **~400 GB snapshot** (S3); the same
> `normalize_work` logic streams snapshot partitions when you need to go past the
> API's 10k-record window — the design scales from a live slice to the full corpus.

---

## Architecture

![Scholar-to-Market architecture: OpenAlex ingestion → index → RAG/analytics → dashboard](docs/architecture.png)

**Pipeline:** `ingest` → `index` → `ask` / `report` / dashboard.

---

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/joshuatreepaik/scholar-to-market.git
cd scholar-to-market
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env      # add an LLM key for the `ask` step (see below)
```

Run the pipeline:

```bash
# 1) Ingest ~600 CRISPR works from OpenAlex (multithreaded), + sample patents
s2m ingest "CRISPR gene editing" -n 600 --patents

# 2) Embed them into the local vector store
s2m index

# 3) Commercialization report (companies, funders, paper↔patent linkage)
s2m report

# 4) Ask a cited question over the corpus
s2m ask "Which companies are developing in vivo CRISPR therapies?"
```

Or do it all from the browser — launch the dashboard and use the **Corpus**
panel to ingest any topic and rebuild the index without touching the terminal:

```bash
streamlit run src/scholar_to_market/dashboard/app.py
```

The dashboard shows headline metrics, publication trends, top industry players
and funders, the paper↔patent linkage table, and a cited RAG "Ask the corpus" box.

### Credentials

- **OpenAlex** — no key needed; set `OPENALEX_MAILTO` to join the faster "polite pool."
- **LLM** (for `ask`) — any OpenAI-compatible endpoint. Set `LLM_API_KEY`,
  `LLM_BASE_URL`, `LLM_MODEL` in `.env` (OpenAI, a university GenAI gateway, or a
  local GPT4All/Ollama server all work).
- **Patents** — optional free [PatentsView API key](https://patentsview.org/apis/keyrequest).
  Without it, the pipeline uses a small bundled sample of real CRISPR patents so
  linkage still runs.

---

## Sample output

`s2m report` on a 600-work CRISPR slice:

```
=== Top companies (industry involvement) ===
              BGI Group (China)   5
    Editas Medicine (US)          4
    Integrated DNA Technologies   3
    ToolGen (South Korea)         3
    Intellia Therapeutics (US)    2

=== Commercialization readiness ===
{ "score": 28.5,
  "components": { "recent_activity": 0.338,
                  "industry_involvement": 0.125,
                  "citation_momentum": 0.498 } }

=== Paper↔patent linkage ===
[US10266850B2] RNA-directed target DNA modification — Univ. of California
    ~0.76  [W2153344788] RNA-programmed genome editing in human cells (2013)
```

That last line is the point: a foundational **UC Berkeley patent** automatically
linked to the **Doudna lab's 2013 paper** — the academic→IP bridge, found by
semantic search.

---

## Project layout

```
scholar-to-market/
├── src/scholar_to_market/
│   ├── config.py              # env-driven settings
│   ├── ingest/
│   │   ├── openalex.py        # multithreaded works ingestion
│   │   └── patents.py         # PatentsView + sample fallback
│   ├── index.py               # embed → Chroma
│   ├── rag.py                 # retrieval + cited synthesis
│   ├── analytics.py           # metrics, readiness, paper↔patent linkage
│   ├── dashboard/app.py       # Streamlit UI
│   ├── samples/               # bundled sample patents
│   └── cli.py                 # `s2m` entry point
├── tests/                     # pytest unit tests (no network)
├── .github/workflows/ci.yml   # lint + test on every push
└── pyproject.toml
```

## Tech stack

**Python** · [OpenAlex API](https://docs.openalex.org/) ·
[PatentsView](https://patentsview.org/) · [ChromaDB](https://www.trychroma.com/) ·
[sentence-transformers](https://www.sbert.net/) · [Streamlit](https://streamlit.io/) ·
[Altair](https://altair-viz.github.io/) · pandas · OpenAI-compatible LLM client.

## Notes & limitations

- The readiness score is an **illustrative** composite, not a validated index —
  it shows how dataset fields become a decision-support metric.
- The live API path caps at OpenAlex's 10k-record paging window; larger runs use
  the snapshot (see note above).
- Paper↔patent links are semantic (embedding similarity), so treat them as
  candidate leads to verify, not ground truth.

## License

MIT — see [LICENSE](LICENSE).
