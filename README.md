# webdev-leads

**Multi-agent discovery of high-rated local businesses that have no website.**

This system finds roofing contractors (or any niche) in a target US state, verifies whether they operate their own website, scores the lead quality, and exports qualified prospects for a web-design sales team.

## Architecture

```
planner ──▶ discovery ──▶ verifier ──▶ qualifier ──▶ critic ──▶ reporter ──▶ export
```

Each agent has a **deterministic fallback** so the pipeline degrades gracefully when the LLM is unavailable or hits its budget. The LLM is used only for small, bounded decisions (boolean verdicts, fit scores, audits) — never for bulk data extraction.

### Agents

| Agent | Job | LLM Use |
|-------|-----|---------|
| **Planner** | Turns campaign brief into search queries & ICP | Proposes queries; deterministic default if unavailable |
| **Discovery** | Finds candidate businesses per ZIP/city via OSM + Tavily | Fallback extractor for rich pages that rules miss |
| **Verifier** | Decides: does this business own a website? | Adjudicates conflicting evidence only |
| **Qualifier** | Gates on rating/reviews, scores fit (0–100), assigns tier | Scores qualitative fit when available |
| **Critic** | Independent audit before export | Flags wrong business kind, fabricated data, contradictions |
| **Reporter** | Writes the run brief | Prose summary; template fallback |

### Data Sources (no Google Maps API needed)

1. **OpenStreetMap / Overpass** — Free, structured `craft=roofer` nodes. Bounding-box queries around each ZIP. No API key.
2. **Tavily Search + Extract** — Finds directory/list pages, pulls full page text. 1 credit/search, 1 credit/5 extracts. Free tier = 1,000 credits/month.
3. **Direct HTTP probes** — Guesses likely domains (`acmeroofing.com`) and probes them. Zero API cost, resolves ~60% of established businesses.

The system **does not use Google Maps Places, ScraperAPI, or Apify**. Those are the legacy implementations in the repo root (kept for reference).

## Quick Start

```bash
# 1. Clone & install
cd webdev
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Copy .env.example → .env and add your keys
cp .env.example .env
#   OPENROUTER_API_KEY=sk-or-v1-...
#   MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
#   TAVILY_API_KEY=tvly-...

# 3. Validate
python -m leads doctor --network

# 4. Preview a plan (no network calls)
python -m leads plan --state TX --zips 75201,75204 --max-zips 2

# 5. Run
python -m leads run --state TX --zips 75201,75204,75751 --max-zips 3
```

### Resume a Run

```bash
# Pass the run_id to resume from checkpoints (skips completed cells)
python -m leads run --state TX --zips 75201,75204,75751 --max-zips 3 \
  --run-id run-20261029-174159 --resume
```

### Export Formats

```bash
python -m leads export --run-id run-20261029-174159 --format xlsx csv
# Also: jsonl
```

## Configuration

All tunables live in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LEADS_CONCURRENCY` | 4 | Parallel workers for each stage |
| `LEADS_MAX_LLM_CALLS` | 500 | Hard cap on LLM calls per run |
| `LEADS_MAX_TAVILY_CREDITS` | 900 | Leave headroom under free 1,000/month |
| `LEADS_HTTP_CACHE_TTL_HOURS` | 168 | Cache Tavily/Overpass responses |
| `LEADS_LOG_LEVEL` | INFO | DEBUG for full tracing |

## Targeting Strategy

**Metro listicles** ("best roofers in Dallas") only contain companies that **already have websites**. They are useless for this product.

**Small-town / county searches** ("roofing contractor Athens TX 75751") surface the long tail of established, high-rated contractors who never built a site — exactly the ICP. The CLI accepts explicit ZIP codes so you can hand-pick the right areas.

## Output

Each run produces:

```
var/runs/run-<timestamp>/
├── manifest.json       # Complete run metadata
├── report.md           # Human-readable brief
├── leads_<state>.xlsx  # Sales-ready spreadsheet
└── leads_<state>.csv   # Machine-readable
```

The spreadsheet contains only `qualified` leads (optionally `needs_review`), with columns: Name, Phone, City, State, ZIP, Address, Rating, Reviews, Tier, Fit Score, Has Website, Verdict Method, Source URL.

## Project Layout

```
leads/
├── agents/              # 6 specialist agents + base class
├── llm/                 # OpenRouter client + JSON repair + prompts
├── tools/               # Tavily, Overpass, WebProbe, cache, rate-limit, extraction
├── models.py            # Pydantic domain models (Campaign, Lead, Candidate, etc.)
├── store.py             # SQLite persistence (WAL, checkpoints, dedupe)
├── orchestrator.py      # Supervisor: ordering, concurrency, budgets, resume
├── export.py            # DataFrame → xlsx/csv/jsonl
├── zipcodes.py          # ZIP → city/state lookup (CSV + prefix fallback)
├── config.py            # Settings from .env
├── errors.py            # Typed exception hierarchy
└── cli.py               # Typer-free argparse commands
```

## Why This Design?

- **Deterministic first, LLM second.** Extraction is rule-based; the model only judges.
- **Small prompts.** Free reasoning models truncate long JSON; every prompt asks for one decision.
- **Checkpointed.** Every cell × stage is a DB row. `--resume` pays for nothing twice.
- **Budgeted.** LLM calls and Tavily credits have hard caps checked before each request.
- **Auditable.** Every lead keeps its source URL and verbatim evidence. Rejections keep reasons.
- **No vendor lock-in.** Swap Tavily for Serper, or the LLM for any OpenAI-compatible endpoint.

## Legacy Code (Reference Only)

The repo root contains three earlier scraper implementations:

- `b-list.py` — Google Maps Python client prototype
- `scraper-pipeline.py` / `statewise-scraper-pipeline.py` — Apify-based batch scrapers
- `b-list1.py` — ScraperAPI SERP-based version with fallback logic

These are **not used** by the agent pipeline and are kept only for reference.

## License

Proprietary — internal use only.