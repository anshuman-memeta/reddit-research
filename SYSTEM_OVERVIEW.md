# Reddit Research Bot — System Overview

A Telegram bot that performs automated brand-mention research across Reddit. Users send `/research <brand>` and receive a comprehensive analysis: fetched posts, LLM-powered sentiment/relevance classification, charts, and a Google Sheet export — all delivered back via Telegram.

---

## Architecture

```
User sends /research <brand> via Telegram
        │
        ▼
┌─────────────────────────────────────────┐
│  FETCH: MultiSourceFetcher              │
│  4 sources with automatic fallback:     │
│  Arctic Shift → Reddit JSON → RSS →    │
│  Pullpush                               │
│  Deduplicates posts by ID              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  ANALYZE: BrandAnalyzer                 │
│  LLM batch processing (10 posts/call)   │
│  Provider chain: Groq → Cerebras →      │
│  SambaNova → Mistral                    │
│  Keyword fallback if all rate-limited   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  VISUALIZE: 3 matplotlib charts         │
│  Sentiment donut · Subreddit dist ·     │
│  Weekly trend                           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  EXPORT: Google Sheets (CSV fallback)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  DELIVER: Summary + charts + sheet link │
│  sent back to user on Telegram          │
└─────────────────────────────────────────┘
```

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `bot.py` | ~615 | Telegram bot interface, command handlers, pipeline orchestration |
| `fetcher.py` | ~823 | Multi-source Reddit post fetching with dedup, retry, and rate-limiting |
| `analyzer.py` | ~448 | LLM-powered batch relevance/sentiment analysis with provider fallback |
| `charts.py` | ~280 | Matplotlib chart generation (sentiment, subreddits, trend) |
| `sheets.py` | ~87 | Google Sheets export with quota retry; CSV fallback |
| `config.py` | ~79 | Environment variable loading and constants |
| `brands.json` | ~176 | Pre-configured brand definitions (6 brands) |

---

## Bot Commands

| Command | Description |
|---------|-------------|
| `/research <brand>` | Run a full 3-month deep dive on a configured brand |
| `/research_add` | Interactive 7-step wizard to add a new brand |
| `/research_delete <brand>` | Delete a brand (with confirmation) |
| `/research_list` | List all configured brands |
| `/start`, `/help` | Show usage instructions |

Only authorized Telegram user IDs (configured in `.env`) can run commands.

---

## Data Pipeline Details

### 1. Fetching (`fetcher.py`)

The `MultiSourceFetcher` searches Reddit for brand mentions over a 90-day lookback window. It uses 4 sources in priority order:

| Source | Class | Notes |
|--------|-------|-------|
| Arctic Shift API | `ArcticShiftFetcher` | Primary; reliable archive with full-text search |
| Reddit JSON | `RedditSearchFetcher` | Official endpoint; often 403-blocked from VPS IPs |
| Reddit RSS | `RedditRSSFetcher` | Limited (~25 results) but bypasses some blocks |
| Pullpush.io | `PullpushFetcher` | Unreliable fallback |

For each source, the fetcher iterates over brand-specific subreddits (from `subreddit_hints` + 18 default Indian consumer/tech subreddits) and searches each keyword. Posts are deduplicated by ID across all sources.

**Resilience**: Tracks consecutive failures per source; skips a source after 5 consecutive subreddit failures. Returns partial results rather than raising errors.

### 2. Analysis (`analyzer.py`)

The `BrandAnalyzer` processes posts in batches of 10 using LLM calls. Each post is classified with:

- **Relevant** (bool): Is this actually about the brand?
- **Sentiment**: positive / negative / neutral
- **Theme**: Short descriptive label
- **Summary**: One-line summary
- **Competitor mentions**: List of competitor names found

The LLM provider chain tries up to 4 providers sequentially:

1. Groq (`llama-3.1-8b-instant`)
2. Cerebras (`llama-3.3-70b`)
3. SambaNova (`Meta-Llama-3.3-70B`)
4. Mistral (`mistral-small-latest`)

If all LLM providers are rate-limited, a **keyword fallback** heuristic takes over — matching product terms and subreddit hints, and counting positive/negative sentiment words.

### 3. Visualization (`charts.py`)

Three charts are generated per research run:

1. **Sentiment donut chart** — Positive/negative/neutral split with percentages
2. **Subreddit distribution donut** — Top 10 subreddits + "Others"
3. **Sentiment trend** — Weekly line chart over the 3-month window

Charts are saved as PNGs at 180 DPI to `/tmp/research_charts/`.

### 4. Export (`sheets.py`)

Results are exported to a new Google Sheet with two worksheets:
- **Research Data**: Full post details (date, subreddit, title, sentiment, theme, summary, upvotes, comments, competitor mentions, URL)
- **Summary**: Aggregate metrics and top themes

The sheet is shared publicly (read-only). If Google Sheets fails (auth issues, quota), a CSV file is generated and sent via Telegram instead.

---

## Brand Configuration (`brands.json`)

Each brand entry contains:

```json
{
  "category": "beauty|finance|food|...",
  "keywords": ["search terms for fetching"],
  "product_terms": ["domain words for relevance filtering"],
  "competitors": ["competitor names for LLM context"],
  "subreddit_hints": ["target subreddit names"],
  "description": "what the brand does"
}
```

Pre-configured brands: **Groww** (finance), **Scapia** (finance), **Dot & Key** (beauty), **Licious** (food), **Deconstruct** (beauty), **Nua** (beauty).

New brands can be added via the `/research_add` Telegram command.

---

## Key Design Patterns

### Multi-layer fallback
Every layer has redundancy: 4 fetcher sources, 4 LLM providers, Sheets → CSV export. The system degrades gracefully rather than failing entirely.

### Async pipeline with progress updates
The research pipeline runs in a background thread (via `asyncio.to_thread`). Progress callbacks marshal real-time status messages to Telegram using `asyncio.run_coroutine_threadsafe`, keeping the bot responsive during long operations.

### Rate-limit awareness
Per-provider rate-limit flags are tracked. When a provider returns 429, it's skipped for that batch. Exponential backoff is used for retries. If all LLM providers are exhausted, keyword-based heuristics provide a lower-accuracy fallback.

### Deduplication
Posts are deduplicated by Reddit post ID across all sources before analysis, preventing redundant LLM calls and inflated results.

---

## Configuration

All secrets are stored in `.env` (see `.env.example` for the template).

| Variable | Required | Purpose |
|----------|----------|---------|
| `RESEARCH_BOT_TOKEN` | Yes | Telegram bot token |
| `AUTHORIZED_USERS` | Yes | JSON array of allowed Telegram user IDs |
| `GROQ_API_KEY` | Yes | Primary LLM provider |
| `CEREBRAS_API_KEY` | No | Fallback LLM provider |
| `SAMBANOVA_API_KEY` | No | Fallback LLM provider |
| `MISTRAL_API_KEY` | No | Fallback LLM provider |
| `GOOGLE_SHEETS_CREDS_FILE` | No | Google Sheets service account credentials |
| `GOOGLE_OAUTH_*` | No | OAuth2 tokens for Sheets (alternative to service account) |

Key constants (in `config.py`):
- `SEARCH_LOOKBACK_DAYS`: 90 (3-month window)
- `REDDIT_RATE_LIMIT_DELAY`: 2.0 seconds between Reddit requests
- `CHART_OUTPUT_DIR`: `/tmp/research_charts`

---

## Deployment

- Runs on a **Vultr VPS** (Mumbai region) managed by **systemd** (`research-bot.service` with `Restart=always`)
- Only one bot instance can poll Telegram at a time — do not run `bot.py` locally while the VPS instance is active
- Deploy flow: push code → SSH to VPS → `git pull` → `systemctl restart research-bot.service`
- See `CLAUDE.md` for full deployment instructions
