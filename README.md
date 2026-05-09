# local-mdsearch

Hybrid vector + BM25 search over your own markdown files.
AI-first: designed for use from Claude Code, Codex CLI, or the terminal.

**Portable**: one Python script, one config file. Qdrant runs in local-file
mode — no Docker, no network port. Clone, install deps, ingest, search.

## How It Works

```
urls.md  ─────────►  mdsearch.py fetch  ─────────►  docs/fetched/*.md
                                                            │
                                docs/**/*.md  ──────────────┤
                                                            ▼
                                            Ollama (qwen3-embedding:0.6b)
                                                            │
                                                 mdsearch.py ingest
                                          (chunk + embed + fit BM25 + upsert)
                                                            │
                                                  Qdrant Local Mode
                                                    (.db/ in repo)
                                            No Docker. No network port.
                                                            │
                                       ┌────────────────────┴────────────────────┐
                                       ▼                                         ▼
                              Dense vector index                         BM25 sparse index
                                 (semantic)                          (jieba CJK + English)
                                       └────────────────────┬────────────────────┘
                                                            ▼
                                                 RRF fusion (hybrid)
                                                            │
                                                            ▼
                                                 mdsearch.py search
```

1. `urls.md` — `[title](url)` list of pages to fetch (single source of truth).
2. `mdsearch.py fetch` — download URLs, strip nav/footer, save Markdown to `docs/fetched/`.
3. `mdsearch.py ingest` — heading-aware chunking, dense embeddings via Ollama, BM25 fit on the full corpus, write into `.db/`. Incremental on subsequent runs (SHA-256 hash cache); orphan chunks for shrunk files are deleted automatically.
4. `mdsearch.py search` — hybrid RRF fusion of dense + BM25 by default, with `--mode semantic|keyword` for exact matches or fuzzy concepts.

## Prerequisites

- Python 3.9+
- Ollama running locally (`ollama serve`)
- `qwen3-embedding:0.6b` pulled (`ollama pull qwen3-embedding:0.6b`)

## Quick Start

```bash
pip3 install -r requirements.txt
# (or, minimal:)
# pip3 install qdrant-client pyyaml jieba beautifulsoup4 markdownify

# Add URLs to urls.md, then:
python3 mdsearch.py fetch      # download → docs/fetched/
python3 mdsearch.py ingest     # index into .db/
python3 mdsearch.py search "your query"
```

## CLI Reference

| Command | What it does |
|---------|-------------|
| `ingest [--rebuild]` | Index markdown files |
| `search QUERY [--mode hybrid\|semantic\|keyword] [--filter k=v] [--json]` | Search |
| `fetch [--force] [--dry-run] [--limit N]` | Download URLs from urls.md |
| `stats` | Collection statistics |
| `filters` | Available filter keys and values |

## OL8 Setup

```bash
sudo dnf install python39 python39-pip
python3.9 -m pip install -r requirements.txt
python3.9 mdsearch.py ingest
```

## Claude Code Skill

```bash
mkdir -p ~/.claude/skills/mdsearch
cp SKILL.md ~/.claude/skills/mdsearch/
```

Then in Claude Code, use `/mdsearch` or Claude will auto-detect local search queries.

## Codex CLI

Pass the SKILL.md content as system context, or just invoke the CLI directly:
```bash
codex "search my local docs for kernel panic" -- python3 mdsearch.py search "kernel panic" --json
```

## Configuration

All settings are in `config.yaml`. Key fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `collection` | `my_docs` | Qdrant collection name |
| `db_path` | `.db/` | Local Qdrant storage |
| `embedding.model` | `qwen3-embedding:0.6b` | Ollama model |
| `dirs` | `[docs/]` | Directories to index |
| `chunk_size` | `1500` | Max chars per chunk |
| `path_meta` | see file | Metadata from path patterns |
| `fetch.output_dir` | `docs/fetched/` | Where fetched pages are saved |
| `fetch.delay` | `1.5` | Seconds between HTTP requests |

## Extending

See **SKILL.md → EXTENDING** section for step-by-step guides.
