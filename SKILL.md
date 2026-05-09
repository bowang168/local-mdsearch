---
name: mdsearch
description: >
  Search a local markdown knowledge base using hybrid vector + BM25 search.
  Wraps mdsearch.py at {{MDSEARCH_HOME}}.
triggers:
  commands:
    - /mdsearch
  phrases:
    - "search my docs"
    - "search local docs"
    - "check local kb"
    - "local knowledge base"
---

# mdsearch

Local hybrid search over your own markdown files.
Engine: Qdrant local mode + Ollama embeddings + BM25 sparse vectors.

`{{MDSEARCH_HOME}}` is the absolute path to the cloned repo, resolved by
`install.sh` at install time. If you see `{{MDSEARCH_HOME}}` literally in this
file, run `install.sh` again.

## SETUP (first time)

```bash
# 1. Install deps
pip3 install -r {{MDSEARCH_HOME}}/requirements.txt

# 2. Start Ollama + pull model
ollama serve &
ollama pull qwen3-embedding:0.6b

# 3. Add your docs (or fetch from web)
python3 {{MDSEARCH_HOME}}/mdsearch.py fetch          # download URLs from urls.md
python3 {{MDSEARCH_HOME}}/mdsearch.py ingest         # index docs/ into Qdrant
```

## SEARCH STRATEGY

| Query type | Recommended mode | Example |
|------------|-----------------|---------|
| Most queries | hybrid (default) | `"kernel panic after upgrade"` |
| Exact string: error codes, CVE, package names, commands | keyword | `"CVE-2024-26704"` |
| Conceptual / fuzzy | semantic | `"how does kdump work"` |

## TOOL INVOCATION (for AI agents)

Always use `--json` for machine-readable output. Errors go to stderr, JSON to
stdout, so parse stdout only. Exit code 2 means malformed `--filter` arg.

```bash
# Default search
python3 {{MDSEARCH_HOME}}/mdsearch.py search "QUERY" --json

# With filters (KEY must be in `filters` output; values vary per kb)
python3 {{MDSEARCH_HOME}}/mdsearch.py search "QUERY" --filter ol_version=ol9 --json
python3 {{MDSEARCH_HOME}}/mdsearch.py search "QUERY" --filter tags=security --json

# Keyword mode (CVE, package names, error codes)
python3 {{MDSEARCH_HOME}}/mdsearch.py search "QUERY" --mode keyword --json

# More results
python3 {{MDSEARCH_HOME}}/mdsearch.py search "QUERY" --limit 10 --json

# Discover available filters and values
python3 {{MDSEARCH_HOME}}/mdsearch.py filters
```

## JSON OUTPUT SCHEMA

`search --json` returns a list of result objects. The fixed top-level keys
appear in every record; extra keys (e.g. `ol_version`, `product`, `topic`,
`page_count`, `fetched`) come from the document's frontmatter and path-derived
metadata, so the exact set varies per knowledge base.

```json
[
  {
    "score": 0.6111,
    "source_file": "docs/fetched/ol9/oracle-linux_9_network.md",
    "title": "Oracle Linux 9: Setting Up Networking",
    "tags": ["oracle-linux", "ol9", "networking"],
    "source_url": "https://docs.oracle.com/en/operating-systems/oracle-linux/9/network/",
    "text": "First 300 chars of the matching chunk...",
    "ol_version": "ol9",
    "product": "shared",
    "topic": "",
    "fetched": "2026-05-09T06:34:47Z",
    "page_count": 30
  }
]
```

| Key | Source | Notes |
|-----|--------|-------|
| `score` | retrieval engine | Higher = better; hybrid uses RRF, range typically 0–1 |
| `source_file` | path | Path relative to repo root; opens with any editor |
| `title` | frontmatter | First non-empty `<title>` from the fetched HTML |
| `tags` | frontmatter | List of strings; queryable via `--filter tags=<tag>` |
| `source_url` | frontmatter | Original URL the doc was fetched from |
| `text` | chunk | First 300 chars of the matched chunk (preview) |
| `ol_version` | path_meta | Filterable; e.g. `ol9` |
| `product`, `topic` | path_meta / frontmatter | May be empty when not classified |
| `fetched`, `page_count` | frontmatter | Provenance from `fetch` command |

## FETCHING DOCS

```bash
# Edit urls.md to add [title](url) links, then:
python3 {{MDSEARCH_HOME}}/mdsearch.py fetch              # fetch new URLs only
python3 {{MDSEARCH_HOME}}/mdsearch.py fetch --force      # re-fetch all
python3 {{MDSEARCH_HOME}}/mdsearch.py fetch --dry-run    # preview what would be fetched
python3 {{MDSEARCH_HOME}}/mdsearch.py fetch --limit 5    # test with 5 URLs
```

`fetch` follows `<link rel="next">` to walk multi-page books, dedupes
sub-pages across overlapping entries, and runs top-level URLs through a
thread pool (`fetch.max_workers` in config.yaml, default 6).

## INGESTING DOCS

```bash
python3 {{MDSEARCH_HOME}}/mdsearch.py ingest             # incremental (unchanged files skipped)
python3 {{MDSEARCH_HOME}}/mdsearch.py ingest --rebuild   # full rebuild (drop + re-index)
```

Incremental ingest re-fits BM25 on the full corpus by scrolling surviving
chunks from Qdrant and merging with new ones, so IDF stays accurate after
add / remove / shrink.

## DIAGNOSTICS

```bash
python3 {{MDSEARCH_HOME}}/mdsearch.py stats              # points count, BM25 vocab, DB path
python3 {{MDSEARCH_HOME}}/mdsearch.py filters            # available filter keys and values
```

## EXTENDING

**Add a new docs directory:**
Edit `config.yaml` → `dirs:` → add path → run `mdsearch.py ingest`

**Add metadata from path:**
Edit `config.yaml` → `path_meta:` → add `{regex, key, format}` entry

**Add tags / topic classification:**
Edit `config.yaml` → `fetch.topic_rules:` (regex → topic) and
`fetch.base_tags:` (fixed prefix). Re-run `fetch --force` to overwrite
existing files, then `ingest --rebuild`.

**Change embedding model:**
Edit `config.yaml` → `embedding.model` and `embedding.dim` → run
`mdsearch.py ingest --rebuild`

**Add URLs to fetch:**
Edit `urls.md` → add `[title](url)` lines → run `mdsearch.py fetch`
