#!/usr/bin/env python3
"""
mdsearch — local markdown knowledge base with hybrid vector + BM25 search

Subcommands:
    ingest   Index markdown files into Qdrant
    search   Search the index
    fetch    Download URLs from urls.md, convert to markdown
    stats    Show collection statistics
    filters  List available filter keys and values

Usage:
    python3 mdsearch.py ingest [--rebuild] [--config config.yaml]
    python3 mdsearch.py search "query" [--mode hybrid|semantic|keyword]
                                       [--limit N] [--filter k=v] [--json]
    python3 mdsearch.py fetch [--force] [--dry-run] [--limit N]
    python3 mdsearch.py stats
    python3 mdsearch.py filters
"""

# ── 1. IMPORTS & OPTIONAL DEPS ──────────────────────────────────────────────

import argparse
import hashlib
import json
import math
import re
import ssl
import sys
import threading
import time
import uuid
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

# jieba transitively imports pkg_resources, which emits a DeprecationWarning on
# 3.12+. AI agents read our stderr, so silence the upstream noise here. The
# filter must be installed before any third-party import that may trigger it.
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

from qdrant_client import QdrantClient, models  # noqa: E402

try:
    import yaml
except ImportError:
    print("PyYAML not found. Run: pip3 install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import jieba
    jieba.setLogLevel(20)
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md_convert
    HAS_FETCH_DEPS = True
except ImportError:
    HAS_FETCH_DEPS = False


# ── Module-level constants ──────────────────────────────────────────────────

UPSERT_BATCH_SIZE = 50
EMBED_BATCH_SIZE = 16
MAX_FILENAME_LEN = 200
HTTP_TIMEOUT_DEFAULT = 30
EMBED_RETRY_DELAY = 1.0
SCROLL_BATCH_SIZE = 200
MAX_SUB_PAGES = 100
MIN_PAGE_BODY_CHARS = 30  # markdown shorter than this is considered empty boilerplate
DEFAULT_FETCH_WORKERS = 6
DEFAULT_FETCH_DELAY = 0.0       # seconds between top-level requests per worker
DEFAULT_FETCH_CRAWL_DELAY = 0.0  # seconds between sub-pages within a rel=next chain

DEFAULT_STRIP_SELECTORS = [
    "nav", "header", "footer", "aside", "script", "style", "noscript",
    ".breadcrumb", ".breadcrumbs", ".sidebar", ".toc", "#toc", "#sidebar",
    ".header-nav", ".footer-nav", ".cookie-banner", "#cookie-banner",
    ".feedback", ".feedback-section", "#feedback", ".copyright",
    ".book-nav", ".navigation", ".nav-bar",
]


def _build_ssl_context() -> ssl.SSLContext:
    """SSL context that prefers certifi's CA bundle if installed.

    macOS python.org installations do not trust the system keychain, so HTTPS
    fetches fail with CERTIFICATE_VERIFY_FAILED. Falling back to certifi gives
    a working bundle without disabling verification.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CONTEXT = _build_ssl_context()


# ── 2. CONFIG ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "config.yaml"


REQUIRED_TOP_FIELDS = ("collection", "db_path", "embedding")
REQUIRED_EMBEDDING_FIELDS = ("url", "model", "dim")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load config.yaml. Resolve relative paths to absolute based on config file location.

    Validates required top-level and embedding fields; exits with a clear message
    if any are missing so the user does not see a downstream KeyError.
    """
    cfg_path = Path(path).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        print(f"[ERROR] config.yaml must be a YAML mapping, got {type(cfg).__name__}",
              file=sys.stderr)
        sys.exit(1)
    for key in REQUIRED_TOP_FIELDS:
        if key not in cfg:
            print(f"[ERROR] config.yaml missing required field: {key}", file=sys.stderr)
            sys.exit(1)
    embedding = cfg.get("embedding") or {}
    if not isinstance(embedding, dict):
        print("[ERROR] config.yaml: 'embedding' must be a mapping", file=sys.stderr)
        sys.exit(1)
    for key in REQUIRED_EMBEDDING_FIELDS:
        if key not in embedding:
            print(f"[ERROR] config.yaml missing required field: embedding.{key}",
                  file=sys.stderr)
            sys.exit(1)
    cfg["_base"] = cfg_path.parent
    return cfg


def resolve_path(cfg: dict, rel: str) -> Path:
    """Resolve a config-relative or absolute path."""
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    return (cfg["_base"] / p).resolve()


# ── 3. BM25 ──────────────────────────────────────────────────────────────────

_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
_EN_SPLIT_RE = re.compile(r'[a-zA-Z0-9][-a-zA-Z0-9_.]*[a-zA-Z0-9]|[a-zA-Z0-9]+')

STOPWORDS = frozenset({
    # Chinese
    "的","了","在","是","我","有","和","就","不","人","都","一","一个","上","也",
    "很","到","说","要","去","你","会","着","没有","看","好","自己","这","他",
    "她","它","们","那","被","从","把","其","与","但","而","对","以","可以",
    # English
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","during","before","after","above","below","between",
    "out","off","over","under","again","further","then","once","here","there",
    "when","where","why","how","all","each","every","both","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","and","but","or","if","while","because","this","that","these",
    "those","it","its","i","me","my","we","our","you","your","he","him","his",
    "she","her","they","them","their","what","which","who","whom",
})


def tokenize(text: str) -> list[str]:
    """Tokenize mixed CJK/English text. Uses jieba if available, else regex."""
    tokens = []
    text_lower = text.lower()
    if HAS_JIEBA:
        for word in jieba.cut(text_lower):
            word = word.strip()
            if not word or word in STOPWORDS:
                continue
            if _CJK_RE.search(word):
                tokens.append(word)
            else:
                for tok in _EN_SPLIT_RE.findall(word):
                    if tok not in STOPWORDS and len(tok) >= 2:
                        tokens.append(tok)
    else:
        for tok in _EN_SPLIT_RE.findall(text_lower):
            if tok not in STOPWORDS and len(tok) >= 2:
                tokens.append(tok)
    return tokens


class BM25Encoder:
    """BM25 sparse vector encoder. k1=1.5, b=0.75 (Okapi defaults)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[int, float] = {}
        self.avg_dl: float = 0.0
        self.n_docs: int = 0

    def fit(self, documents: list[str]) -> "BM25Encoder":
        self.n_docs = len(documents)
        if not self.n_docs:
            return self
        doc_freq: Counter = Counter()
        total_len = 0
        all_tokens: set[str] = set()
        for doc in documents:
            toks = tokenize(doc)
            total_len += len(toks)
            unique = set(toks)
            for t in unique:
                doc_freq[t] += 1
            all_tokens |= unique
        self.avg_dl = total_len / self.n_docs
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        for t, i in self.vocab.items():
            df = doc_freq[t]
            self.idf[i] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def encode(self, text: str) -> models.SparseVector:
        toks = tokenize(text)
        if not toks:
            return models.SparseVector(indices=[], values=[])
        tf = Counter(toks)
        dl = len(toks)
        avg_dl = self.avg_dl or 1.0
        indices, values = [], []
        for tok, count in tf.items():
            idx = self.vocab.get(tok)
            if idx is None:
                idx = len(self.vocab)
                self.vocab[tok] = idx
                self.idf[idx] = math.log((self.n_docs + 0.5) / 0.5 + 1.0) if self.n_docs else 1.0
            idf = self.idf.get(idx, 1.0)
            score = idf * (count * (self.k1 + 1)) / (count + self.k1 * (1 - self.b + self.b * dl / avg_dl))
            if score > 0:
                indices.append(idx)
                values.append(round(score, 6))
        return models.SparseVector(indices=indices, values=values)

    def save(self, path: str):
        state = {"k1": self.k1, "b": self.b, "vocab": self.vocab,
                 "idf": {str(k): v for k, v in self.idf.items()},
                 "avg_dl": self.avg_dl, "n_docs": self.n_docs}
        Path(path).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25Encoder":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls(k1=state["k1"], b=state["b"])
        enc.vocab = state["vocab"]
        enc.idf = {int(k): v for k, v in state["idf"].items()}
        enc.avg_dl = state["avg_dl"]
        enc.n_docs = state["n_docs"]
        return enc

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ── 4. EMBED ─────────────────────────────────────────────────────────────────

def _embed_request(inputs: list[str], cfg: dict) -> list[list[float]]:
    """POST to Ollama /api/embed with a list of inputs; retries once on transport error."""
    url = cfg["embedding"]["url"]
    model = cfg["embedding"]["model"]
    payload = json.dumps({"model": model, "input": inputs}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(req, timeout=HTTP_TIMEOUT_DEFAULT, context=_SSL_CONTEXT) as resp:
                body = json.loads(resp.read())
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
                raise RuntimeError(
                    f"Ollama returned malformed response: expected {len(inputs)} embeddings, "
                    f"got {type(embeddings).__name__} of length "
                    f"{len(embeddings) if isinstance(embeddings, list) else 'n/a'}"
                )
            return embeddings
        except (URLError, TimeoutError) as e:
            last_err = e
            if attempt == 0:
                time.sleep(EMBED_RETRY_DELAY)
                continue
            raise RuntimeError(f"Ollama unreachable at {url}: {e}") from e
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Ollama returned invalid payload: {e}") from e
    # Defensive: loop above always returns or raises.
    raise RuntimeError(f"Embedding failed at {url}: {last_err}")


def embed(text: str, cfg: dict) -> list[float]:
    """Embed a single string. Raises RuntimeError on failure."""
    return _embed_request([text], cfg)[0]


def embed_batch(texts: list[str], cfg: dict, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed many strings using Ollama's batch API; chunked to avoid huge payloads."""
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        out.extend(_embed_request(texts[start:start + batch_size], cfg))
    return out


# ── 5. INGEST ────────────────────────────────────────────────────────────────

_FRONTMATTER_END_RE = re.compile(r'\n---[ \t]*(?:\n|$)')


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown. Returns (meta, body).

    Recognises only a closing ``---`` on its own line, so YAML document separators
    embedded in frontmatter values do not truncate parsing.
    """
    if not text.startswith("---\n") and text.rstrip("\r") != "---":
        return {}, text
    m = _FRONTMATTER_END_RE.search(text, 3)
    if m is None:
        return {}, text
    fm_text = text[3:m.start()].strip()
    body = text[m.end():].lstrip("\n")
    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        print(f"[WARN] frontmatter YAML parse error: {e}", file=sys.stderr)
        parsed = None
    # Frontmatter that parses to a scalar/list is not usable as metadata.
    meta = parsed if isinstance(parsed, dict) else {}
    return meta, body


def extract_path_meta(path: str, patterns: list[dict]) -> dict:
    """Apply all matching path_meta patterns from config. Returns merged dict."""
    meta = {}
    for rule in patterns:
        regex = rule.get("regex")
        key = rule.get("key")
        if not regex or not key:
            print(f"[WARN] path_meta rule missing regex/key: {rule}", file=sys.stderr)
            continue
        m = re.search(regex, path)
        if not m:
            continue
        if m.groups():
            captured = m.group(1)
        else:
            print(f"[WARN] path_meta regex {regex!r} has no capture group; skipping",
                  file=sys.stderr)
            continue
        fmt = rule.get("format")
        meta[key] = fmt.replace("{1}", captured) if fmt else captured
    return meta


def heading_aware_chunk(text: str, max_chars: int = 1500) -> list[str]:
    """Split markdown by headings, keeping heading breadcrumb context.
    If a section exceeds max_chars, split line-by-line.
    """
    heading_re = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    headings = [(m.start(), len(m.group(1)), m.group(2)) for m in heading_re.finditer(text)]
    if not headings:
        return _split_by_size(text, max_chars)

    chunks = []
    breadcrumb: list[str] = []

    positions = [h[0] for h in headings] + [len(text)]
    for i, (pos, level, title) in enumerate(headings):
        section_text = text[pos:positions[i + 1]].strip()
        breadcrumb = breadcrumb[:level - 1] + [title]
        prefix = " > ".join(breadcrumb[:-1])
        header = f"[{prefix}] " if prefix else ""
        full = header + section_text
        if len(full) <= max_chars:
            chunks.append(full)
        else:
            for sub in _split_by_size(full, max_chars):
                chunks.append(sub)
    return chunks


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """Fallback: split text into chunks of max_chars on line boundaries."""
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > max_chars and current:
            chunks.append("".join(current).strip())
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


PAYLOAD_INDEXES = ["source_file", "title", "ol_version", "product", "topic", "tags", "source_url"]


def ensure_collection(client: QdrantClient, cfg: dict):
    """Create collection with dense + sparse vectors if it doesn't exist."""
    name = cfg["collection"]
    dim = cfg["embedding"]["dim"]
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams()},
        )
        for field in PAYLOAD_INDEXES:
            client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        print(f"Created collection: {name}")


HASH_CACHE_FILE = ".hash_cache.json"
BM25_MODEL_FILE = ".bm25.json"
SKIP_PATTERNS = [".db", ".git", "__pycache__", ".DS_Store"]


def _scroll_existing_chunks(
    client: QdrantClient, collection: str, exclude_files: set[str]
) -> list[str]:
    """Stream every chunk_text from Qdrant whose source_file is NOT in exclude_files.

    Used to refit BM25 on the full corpus on incremental ingest, so IDF stays
    accurate as files are added/removed.
    """
    chunks: list[str] = []
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=collection, limit=SCROLL_BATCH_SIZE, offset=offset,
            with_payload=["chunk_text", "source_file"], with_vectors=False,
        )
        for pt in pts:
            payload = pt.payload or {}
            if payload.get("source_file") in exclude_files:
                continue
            text = payload.get("chunk_text") or ""
            if text:
                chunks.append(text)
        if next_off is None:
            break
        offset = next_off
    return chunks


def _delete_file_points(client: QdrantClient, collection: str, source_file: str) -> None:
    """Delete all points for a given source_file (handles chunk-count shrinkage)."""
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(
                key="source_file", match=models.MatchValue(value=source_file),
            )])
        ),
    )


def cmd_ingest(cfg: dict, rebuild: bool = False):
    """
    Scan configured dirs, chunk markdown, embed, upsert to Qdrant.

    Incremental by default: skip files whose SHA-256 hash hasn't changed.
    --rebuild: drops the collection and re-creates from scratch.

    BM25 is refit on the full corpus on every run that has changes, by combining
    new chunks with surviving chunks scrolled from Qdrant. This keeps IDF accurate
    after files are added, removed, or shrunk.
    """
    db_path = resolve_path(cfg, cfg["db_path"])
    db_path.mkdir(parents=True, exist_ok=True)
    hash_cache_path = db_path / HASH_CACHE_FILE
    bm25_path = db_path / BM25_MODEL_FILE

    client = QdrantClient(path=str(db_path))
    collection = cfg["collection"]
    chunk_size = cfg.get("chunk_size", 1500)

    if rebuild:
        if collection in [c.name for c in client.get_collections().collections]:
            client.delete_collection(collection)
            print(f"Dropped collection: {collection}")
        hash_cache: dict = {}
    else:
        hash_cache = json.loads(hash_cache_path.read_text()) if hash_cache_path.exists() else {}

    ensure_collection(client, cfg)

    md_files: list[Path] = []
    for d in cfg.get("dirs", []):
        dir_path = resolve_path(cfg, d)
        if not dir_path.exists():
            print(f"[WARN] Dir not found: {dir_path}", file=sys.stderr)
            continue
        for f in dir_path.rglob("*.md"):
            if any(skip in str(f) for skip in SKIP_PATTERNS):
                continue
            md_files.append(f)

    print(f"Found {len(md_files)} markdown files")

    path_meta_rules = cfg.get("path_meta", [])
    changed_files: list[tuple[Path, str, str, str]] = []
    new_chunks_per_file: list[list[str]] = []

    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        rel = str(f.relative_to(cfg["_base"]))
        if hash_cache.get(rel) == file_hash and not rebuild:
            continue
        _, body = parse_frontmatter(text)
        chunks = heading_aware_chunk(body, chunk_size)
        changed_files.append((f, rel, text, file_hash))
        new_chunks_per_file.append(chunks)

    if not changed_files:
        print("Nothing changed. Index is up to date.")
        client.close()
        return

    print(f"Processing {len(changed_files)} changed files...")

    # Refit BM25 on full corpus: surviving chunks + newly produced chunks.
    changed_set = {rel for _, rel, _, _ in changed_files}
    all_chunks_for_bm25: list[str] = []
    if not rebuild:
        existing = _scroll_existing_chunks(client, collection, changed_set)
        all_chunks_for_bm25.extend(existing)
    for chunks in new_chunks_per_file:
        all_chunks_for_bm25.extend(chunks)

    print(f"Fitting BM25 on {len(all_chunks_for_bm25)} chunks...")
    bm25 = BM25Encoder()
    bm25.fit(all_chunks_for_bm25)
    bm25.save(str(bm25_path))

    # Delete any orphan chunks for changed files (handles shrinkage).
    if not rebuild:
        for _, rel, _, _ in changed_files:
            _delete_file_points(client, collection, rel)

    batch: list[models.PointStruct] = []

    for (f, rel, text, file_hash), chunks in zip(changed_files, new_chunks_per_file):
        if not chunks:
            hash_cache[rel] = file_hash
            print(f"  ingested: {rel} (0 chunks)")
            continue
        meta, _ = parse_frontmatter(text)
        path_meta = extract_path_meta(rel, path_meta_rules)
        try:
            dense_vecs = embed_batch(chunks, cfg)
        except RuntimeError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vecs)):
            sparse_vec = bm25.encode(chunk)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}#{i}"))
            payload = {
                "source_file": rel,
                "chunk_index": i,
                "chunk_text": chunk[:2000],
                "title": meta.get("title", f.stem),
                "source_url": meta.get("source_url", ""),
                "tags": meta.get("tags", []),
                "topic": meta.get("topic", ""),
                **path_meta,
                **{k: v for k, v in meta.items()
                   if k not in ("title", "source_url", "tags", "topic")},
            }

            vectors: dict = {"dense": dense_vec}
            if sparse_vec.indices:
                vectors["bm25"] = sparse_vec

            batch.append(models.PointStruct(id=point_id, vector=vectors, payload=payload))
            if len(batch) >= UPSERT_BATCH_SIZE:
                client.upsert(collection_name=collection, points=batch)
                batch = []

        hash_cache[rel] = file_hash
        print(f"  ingested: {rel} ({len(chunks)} chunks)")

    if batch:
        client.upsert(collection_name=collection, points=batch)

    hash_cache_path.write_text(json.dumps(hash_cache, ensure_ascii=False, indent=2))
    print("Done. Hash cache updated.")
    client.close()


# ── 6. SEARCH ────────────────────────────────────────────────────────────────

def build_filter(filters: Optional[dict]) -> Optional[models.Filter]:
    if not filters:
        return None
    conditions = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
                  for k, v in filters.items()]
    return models.Filter(must=list(conditions))


def search(query: str, cfg: dict, mode: str = "hybrid",
           limit: int = 5, filters: Optional[dict] = None) -> list[dict]:
    """
    Search modes:
      hybrid  — dense prefetch + BM25 prefetch → RRF fusion (default)
      semantic — dense only
      keyword  — BM25 only

    Falls back to semantic if BM25 model missing or sparse vector is empty.
    """
    db_path = resolve_path(cfg, cfg["db_path"])
    bm25_path = db_path / BM25_MODEL_FILE
    client = QdrantClient(path=str(db_path))
    collection = cfg["collection"]
    qdrant_filter = build_filter(filters)

    bm25: Optional[BM25Encoder] = None
    if mode in ("hybrid", "keyword"):
        if bm25_path.exists():
            bm25 = BM25Encoder.load(str(bm25_path))
        else:
            print("[WARN] BM25 model not found; falling back to semantic", file=sys.stderr)
            mode = "semantic"

    results_raw = None

    if mode == "keyword":
        assert bm25 is not None  # guaranteed by branch above
        sparse_vec = bm25.encode(query)
        if not sparse_vec.indices:
            print("[WARN] Query produced empty BM25 vector; falling back to semantic",
                  file=sys.stderr)
            mode = "semantic"
        else:
            results_raw = client.query_points(
                collection_name=collection, query=sparse_vec,
                using="bm25", query_filter=qdrant_filter,
                limit=limit, with_payload=True,
            )

    if mode == "semantic":
        vec = embed(query, cfg)
        results_raw = client.query_points(
            collection_name=collection, query=vec,
            using="dense", query_filter=qdrant_filter,
            limit=limit, with_payload=True,
        )

    if mode == "hybrid":
        assert bm25 is not None  # guaranteed by branch above
        vec = embed(query, cfg)
        sparse_vec = bm25.encode(query)
        prefetch = [models.Prefetch(query=vec, using="dense",
                                    limit=limit * 3, filter=qdrant_filter)]
        if sparse_vec.indices:
            prefetch.append(models.Prefetch(query=sparse_vec, using="bm25",
                                            limit=limit * 3, filter=qdrant_filter))
            fusion = models.FusionQuery(fusion=models.Fusion.RRF)
            results_raw = client.query_points(
                collection_name=collection, prefetch=prefetch,
                query=fusion, limit=limit, with_payload=True,
            )
        else:
            results_raw = client.query_points(
                collection_name=collection, query=vec,
                using="dense", query_filter=qdrant_filter,
                limit=limit, with_payload=True,
            )

    client.close()
    if results_raw is None:
        return []
    return [{"score": p.score, "payload": p.payload or {}} for p in results_raw.points]


# ── 7. FETCH ─────────────────────────────────────────────────────────────────

def _fetch_html(url: str, timeout: int, user_agent: str) -> tuple[str, str, str]:
    """Fetch one URL. Returns ``(html, content_type, final_url)``; on failure
    returns ``("", "", url)`` so the caller can decide how to recover."""
    req = Request(url, headers={"User-Agent": user_agent,
                                "Accept": "text/html,application/xhtml+xml"})
    try:
        with urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            ctype = resp.headers.get("Content-Type", "") or ""
            return html, ctype, resp.url or url
    except (URLError, TimeoutError, OSError, UnicodeDecodeError) as e:
        print(f"    [ERROR] fetch failed: {url}: {e}", file=sys.stderr)
        return "", "", url


def _html_to_markdown(html: str, strip_selectors: list[str]) -> tuple[str, str]:
    """Pick the main content area, strip chrome, convert to Markdown.
    Returns ``(page_title, markdown)``."""
    soup = BeautifulSoup(html, "html.parser")
    page_title = (soup.title.get_text(strip=True) if soup.title else "") or ""

    main = (
        soup.find("div", id="content")
        or soup.find("div", class_="book-body")
        or soup.find("div", class_="chapter")
        or soup.find("div", class_="article")
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="content")
        or soup.find("body")
        or soup
    )

    for tag in main.find_all(["nav", "header", "footer", "aside",
                              "script", "style", "noscript"]):
        tag.decompose()
    for sel in strip_selectors:
        for el in main.select(sel):
            el.decompose()

    markdown = md_convert(
        str(main), heading_style="ATX", bullets="-",
        strip=["img", "svg", "input", "button", "form", "iframe"],
    )
    markdown = re.sub(r'\n{4,}', '\n\n\n', markdown).strip()
    return page_title, markdown


class _ThreadSafeSet:
    """Minimal thread-safe set with atomic add-if-absent for cross-book dedup."""

    def __init__(self) -> None:
        self._set: set = set()
        self._lock = threading.Lock()

    def add_if_absent(self, item: str) -> bool:
        """Atomically claim ownership of ``item``. Returns True if newly added,
        False if it was already present (i.e. another worker claimed it)."""
        with self._lock:
            if item in self._set:
                return False
            self._set.add(item)
            return True


def _discover_rel_next(start_url: str, first_html: str,
                       fetched_global: _ThreadSafeSet,
                       timeout: int, user_agent: str,
                       crawl_delay: float) -> list[tuple[str, str]]:
    """Follow ``<link rel="next">`` from ``<head>`` to walk a multi-page book.

    Returns ``[(url, html), ...]`` starting with the entry page. Stays under
    the entry URL's base directory, dedupes against ``fetched_global`` so the
    same chapter is not re-downloaded across overlapping book entries, and
    caps at ``MAX_SUB_PAGES`` for safety.
    """
    start_clean = start_url.rstrip("/")
    pages: list[tuple[str, str]] = [(start_url, first_html)]
    seen = {start_clean}
    fetched_global.add_if_absent(start_clean)

    base_dir = urlparse(start_url).path.rstrip("/") + "/"
    current_url, current_html = start_url, first_html

    while len(pages) < MAX_SUB_PAGES:
        soup = BeautifulSoup(current_html, "html.parser")
        nxt = soup.find("link", rel="next")
        href = nxt.get("href") if nxt else None
        if not href:
            break

        next_url = urljoin(current_url, href)
        parsed = urlparse(next_url)
        next_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        clean = next_url.rstrip("/")

        if clean in seen:
            break
        if not parsed.path.startswith(base_dir):
            break  # rel=next escaping the book scope; stop
        if not fetched_global.add_if_absent(clean):
            break  # another worker already claimed this sub-page

        if crawl_delay > 0:
            time.sleep(crawl_delay)
        html, _, _ = _fetch_html(next_url, timeout, user_agent)
        if not html:
            break

        pages.append((next_url, html))
        seen.add(clean)
        current_url, current_html = next_url, html

    return pages


def _fetch_one_book(
    idx: int, total: int, entry: dict, *,
    output_dir_resolved: Path, base_dir: Path,
    subfolder_rules: list[dict], topic_rules: list[dict], base_tags: list[str],
    strip_selectors: list[str],
    timeout: int, user_agent: str, crawl_delay: float, delay: float,
    force: bool, fetched_global: _ThreadSafeSet, print_lock: threading.Lock,
) -> None:
    """Worker: process one top-level URL — fetch + crawl + convert + write."""
    from datetime import datetime, timezone

    url = entry["url"]
    subfolder = classify_url(url, subfolder_rules)
    out_file = _safe_output_path(url, output_dir_resolved, subfolder=subfolder)
    if out_file is None:
        with print_lock:
            print(f"  [{idx}/{total}] [ERROR] refusing unsafe URL path: {url}",
                  file=sys.stderr)
        return

    if out_file.exists() and not force:
        with print_lock:
            print(f"  [{idx}/{total}] skip (exists): "
                  f"{out_file.relative_to(output_dir_resolved)}")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    if delay > 0:
        time.sleep(delay)

    html, _, final_url = _fetch_html(url, timeout, user_agent)
    if not html:
        return  # _fetch_html already logged

    pages = _discover_rel_next(final_url, html, fetched_global,
                               timeout, user_agent, crawl_delay)

    page_title = entry["title"]
    parts: list[str] = []
    for purl, phtml in pages:
        ptitle, pmd = _html_to_markdown(phtml, strip_selectors)
        if not parts and ptitle:
            page_title = ptitle
        if not pmd or len(pmd) < MIN_PAGE_BODY_CHARS:
            continue
        if parts:
            parts.append(f"\n\n---\n<!-- page: {purl} -->\n\n{pmd}")
        else:
            parts.append(pmd)

    markdown = "\n".join(parts).strip()
    if not markdown:
        with print_lock:
            print(f"  [{idx}/{total}] [WARN] no extractable content: {url}",
                  file=sys.stderr)
        return

    tags = build_doc_tags(url, base_tags, subfolder, topic_rules)
    fetched_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frontmatter_lines = [
        "---",
        f"title: {json.dumps(page_title)}",
        f"source_url: {url}",
        f"fetched: {fetched_ts}",
        f"page_count: {len(pages)}",
    ]
    if tags:
        frontmatter_lines.append("tags:")
        frontmatter_lines.extend(f"  - {json.dumps(t)}" for t in tags)
    frontmatter_lines.append("---")
    frontmatter_lines.append("")
    frontmatter_lines.append("")
    frontmatter = "\n".join(frontmatter_lines)

    out_file.write_text(frontmatter + markdown, encoding="utf-8")
    with print_lock:
        print(f"  [{idx}/{total}] saved: {out_file.relative_to(base_dir)} "
              f"({len(pages)} pages, {len(markdown):,} chars, "
              f"tags={tags or '∅'})")


def cmd_fetch(cfg: dict, force: bool = False, dry_run: bool = False,
              limit: int | None = None, filter_str: str | None = None):
    """
    Read urls.md, fetch each URL (and its rel="next" chain for multi-page docs),
    convert HTML → Markdown, save to output_dir. Top-level URLs are fetched
    concurrently using ``fetch.max_workers`` workers (default 6); rel=next
    chains stay sequential per book because each chapter URL is only known
    after the previous chapter is downloaded.

    Skip already-saved files unless --force.
    """
    if not HAS_FETCH_DEPS:
        print("Missing deps: pip3 install markdownify beautifulsoup4", file=sys.stderr)
        sys.exit(1)

    fetch_cfg = cfg.get("fetch", {})
    urls_file = resolve_path(cfg, fetch_cfg.get("urls_file", "urls.md"))
    output_dir = resolve_path(cfg, fetch_cfg.get("output_dir", "docs/fetched/"))
    delay = float(fetch_cfg.get("delay", DEFAULT_FETCH_DELAY))
    crawl_delay = float(fetch_cfg.get("crawl_delay", DEFAULT_FETCH_CRAWL_DELAY))
    timeout = int(fetch_cfg.get("timeout", HTTP_TIMEOUT_DEFAULT))
    user_agent = fetch_cfg.get("user_agent", "mdsearch/1.0")
    strip_selectors = fetch_cfg.get("strip_selectors", DEFAULT_STRIP_SELECTORS)
    subfolder_rules = fetch_cfg.get("subfolder_rules", [])
    topic_rules = fetch_cfg.get("topic_rules", [])
    base_tags = fetch_cfg.get("base_tags", []) or []
    max_workers = max(1, int(fetch_cfg.get("max_workers", DEFAULT_FETCH_WORKERS)))

    if not urls_file.exists():
        print(f"urls.md not found at {urls_file}. Create it with [title](url) links.")
        return

    link_re = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    text = urls_file.read_text(encoding="utf-8")
    entries: list[dict] = []
    seen: set[str] = set()
    for m in link_re.finditer(text):
        url = m.group(2).strip().rstrip("/")
        if url in seen:
            continue
        seen.add(url)
        if filter_str and filter_str not in url:
            continue
        entries.append({"title": m.group(1).strip(), "url": url})

    if limit:
        entries = entries[:limit]

    print(f"Found {len(entries)} URLs to fetch (workers={max_workers}, "
          f"delay={delay}s, crawl_delay={crawl_delay}s)")
    if dry_run:
        for e in entries:
            print(f"  {e['url']}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()
    base_dir = cfg["_base"]

    fetched_global = _ThreadSafeSet()
    print_lock = threading.Lock()
    total = len(entries)
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fetch") as ex:
        futures = [
            ex.submit(
                _fetch_one_book, i + 1, total, entry,
                output_dir_resolved=output_dir_resolved, base_dir=base_dir,
                subfolder_rules=subfolder_rules, topic_rules=topic_rules,
                base_tags=base_tags, strip_selectors=strip_selectors,
                timeout=timeout, user_agent=user_agent,
                crawl_delay=crawl_delay, delay=delay,
                force=force, fetched_global=fetched_global, print_lock=print_lock,
            )
            for i, entry in enumerate(entries)
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                with print_lock:
                    print(f"  [ERROR] worker raised: {e}", file=sys.stderr)

    elapsed = time.monotonic() - started
    print(f"Fetch complete in {elapsed:.1f}s.")


_UNSAFE_FILENAME_RE = re.compile(r'[^\w\-.]')


def sanitize_url_to_filename(url: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Derive a safe filename stem from a URL path.

    Strips leading/trailing slashes, replaces path separators with underscores,
    drops .html, collapses anything outside ``[\\w\\-.]`` to ``_``, then
    collapses any ``..`` runs (parent-dir literals) and truncates to ``max_len``.
    Containment under the output dir is still enforced separately by
    ``_safe_output_path``.
    """
    parsed = urlparse(url)
    raw = parsed.path.strip("/").replace("/", "_")
    if raw.lower().endswith(".html"):
        raw = raw[:-5]
    cleaned = _UNSAFE_FILENAME_RE.sub("_", raw)
    cleaned = re.sub(r'\.{2,}', '_', cleaned)
    cleaned = cleaned.strip("._") or "index"
    return cleaned[:max_len]


def _safe_output_path(url: str, output_dir_resolved: Path,
                      subfolder: str = "") -> Optional[Path]:
    """Build an output path under output_dir, refusing any path that escapes it.

    ``subfolder`` is appended after sanitization so configured bucketing rules
    cannot smuggle ``..`` segments. Unsafe candidates (parent escape via symlink,
    odd separators) return None.
    """
    name = sanitize_url_to_filename(url)
    safe_sub = ""
    if subfolder:
        safe_sub = _UNSAFE_FILENAME_RE.sub("_", subfolder).strip("._")
    target_dir = (output_dir_resolved / safe_sub) if safe_sub else output_dir_resolved
    candidate = (target_dir / f"{name}.md").resolve()
    try:
        candidate.relative_to(output_dir_resolved)
    except ValueError:
        return None
    return candidate


def classify_url(url: str, rules: list[dict]) -> str:
    """Map URL to a subfolder by matching the first rule whose regex hits its path.

    Each rule is ``{"regex": "...", "folder": "..."}``. Returns ``""`` if no
    rule matches, in which case the file goes to the root output_dir.
    """
    path = urlparse(url).path
    for rule in rules:
        regex = rule.get("regex")
        folder = rule.get("folder")
        if not regex or not folder:
            continue
        if re.search(regex, path):
            return folder
    return ""


def classify_topic(url: str, rules: list[dict]) -> str:
    """Map URL to a topic by matching the first rule whose regex hits its path.

    Each rule is ``{"regex": "...", "topic": "..."}``. Used to populate the
    ``tags`` list in fetched-doc frontmatter so ingest can index them.
    """
    path = urlparse(url).path
    for rule in rules:
        regex = rule.get("regex")
        topic = rule.get("topic")
        if not regex or not topic:
            continue
        if re.search(regex, path):
            return topic
    return ""


def build_doc_tags(url: str, base_tags: list[str], subfolder: str,
                   topic_rules: list[dict]) -> list[str]:
    """Compose deduped tags from fixed base + subfolder + classified topic."""
    tags: list[str] = []
    for t in (base_tags or []) + [subfolder, classify_topic(url, topic_rules)]:
        if t and t not in tags:
            tags.append(t)
    return tags


# ── 8. CLI ───────────────────────────────────────────────────────────────────

def cmd_stats(cfg: dict):
    """Print collection stats and BM25 vocab size. Exits non-zero on failure."""
    db_path = resolve_path(cfg, cfg["db_path"])
    client = QdrantClient(path=str(db_path))
    name = cfg["collection"]
    try:
        info = client.get_collection(name)
        print(f"Collection : {name}")
        print(f"Points     : {info.points_count}")
        print(f"Status     : {info.status}")
        print(f"DB path    : {db_path}")
        bm25_path = db_path / BM25_MODEL_FILE
        if bm25_path.exists():
            bm25 = BM25Encoder.load(str(bm25_path))
            print(f"BM25 vocab : {bm25.vocab_size} tokens")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        client.close()
        sys.exit(1)
    client.close()


def cmd_filters(cfg: dict):
    """List distinct values for all payload fields, streaming to bound memory."""
    db_path = resolve_path(cfg, cfg["db_path"])
    client = QdrantClient(path=str(db_path))
    name = cfg["collection"]
    fields = sorted(set(PAYLOAD_INDEXES) | {r["key"] for r in cfg.get("path_meta", [])
                                            if r.get("key")})
    vals: dict[str, set[str]] = {f: set() for f in fields}
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=name, limit=SCROLL_BATCH_SIZE, offset=offset,
            with_payload=fields, with_vectors=False,
        )
        for pt in pts:
            payload = pt.payload or {}
            for field in fields:
                v = payload.get(field)
                if v is None or v == "":
                    continue
                if isinstance(v, list):
                    vals[field].update(str(x) for x in v)
                else:
                    vals[field].add(str(v))
        if next_off is None:
            break
        offset = next_off
    for field in fields:
        if vals[field]:
            print(f"{field}: {', '.join(sorted(vals[field]))}")
    client.close()


def main():
    parser = argparse.ArgumentParser(prog="mdsearch", description="Local markdown search")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Index markdown files")
    p_ingest.add_argument("--rebuild", action="store_true", help="Drop and rebuild from scratch")

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query")
    p_search.add_argument("--mode", "-m", choices=["hybrid", "semantic", "keyword"],
                          default="hybrid")
    p_search.add_argument("--limit", "-n", type=int, default=5)
    p_search.add_argument("--filter", "-f", action="append", metavar="KEY=VALUE")
    p_search.add_argument("--json", action="store_true", dest="as_json")

    p_fetch = sub.add_parser("fetch", help="Download URLs from urls.md")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.add_argument("--dry-run", action="store_true")
    p_fetch.add_argument("--limit", type=int)
    p_fetch.add_argument("--filter", metavar="SUBSTRING")

    sub.add_parser("stats", help="Collection statistics")
    sub.add_parser("filters", help="List filter values")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    cfg_path = args.config
    if not Path(cfg_path).exists():
        fallback = Path(__file__).parent / cfg_path
        if fallback.exists():
            cfg_path = str(fallback)
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        print(f"Config not found: {cfg_path}. Run from the repo root or pass --config.")
        sys.exit(1)

    if args.cmd == "ingest":
        cmd_ingest(cfg, rebuild=args.rebuild)

    elif args.cmd == "search":
        filters: dict = {}
        if args.filter:
            allowed_keys = set(PAYLOAD_INDEXES) | {
                r["key"] for r in cfg.get("path_meta", []) if r.get("key")
            }
            for f in args.filter:
                if "=" not in f:
                    print(f"[ERROR] --filter must be KEY=VALUE, got: {f!r}",
                          file=sys.stderr)
                    sys.exit(2)
                k, v = f.split("=", 1)
                k = k.strip()
                if k not in allowed_keys:
                    print(f"[ERROR] unknown filter key {k!r}; allowed: "
                          f"{', '.join(sorted(allowed_keys))}", file=sys.stderr)
                    sys.exit(2)
                filters[k] = v
        results = search(args.query, cfg, mode=args.mode,
                         limit=args.limit, filters=filters or None)
        if args.as_json:
            out = [{"score": round(r["score"], 4),
                    "source_file": r["payload"].get("source_file", ""),
                    "title": r["payload"].get("title", ""),
                    "tags": r["payload"].get("tags", []),
                    "source_url": r["payload"].get("source_url", ""),
                    "text": (r["payload"].get("chunk_text") or "")[:300],
                    **{k: v for k, v in r["payload"].items()
                       if k not in ("chunk_text", "source_file", "title", "tags", "source_url",
                                    "chunk_index", "file_hash")}}
                   for r in results]
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"Query: {args.query} (mode: {args.mode})\nResults: {len(results)}\n")
            for i, r in enumerate(results):
                p = r["payload"]
                print(f"[{i+1}] score={r['score']:.4f}  {p.get('source_file','')}")
                print(f"     title: {p.get('title','')}")
                text = (p.get("chunk_text") or "").replace("\n", " ")[:200]
                print(f"     {text}...\n")

    elif args.cmd == "fetch":
        cmd_fetch_filter = getattr(args, "filter", None)
        cmd_fetch(cfg, force=args.force, dry_run=args.dry_run,
                  limit=args.limit, filter_str=cmd_fetch_filter)

    elif args.cmd == "stats":
        cmd_stats(cfg)

    elif args.cmd == "filters":
        cmd_filters(cfg)


if __name__ == "__main__":
    main()
