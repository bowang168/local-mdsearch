"""Unit tests for pure functions in mdsearch.py.

Covers tokenize, BM25Encoder, parse_frontmatter, extract_path_meta,
heading_aware_chunk, _split_by_size, sanitize_url_to_filename, _safe_output_path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mdsearch  # noqa: E402


# ── tokenize ────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_empty_string(self):
        assert mdsearch.tokenize("") == []

    def test_only_stopwords(self):
        assert mdsearch.tokenize("the a an is are") == []

    def test_english_words(self):
        toks = mdsearch.tokenize("kernel panic on Oracle Linux")
        assert "kernel" in toks
        assert "panic" in toks
        assert "oracle" in toks
        assert "linux" in toks

    def test_lowercases(self):
        toks = mdsearch.tokenize("KERNEL")
        assert toks == ["kernel"]

    def test_min_length_filter(self):
        # Single-char tokens are dropped.
        assert "x" not in mdsearch.tokenize("x kernel y")

    def test_hyphenated_identifier_kept(self):
        toks = mdsearch.tokenize("CVE-2024-26704 affects ol9")
        assert any("2024" in t for t in toks) or any("cve" in t for t in toks)

    def test_cjk_text(self):
        toks = mdsearch.tokenize("内核崩溃")
        assert toks  # at least one token


# ── BM25Encoder ─────────────────────────────────────────────────────────────

class TestBM25:
    def test_fit_empty(self):
        enc = mdsearch.BM25Encoder().fit([])
        assert enc.n_docs == 0
        assert enc.vocab == {}

    def test_fit_and_encode_basic(self):
        docs = [
            "kernel panic on boot",
            "network configuration on Oracle Linux",
            "kernel module signing",
        ]
        enc = mdsearch.BM25Encoder().fit(docs)
        assert enc.n_docs == 3
        assert enc.vocab_size > 0
        sv = enc.encode("kernel panic")
        assert len(sv.indices) == len(sv.values)
        assert len(sv.indices) >= 1

    def test_encode_empty_query(self):
        enc = mdsearch.BM25Encoder().fit(["hello world"])
        sv = enc.encode("")
        assert sv.indices == []
        assert sv.values == []

    def test_save_load_roundtrip(self, tmp_path):
        enc = mdsearch.BM25Encoder().fit(["alpha beta", "beta gamma", "gamma delta"])
        f = tmp_path / "bm25.json"
        enc.save(str(f))
        loaded = mdsearch.BM25Encoder.load(str(f))
        assert loaded.n_docs == enc.n_docs
        assert loaded.vocab == enc.vocab
        assert loaded.idf == enc.idf
        assert pytest.approx(loaded.avg_dl) == enc.avg_dl

    def test_unknown_token_dynamic_extend(self):
        enc = mdsearch.BM25Encoder().fit(["hello world"])
        before = enc.vocab_size
        sv = enc.encode("zzzunseenword")
        # Unseen non-stopword token gets added to vocab.
        assert enc.vocab_size > before
        assert sv.indices  # produces a non-empty vector

    def test_idf_for_rare_token_higher(self):
        docs = ["common common common", "common rare"]
        enc = mdsearch.BM25Encoder().fit(docs)
        common_idx = enc.vocab["common"]
        rare_idx = enc.vocab["rare"]
        assert enc.idf[rare_idx] > enc.idf[common_idx]


# ── parse_frontmatter ───────────────────────────────────────────────────────

class TestFrontmatter:
    def test_no_frontmatter(self):
        meta, body = mdsearch.parse_frontmatter("# Heading\nbody")
        assert meta == {}
        assert body == "# Heading\nbody"

    def test_basic(self):
        text = "---\ntitle: Test\ntags: [a, b]\n---\n\n# Body\n"
        meta, body = mdsearch.parse_frontmatter(text)
        assert meta == {"title": "Test", "tags": ["a", "b"]}
        assert body.startswith("# Body")

    def test_unterminated_frontmatter(self):
        text = "---\ntitle: Test\nno closing"
        meta, body = mdsearch.parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_invalid_yaml_returns_empty(self, capsys):
        text = "---\nthis: is: invalid: : yaml\n---\nbody"
        meta, body = mdsearch.parse_frontmatter(text)
        assert meta == {}
        assert "body" in body
        err = capsys.readouterr().err
        assert "frontmatter" in err.lower() or "yaml" in err.lower()

    def test_yaml_doc_separator_inside_value_does_not_truncate(self):
        # A "---" inside a multi-line scalar must not be picked up as the closer.
        text = ("---\ntitle: Test\nnotes: |\n  line1\n  ---\n  line3\n---\nbody\n")
        meta, body = mdsearch.parse_frontmatter(text)
        assert meta.get("title") == "Test"
        assert body.startswith("body")

    def test_non_dict_yaml_yields_empty_meta(self):
        text = "---\n- just\n- a\n- list\n---\nbody"
        meta, _ = mdsearch.parse_frontmatter(text)
        assert meta == {}


# ── extract_path_meta ───────────────────────────────────────────────────────

class TestExtractPathMeta:
    def test_basic_capture(self):
        rules = [{"regex": r"/ol(\d+)/", "key": "ol_version", "format": "ol{1}"}]
        assert mdsearch.extract_path_meta("docs/ol9/foo.md", rules) == {"ol_version": "ol9"}

    def test_no_format_uses_capture_directly(self):
        rules = [{"regex": r"/(uek|olcne)/", "key": "product"}]
        assert mdsearch.extract_path_meta("docs/uek/x.md", rules) == {"product": "uek"}

    def test_no_match_returns_empty(self):
        rules = [{"regex": r"/ol(\d+)/", "key": "ol_version"}]
        assert mdsearch.extract_path_meta("docs/other/x.md", rules) == {}

    def test_regex_without_capture_group_warns_and_skips(self, capsys):
        rules = [{"regex": r"/ol\d+/", "key": "ol_version"}]
        out = mdsearch.extract_path_meta("docs/ol9/x.md", rules)
        assert out == {}
        assert "capture group" in capsys.readouterr().err

    def test_missing_key_or_regex_skipped(self, capsys):
        rules = [{"regex": r"/ol(\d+)/"}]  # no key
        out = mdsearch.extract_path_meta("docs/ol9/x.md", rules)
        assert out == {}
        assert "regex/key" in capsys.readouterr().err


# ── chunking ────────────────────────────────────────────────────────────────

class TestChunking:
    def test_split_by_size_short(self):
        chunks = mdsearch._split_by_size("short text", 1500)
        assert chunks == ["short text"]

    def test_split_by_size_splits_long_text(self):
        text = "\n".join("line " + str(i) for i in range(500))
        chunks = mdsearch._split_by_size(text, 200)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 400  # generous slack: line boundaries can overshoot

    def test_heading_aware_no_headings(self):
        chunks = mdsearch.heading_aware_chunk("just some text without headings", 1500)
        assert chunks == ["just some text without headings"]

    def test_heading_aware_multiple_sections(self):
        text = "# Top\nintro\n## Sub\nbody\n## Sub2\nmore\n"
        chunks = mdsearch.heading_aware_chunk(text, 1500)
        assert len(chunks) >= 2
        # Breadcrumb prefix appears for child sections.
        assert any(c.startswith("[") for c in chunks)

    def test_heading_aware_oversize_section_subdivided(self):
        big = "x\n" * 2000  # ~4000 chars
        text = "# Section\n" + big
        chunks = mdsearch.heading_aware_chunk(text, 500)
        assert len(chunks) > 1


# ── sanitize_url_to_filename / _safe_output_path ────────────────────────────

class TestUrlSanitization:
    def test_basic_url(self):
        assert mdsearch.sanitize_url_to_filename(
            "https://example.com/foo/bar.html"
        ) == "foo_bar"

    def test_root_url_yields_index(self):
        assert mdsearch.sanitize_url_to_filename("https://example.com/") == "index"

    def test_strips_dangerous_chars(self):
        out = mdsearch.sanitize_url_to_filename("https://example.com/%2F..%2Fetc%2Fpasswd")
        assert ".." not in out
        assert "/" not in out
        assert "\\" not in out

    def test_truncates_long_path(self):
        long = "https://example.com/" + "a" * 1000
        out = mdsearch.sanitize_url_to_filename(long)
        assert len(out) <= mdsearch.MAX_FILENAME_LEN

    def test_safe_output_path_within_dir(self, tmp_path):
        out = mdsearch._safe_output_path("https://example.com/foo", tmp_path)
        assert out is not None
        assert tmp_path in out.parents
        assert out.name == "foo.md"

    def test_safe_output_path_pathological_input_still_contained(self, tmp_path):
        # Even adversarial inputs must resolve inside tmp_path.
        out = mdsearch._safe_output_path("https://x.com/../../../etc/passwd", tmp_path)
        assert out is not None
        assert str(out).startswith(str(tmp_path))


# ── load_config ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(body, encoding="utf-8")
        return p

    def test_valid_config(self, tmp_path):
        p = self._write(tmp_path, (
            "collection: c\n"
            "db_path: .db/\n"
            "embedding:\n"
            "  url: http://localhost:11434/api/embed\n"
            "  model: m\n"
            "  dim: 1024\n"
        ))
        cfg = mdsearch.load_config(str(p))
        assert cfg["collection"] == "c"
        assert cfg["_base"] == p.parent

    def test_missing_top_field_exits(self, tmp_path):
        p = self._write(tmp_path, "collection: c\ndb_path: .db/\n")
        with pytest.raises(SystemExit):
            mdsearch.load_config(str(p))

    def test_missing_embedding_field_exits(self, tmp_path):
        p = self._write(tmp_path, (
            "collection: c\ndb_path: .db/\nembedding:\n  url: u\n  model: m\n"
        ))
        with pytest.raises(SystemExit):
            mdsearch.load_config(str(p))

    def test_non_mapping_root_exits(self, tmp_path):
        p = self._write(tmp_path, "- a\n- b\n")
        with pytest.raises(SystemExit):
            mdsearch.load_config(str(p))


# ── classify_url ────────────────────────────────────────────────────────────

class TestClassifyUrl:
    rules = [
        {"regex": r"/uek/", "folder": "uek"},
        {"regex": r"/oracle-linux/9/", "folder": "ol9"},
        {"regex": r"/oracle-linux/", "folder": "shared"},
    ]

    def test_first_match_wins(self):
        assert mdsearch.classify_url(
            "https://docs.oracle.com/en/operating-systems/oracle-linux/9/network/",
            self.rules,
        ) == "ol9"

    def test_falls_back_to_later_rule(self):
        assert mdsearch.classify_url(
            "https://docs.oracle.com/en/operating-systems/oracle-linux/openssh/",
            self.rules,
        ) == "shared"

    def test_uek_branch(self):
        assert mdsearch.classify_url(
            "https://docs.oracle.com/en/operating-systems/uek/8/relnotes8.0/",
            self.rules,
        ) == "uek"

    def test_no_rules_yields_empty(self):
        assert mdsearch.classify_url("https://example.com/foo", []) == ""

    def test_no_match_yields_empty(self):
        assert mdsearch.classify_url("https://other.com/bar", self.rules) == ""

    def test_skips_malformed_rule(self):
        bad = [{"regex": "/x/"}, {"folder": "y"}, {"regex": r"/z/", "folder": "zz"}]
        assert mdsearch.classify_url("https://x.com/z/", bad) == "zz"


# ── classify_topic / build_doc_tags ─────────────────────────────────────────

class TestClassifyTopic:
    rules = [
        {"regex": r"/firewall", "topic": "security"},
        {"regex": r"/network", "topic": "networking"},
    ]

    def test_first_match(self):
        assert mdsearch.classify_topic(
            "https://x.com/firewall/zone", self.rules) == "security"

    def test_no_match(self):
        assert mdsearch.classify_topic("https://x.com/other", self.rules) == ""

    def test_empty_rules(self):
        assert mdsearch.classify_topic("https://x.com/firewall", []) == ""


class TestBuildDocTags:
    rules = [{"regex": r"/network", "topic": "networking"}]

    def test_combines_base_subfolder_topic(self):
        out = mdsearch.build_doc_tags(
            "https://x.com/network/foo",
            ["oracle-linux"], "ol9", self.rules)
        assert out == ["oracle-linux", "ol9", "networking"]

    def test_dedupes(self):
        out = mdsearch.build_doc_tags(
            "https://x.com/network", ["ol9"], "ol9", self.rules)
        assert out == ["ol9", "networking"]

    def test_skips_empty(self):
        out = mdsearch.build_doc_tags(
            "https://x.com/other", ["base"], "", [])
        assert out == ["base"]

    def test_no_base(self):
        out = mdsearch.build_doc_tags(
            "https://x.com/network", [], "ol9", self.rules)
        assert out == ["ol9", "networking"]


# ── _safe_output_path with subfolder ────────────────────────────────────────

class TestSafeOutputPathSubfolder:
    def test_subfolder_appended(self, tmp_path):
        out = mdsearch._safe_output_path(
            "https://docs.oracle.com/en/operating-systems/oracle-linux/9/network/",
            tmp_path, subfolder="ol9")
        assert out is not None
        assert out.parent == tmp_path / "ol9"

    def test_subfolder_with_dotdot_is_sanitized(self, tmp_path):
        out = mdsearch._safe_output_path(
            "https://x.com/foo", tmp_path, subfolder="../escape")
        assert out is not None
        # `..` is collapsed to `_`, so target stays under tmp_path
        assert str(out).startswith(str(tmp_path))


# ── build_filter ────────────────────────────────────────────────────────────

class TestBuildFilter:
    def test_empty_returns_none(self):
        assert mdsearch.build_filter(None) is None
        assert mdsearch.build_filter({}) is None

    def test_non_empty_returns_filter(self):
        f = mdsearch.build_filter({"ol_version": "ol9"})
        assert f is not None
        assert len(f.must) == 1
