r"""Phase 0 语料: 新 parser 重建 + 新旧结构对照。

产出一份可复现的对照报告, 回答"去掉 Markdown 层值不值"。

源数据策略 (两层)
-----------------
1. **7 篇 Anthropic**: 用 playwright-cli 抓真实 HTML 到
   `experiments/_sources/<doc>.html`, 走 `html_readability`。
   (OpenAI 三篇被 Cloudflare 挡在 403/challenge 页, 抓不到真实 HTML。)

2. **3 篇 OpenAI**: 无法重抓源 HTML, 退而用旧产物的 `document.md`
   (它是当年从真实 HTML 抽出来的, 内容保真), 走 `plain_text`。
   注意: 这条路径下 `url` / `caption` / `links` 拿不到 —— 因为
   Markdown 里本来就没有这些信息。**这本身就是"去掉 Markdown 层"
   这个论点的直接证据。**

输出
----
    experiments_rebuilt/compare.json     完整机读对照
    experiments_rebuilt/report.md        人读报告

用法
----
    cd pipeline
    .\.venv_rag\Scripts\python.exe rebuild_phase0_corpus.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.models import SourcePayload  # noqa: E402
from ingest.parsers.html import parse_html  # noqa: E402
from ingest.parsers.markdown import parse_markdown_text  # noqa: E402
from ingest.chunker import BlockAwareHierarchicalChunkBuilder  # noqa: E402

EXPERIMENTS = Path("experiments")
SIDECAR = EXPERIMENTS / "_sources"

ANTHROPIC = [
    "anthropic_contextual_retrieval",
    "anthropic_building_effective_agents",
    "anthropic_context_engineering",
    "anthropic_multi_agent_research",
    "anthropic_agent_skills",
    "anthropic_code_execution_mcp",
    "anthropic_demystifying_evals",
]
OPENAI = [
    "openai_scaling_storage",
    "openai_agents_api",
    "openai_model_misalignment",
]
ALL_DOCS = ANTHROPIC + OPENAI


# --------------------------------------------------------------------------

def _payload(data: bytes, mime: str, uri: str) -> SourcePayload:
    return SourcePayload(
        source_type="url", uri=uri, final_uri=uri,
        mime_type=mime, content_type=mime,
        data=data, encoding="utf-8",
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )


def side_of(doc: str) -> str:
    return "anthropic" if doc in ANTHROPIC else "openai"


def old_stats(doc: str) -> Dict[str, Any]:
    root = EXPERIMENTS / doc
    art = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in (root / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    parents = json.loads((root / "parents.json").read_text(encoding="utf-8"))
    blocks = art.get("blocks") or []
    head = [b for b in blocks if b.get("type") == "heading"]
    depths = [len(b.get("section_path") or []) for b in blocks]

    return {
        "parser": art.get("parser"),
        "title": art.get("title"),
        "blocks": len(blocks),
        "block_types": dict(collections.Counter(b.get("type") for b in blocks)),
        "headings": len(head),
        "heading_levels": dict(collections.Counter(
            float((b.get("metadata") or {}).get("level") or 0) for b in head)),
        "section_depth_max": max(depths) if depths else 0,
        "section_depth_dist": dict(collections.Counter(depths)),
        "parents": len(parents),
        "chunks": len(chunks),
        "chunk_tokens_total": sum(int(c.get("token_count") or 0) for c in chunks),
        "chunk_chars_total": sum(int(c.get("char_count") or 0) for c in chunks),
        # 旧链路根本没有这些字段
        "url_filled": 0,
        "code_total": sum(1 for b in blocks if b.get("type") == "code"),
        "code_with_language": sum(
            1 for b in blocks if b.get("type") == "code" and b.get("code_language")),
        "tables": sum(1 for b in blocks if b.get("type") == "table"),
        "with_caption": sum(1 for b in blocks if b.get("caption")),
        "with_links": sum(1 for b in blocks if (b.get("metadata") or {}).get("links")),
        "payload_bytes": None,
    }


def new_stats(doc: str) -> Dict[str, Any]:
    """跑新链路。返回 status=ok / no_source。"""
    root = EXPERIMENTS / doc
    art_old = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    uri = art_old.get("source_uri") or ""

    sidecar = SIDECAR / f"{doc}.html"
    if sidecar.exists() and sidecar.stat().st_size > 50000:
        html = sidecar.read_text(encoding="utf-8", errors="replace")
        # 真实性校验: challenge 页 title 是 "请稍候…"
        if "请稍候" in html[:2000]:
            return {"status": "no_source", "reason": "sidecar 是 Cloudflare challenge 页"}
        source_how = f"html (playwright, {sidecar.stat().st_size} B)"
        artifact = parse_html(_payload(html.encode("utf-8"), "text/html", uri))
    else:
        md = root / "document.md"
        if not md.exists():
            return {"status": "no_source", "reason": "无 sidecar HTML, 也无 document.md"}
        text = md.read_text(encoding="utf-8")
        # 注意: document.md 是 **Markdown**, 不是纯文本。
        # 所以这里用 markdown 解析器 —— 用它才能让"旧 vs 新"对比只差
        # 在新 parser 能力上, 而不是差在"拿错了格式解析器"。
        # (OpenAI 三篇被 Cloudflare 403, 拿不到原始 HTML, 这是唯一可选项。
        #  代价: 这条路径下 url/caption/links 必然为空, 因为 Markdown
        #  里本来就没有这些信息 —— 这恰好是"去掉 Markdown 层"的论据。)
        source_how = f"document.md 回退 (markdown parser, {len(text)} B)"
        artifact = parse_markdown_text(
            text, _payload(text.encode("utf-8"), "text/markdown", uri))

    builder = BlockAwareHierarchicalChunkBuilder(chunk_tokens=800)
    parents, chunks = builder.build(artifact)

    blocks = artifact.blocks
    t = collections.Counter(b.type for b in blocks)
    head = [b for b in blocks if b.type == "heading"]
    depths = [len(b.section_path or []) for b in blocks]

    from ingest.qdrant_indexer import _chunk_payload

    art_dict = artifact.to_dict()
    payload_bytes = sum(
        len(json.dumps(_chunk_payload(c.to_dict(), art_dict, i), ensure_ascii=False).encode("utf-8"))
        for i, c in enumerate(chunks)
    )

    return {
        "status": "ok",
        "source_how": source_how,
        "parser": artifact.parser,
        "title": artifact.title,
        "blocks": len(blocks),
        "block_types": dict(t),
        "headings": t.get("heading", 0),
        "heading_levels": dict(collections.Counter(
            float((b.metadata or {}).get("level") or 0) for b in head)),
        "section_depth_max": max(depths) if depths else 0,
        "section_depth_dist": dict(collections.Counter(depths)),
        "parents": len(parents),
        "chunks": len(chunks),
        "chunk_tokens_total": sum(c.token_count for c in chunks),
        "chunk_chars_total": sum(c.char_count for c in chunks),
        "url_filled": sum(1 for b in blocks if b.url),
        "code_total": t.get("code", 0),
        "code_with_language": sum(1 for b in blocks if b.type == "code" and b.code_language),
        "tables": t.get("table", 0),
        "with_caption": sum(1 for b in blocks if b.caption),
        "with_links": sum(1 for b in blocks if (b.metadata or {}).get("links")),
        "payload_bytes": payload_bytes,
    }


ROWS = [
    ("blocks", "块数"),
    ("headings", "标题数"),
    ("code_total", "代码块"),
    ("code_with_language", "代码带语言"),
    ("tables", "表格块"),
    ("with_caption", "带 caption"),
    ("with_links", "带 links"),
    ("url_filled", "block 带 url"),
    ("parents", "parents"),
    ("chunks", "chunks"),
    ("chunk_tokens_total", "chunk tokens"),
    ("payload_bytes", "payload 字节"),
]


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:,}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments_rebuilt")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    docs = [args.only] if args.only else ALL_DOCS
    results: Dict[str, Any] = {}

    for doc in docs:
        o = old_stats(doc)
        n = new_stats(doc)
        results[doc] = {"side": side_of(doc), "old": o, "new": n}

    ok = {d: r for d, r in results.items() if r["new"].get("status") == "ok"}

    # ---------- 逐篇 ----------
    for doc, r in results.items():
        o, n = r["old"], r["new"]
        print("=" * 88)
        print(f"{doc}   [{r['side']}]")
        print("=" * 88)
        print(f"  old parser : {o['parser']}")
        if n.get("status") != "ok":
            print(f"  [跳过] {n.get('reason')}")
            print()
            continue
        print(f"  new parser : {n['parser']}")
        print(f"  new source : {n['source_how']}")
        print()
        print(f"  {'指标':<16}{'旧':>10}{'新':>10}{'变化':>14}")
        print("  " + "-" * 50)
        for key, label in ROWS:
            a, b = o.get(key), n.get(key)
            if a is None and b is None:
                continue
            if isinstance(a, int) and isinstance(b, int):
                d = b - a
                pct = f" ({(d / a * 100):+.0f}%)" if a else ""
                print(f"  {label:<16}{a:>10,}{b:>10,}{f'{d:+,}' + pct:>14}")
            else:
                print(f"  {label:<16}{fmt(a):>10}{fmt(b):>10}")
        print()
        print(f"  old types : {o['block_types']}")
        print(f"  new types : {n['block_types']}")
        print()

    # ---------- 汇总 ----------
    print("=" * 88)
    print("汇总 (仅统计能重跑的文档)")
    print("=" * 88)
    print(f"  可重跑: {len(ok)} / {len(results)}")
    print()
    print(f"  {'指标':<16}{'旧':>10}{'新':>10}{'变化':>14}")
    print("  " + "-" * 50)
    for key, label in ROWS:
        a = sum(r["old"].get(key) or 0 for r in ok.values())
        b = sum(r["new"].get(key) or 0 for r in ok.values())
        if a == 0 and b == 0:
            continue
        d = b - a
        pct = f" ({(d / a * 100):+.0f}%)" if a else ""
        print(f"  {label:<16}{a:>10,}{b:>10,}{f'{d:+,}' + pct:>14}")
    print()

    names_ok = [d for d in ANTHROPIC if d in ok]
    names_fb = [d for d in OPENAI if d in ok]
    print(f"  真实 HTML 路径 : {len(names_ok)} 篇 (Anthropic, 可拿到 url/caption/links)")
    print(f"  md 回退路径    : {len(names_fb)} 篇 (OpenAI, 被 Cloudflare 403)")
    print()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {out / 'compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
