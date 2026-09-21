r"""chunk_tokens 灵敏度扫描 (600 / 800 / 1200)。

目的
----
回答一个问题: `chunk_tokens=800` 是**平台区的安全选择**, 还是**随手定的习惯值**?

方法
----
同一批 10 篇语料, 同一 parser, 只改 `chunk_tokens`, 观察:

    chunks          切出多少个 chunk
    tokens total    语料总 token (反映 embedding 成本)
    avg / median    chunk 平均大小
    at_cap          顶到上限的比例 (说明上限是否真的在起作用)

注: overlap 已于 2026-09-21 取消（见 chunker 类 docstring），
因此本脚本不再统计 overlap 相关指标。

判据
----
如果 800 在平台区, 那么 600 -> 800 -> 1200 之间:
    - chunks 数变化平缓 (不是断崖)
    - tokens total 变化平缓
如果 800 不在平台区, 会看到某个值之后曲线突然变平/突然抬升。

本脚本**不写索引、不调 API**, 纯本地计算。

用法
----
    cd pipeline
    .\.venv_rag\Scripts\python.exe scan_chunk_tokens.py
    .\.venv_rag\Scripts\python.exe scan_chunk_tokens.py --sizes 400,600,800,1000,1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.models import SourcePayload  # noqa: E402
from ingest.parsers.html import parse_html  # noqa: E402
from ingest.parsers.markdown import parse_markdown_text  # noqa: E402
from ingest.chunker import BlockAwareHierarchicalChunkBuilder  # noqa: E402

EXPERIMENTS = Path("experiments")
SIDECAR = EXPERIMENTS / "_sources"

ALL_DOCS = [
    "anthropic_contextual_retrieval",
    "anthropic_building_effective_agents",
    "anthropic_context_engineering",
    "anthropic_multi_agent_research",
    "anthropic_agent_skills",
    "anthropic_code_execution_mcp",
    "anthropic_demystifying_evals",
    "openai_scaling_storage",
    "openai_agents_api",
    "openai_model_misalignment",
]


def _payload(data: bytes, mime: str, uri: str) -> SourcePayload:
    return SourcePayload(
        source_type="url", uri=uri, final_uri=uri,
        mime_type=mime, content_type=mime,
        data=data, encoding="utf-8",
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )


def load_artifact(doc: str):
    """按 rebuild_phase0_index 的同一条路径重解析一篇, 只做一次。"""
    root = EXPERIMENTS / doc
    old = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    uri = old.get("source_uri") or ""

    sidecar = SIDECAR / f"{doc}.html"
    if sidecar.exists() and sidecar.stat().st_size > 50000:
        html = sidecar.read_text(encoding="utf-8", errors="replace")
        if "请稍候" in html[:2000]:
            raise RuntimeError(f"{doc}: sidecar 是 challenge 页")
        return parse_html(_payload(html.encode("utf-8"), "text/html", uri))

    text = (root / "document.md").read_text(encoding="utf-8")
    return parse_markdown_text(
        text, _payload(text.encode("utf-8"), "text/markdown", uri))


def scan_one(artifact, chunk_tokens: int) -> Dict[str, Any]:
    """对一篇文章按给定 chunk_tokens 切块并统计。"""
    builder = BlockAwareHierarchicalChunkBuilder(chunk_tokens=chunk_tokens)
    parents, chunks = builder.build(artifact)

    toks = [c.token_count for c in chunks]
    total = sum(toks)
    # 顶到上限: 留 2% 容差 (SentenceSplitter 不一定精确命中)
    cap = chunk_tokens * 0.98

    return {
        "chunks": len(chunks),
        "parents": len(parents),
        "tokens_total": total,
        "tokens_avg": (total / len(toks)) if toks else 0.0,
        "tokens_median": statistics.median(toks) if toks else 0,
        "tokens_max": max(toks) if toks else 0,
        "at_cap": sum(1 for t in toks if t >= cap),
        "at_cap_ratio": (sum(1 for t in toks if t >= cap) / len(toks)) if toks else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="600,800,1200",
                    help="逗号分隔的 chunk_tokens 列表")
    ap.add_argument("--json-out", default="scan_chunk_tokens.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    # 预解析 10 篇, 只做一次 (parser 与 chunk 无关)
    artifacts: List[Tuple[str, Any]] = []
    for doc in ALL_DOCS:
        try:
            artifacts.append((doc, load_artifact(doc)))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {doc}: {exc}", file=sys.stderr)

    print(f"已解析 {len(artifacts)} 篇\n")

    report: Dict[str, Any] = {"sizes": {}, "per_doc": {}}
    for size in sizes:
        agg: List[Dict[str, Any]] = []
        for doc, art in artifacts:
            st = scan_one(art, size)
            report["per_doc"].setdefault(str(size), {})[doc] = st
            agg.append(st)

        def _sum(key: str) -> float:
            return sum(a[key] for a in agg)

        def _avg(key: str) -> float:
            return _sum(key) / len(agg) if agg else 0.0

        chunks_total = int(_sum("chunks"))
        tokens_total = int(_sum("tokens_total"))
        at_cap = int(_sum("at_cap"))

        report["sizes"][str(size)] = {
            "chunk_tokens": size,
            "chunks_total": chunks_total,
            "tokens_total": tokens_total,
            "tokens_avg": _avg("tokens_avg"),
            "tokens_median_avg": _avg("tokens_median"),
            "tokens_max": max(a["tokens_max"] for a in agg),
            "at_cap": at_cap,
            "at_cap_ratio": (at_cap / chunks_total) if chunks_total else 0.0,
        }

    # ---------------- 输出 ----------------
    print("=" * 80)
    print("chunk_tokens 灵敏度扫描  (%d 篇, 固定大小, 无 overlap)" % len(artifacts))
    print("=" * 80)
    hdr = ("%8s %9s %12s %9s %8s %9s %12s" % (
        "chunk", "chunks", "tokens_total", "avg", "median", "max", "at_cap"))
    print(hdr)
    print("-" * 80)
    for size in sizes:
        r = report["sizes"][str(size)]
        print("%8d %9d %12d %9.1f %8.0f %9d %6d(%2.0f%%)" % (
            r["chunk_tokens"], r["chunks_total"],
            r["tokens_total"], r["tokens_avg"], r["tokens_median_avg"],
            r["tokens_max"], r["at_cap"], r["at_cap_ratio"] * 100))
    print("-" * 80)

    # 平台区判据: 用**相邻档**的斜率判断 (与最小档比没有意义)
    print("\n=== 灵敏度 (相邻档之间) ===")
    print("%9s %9s %11s %11s %9s" % ("from", "to", "d_chunks%", "斜率*1000", "at_cap"))
    print("-" * 56)
    prev = None
    for size in sizes:
        r = report["sizes"][str(size)]
        if prev is not None:
            d = (r["chunks_total"] - prev[1]) / prev[1] * 100
            slope = (r["chunks_total"] - prev[1]) / (size - prev[0]) * 1000
            print("%9d %9d %10.1f%% %11.2f %9d" % (
                prev[0], size, d, slope, r["at_cap"]))
        prev = (size, r["chunks_total"])
    print("-" * 56)

    # 平台定义: 相邻档 chunks 变化 < 2%
    plateau = []
    prev = None
    for size in sizes:
        c = report["sizes"][str(size)]["chunks_total"]
        if prev is not None:
            d = abs((c - prev[1]) / prev[1] * 100)
            if d < 2.0:
                plateau.append((prev[0], size, d))
        prev = (size, c)
    if plateau:
        print("\n判定: 变化 <2% 的区间 (平台区):")
        for a, b, d in plateau:
            print("   %d -> %d  (%.1f%%)" % (a, b, d))
        print("   => 平台起点约 %d" % plateau[0][0])
    else:
        print("\n判定: 扫描范围内未出现 <2% 的平台区, 建议扩大 --sizes 范围")

    # 逐篇: 哪些文章对 size 敏感
    print("\n=== 对 size 最敏感的 5 篇 (chunks 数变化幅度) ===")
    sensitivity = []
    for doc in report["per_doc"][str(sizes[0])]:
        vals = [report["per_doc"][str(s)][doc]["chunks"] for s in sizes]
        span = (max(vals) - min(vals)) / max(1, min(vals)) * 100
        sensitivity.append((span, doc, vals))
    sensitivity.sort(reverse=True)
    for span, doc, vals in sensitivity[:5]:
        print("  %-42s changed %.0f%%   %s" % (doc, span, vals))

    Path(args.json_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
