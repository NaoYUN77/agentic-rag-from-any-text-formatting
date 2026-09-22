r"""用新 parser 重建 Phase 0 索引 (dense + sparse)。

流程
----
    读 10 篇源 (sidecar HTML / document.md 回退)
      -> 新 parser 直产 DocumentBlock
      -> BlockAwareHierarchicalChunkBuilder
      -> 写 experiments_rebuilt/<doc>/{artifact,parents,chunks}
      -> ingest.qdrant_indexer 写入 Qdrant

关键取舍
--------
**embedding 后端决定这次重建的"真值等级"**:

    backend=qwen   真语义向量。需要 DASHSCOPE_API_KEY。当前 .dashscope_key
                   是占位符, 所以默认跑不了 —— 此时**检索指标无法得出**。
    backend=local  哈希向量 (HashingEmbeddings)。零依赖、确定性, 但**没有语义**,
                   dense 检索退化为"字符 n-gram 重合度"。
                   → 只用它验证**管线通不通**, 绝不能拿它的指标当检索质量结论。

所以本脚本会把 backend 写进 manifest, 并在结尾明确声明本次指标是否可用于评估。

用法
----
    cd pipeline

    # 结构性重建 + 本地哈希向量 (验证管线, 指标不可用于评估)
    .\.venv_rag\Scripts\python.exe rebuild_phase0_index.py --backend local

    # 真语义重建 (需要 key)
    $env:DASHSCOPE_API_KEY="sk-..."
    .\.venv_rag\Scripts\python.exe rebuild_phase0_index.py --backend qwen
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _payload(data: bytes, mime: str, uri: str) -> SourcePayload:
    return SourcePayload(
        source_type="url", uri=uri, final_uri=uri,
        mime_type=mime, content_type=mime,
        data=data, encoding="utf-8",
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )


def build_export(
    doc: str,
    out_root: Path,
    chunk_tokens: int = 800,
    parent_tokens: int = 1600,
    strip_headings: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """重跑一篇, 写出 artifact/parents/chunks, 返回 (parser, 统计)。

    chunk_tokens 显式传入 (此前硬编码 800) —— 这样扫描不同切块参数时
    不必改代码, 也避免"CLI 与库不一致"的口径问题。
    strip_headings 同理: 标题只作 metadata 不进正文, 见 chunker._render_block。

    注: overlap 已于 2026-09-21 取消（见 chunker 类 docstring），
    因此这里不再有 overlap_tokens 参数。
    """
    root = EXPERIMENTS / doc
    old = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    uri = old.get("source_uri") or ""

    sidecar = SIDECAR / f"{doc}.html"
    if sidecar.exists() and sidecar.stat().st_size > 50000:
        html = sidecar.read_text(encoding="utf-8", errors="replace")
        if "请稍候" in html[:2000]:
            raise RuntimeError(f"{doc}: sidecar 是 challenge 页")
        artifact = parse_html(_payload(html.encode("utf-8"), "text/html", uri))
    else:
        text = (root / "document.md").read_text(encoding="utf-8")
        artifact = parse_markdown_text(
            text, _payload(text.encode("utf-8"), "text/markdown", uri))

    builder = BlockAwareHierarchicalChunkBuilder(
        chunk_tokens=chunk_tokens,
        parent_tokens=parent_tokens,
        strip_headings=strip_headings)
    parents, chunks = builder.build(artifact)

    out = out_root / doc
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifact.json").write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "parents.json").write_text(
        json.dumps([p.to_dict() for p in parents], ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    stats = {
        "blocks": len(artifact.blocks),
        "parents": len(parents),
        "chunks": len(chunks),
        "tokens": sum(c.token_count for c in chunks),
    }
    return artifact.parser, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local", "qwen"])
    ap.add_argument("--chunk-tokens", type=int, default=800,
                    help="切块上限 (默认 800)。此前硬编码, 现可扫参数。")
    ap.add_argument("--parent-tokens", type=int, default=1600,
                    help="单个 parent 的 token 上限 (默认 1600 = 2x chunk_tokens)。"
                         "主要兜住'无 heading 文档整篇落进一个分组'; 0 表示不限制。")
    ap.add_argument("--keep-headings", dest="strip_headings", action="store_false",
                    help="把标题渲染进 chunk 正文 (旧行为)。"
                         "默认剥离标题, 只作 metadata —— 见 chunker._render_block。")
    ap.add_argument("--out-dir", default="experiments_rebuilt")
    ap.add_argument("--qdrant-path", default="qdrant_data")
    ap.add_argument("--dense-collection", default="phase0_dense")
    ap.add_argument("--sparse-collection", default="phase0_sparse")
    ap.add_argument("--artifact-dir", default="index_artifacts/phase0_rebuilt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    if out_root.exists() and not args.dry_run:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"重建 Phase 0 导出产物  (backend={args.backend})")
    print("=" * 80)
    parsers: Dict[str, str] = {}
    totals = collections.Counter()
    for doc in ALL_DOCS:
        parser, stats = build_export(
            doc, out_root,
            chunk_tokens=args.chunk_tokens,
            parent_tokens=args.parent_tokens,
        )
        parsers[doc] = parser
        for k, v in stats.items():
            totals[k] += v
        print(f"  {doc:<42} {parser:<16} blocks={stats['blocks']:<4} "
              f"parents={stats['parents']:<3} chunks={stats['chunks']:<3}")
    print()
    print(f"  合计: blocks={totals['blocks']} parents={totals['parents']} "
          f"chunks={totals['chunks']} tokens={totals['tokens']}")
    print()

    if args.dry_run:
        print("[dry-run] 不写索引")
        return 0

    print("=" * 80)
    print("写入 Qdrant")
    print("=" * 80)

    from ingest.qdrant_indexer import write_index, build_index_plan, load_export_bundle
    from embeddings import build_embeddings

    bundles = [load_export_bundle(str(out_root / d)) for d in ALL_DOCS]
    plan = build_index_plan(bundles)
    print(f"  bundles  : {len(bundles)}")
    print(f"  chunks   : {plan.chunks_loaded}")
    print(f"  dense    : {plan.dense_count}")
    print(f"  sparse   : {plan.sparse_count}")
    print(f"  parents  : {plan.parent_count}")
    print()

    embeddings = build_embeddings(args.backend)
    t0 = time.time()
    manifest = write_index(
        plan=plan,
        embeddings=embeddings,
        qdrant_path=args.qdrant_path,
        dense_collection=args.dense_collection,
        sparse_collection=args.sparse_collection,
        artifact_dir=args.artifact_dir,
        rebuild=True,
        source="phase0_rebuilt_html_dom",
    )
    elapsed = time.time() - t0

    manifest["embedding_backend"] = args.backend
    manifest["chunk_tokens"] = args.chunk_tokens
    manifest["parent_tokens"] = args.parent_tokens
    # 显式记 0: overlap 已于 2026-09-21 取消（见 chunker 类 docstring）。
    # 保留该字段是为了让产物可追溯 —— 否则"这份索引到底有没有 overlap"
    # 只能靠翻代码判断，而旧索引里这个值曾是 400。
    manifest["overlap_tokens"] = 0
    manifest["parsers"] = parsers
    manifest["rebuild_seconds"] = round(elapsed, 1)
    (Path(args.artifact_dir) / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print()
    if args.backend == "local":
        print("⚠ backend=local: dense 用的是哈希向量, **没有语义**。")
        print("  本次结果只能证明管线连通, 不能用于回答'检索指标变了吗'。")
    else:
        print("✓ backend=qwen: 真语义向量, 指标可用于评估。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
