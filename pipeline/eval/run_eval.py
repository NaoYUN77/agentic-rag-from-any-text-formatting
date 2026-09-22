r"""评估主入口: 分阶段跑检索并出报告。

用法
----
    cd pipeline
    .\.venv_rag\Scripts\python.exe -m eval.run_eval \
        --qrels eval/qrels/phase0_url_draft.jsonl \
        --chunks experiments_rebuilt_qwen/*/chunks.jsonl \
        --artifact-dir index_artifacts/phase0_qwen \
        --dense-collection phase0_qwen_dense \
        --sparse-collection phase0_qwen_sparse \
        --out-dir eval/runs

⚠️ `--chunks` 必须传**每个文档各自的** chunks.jsonl (它是 nargs="+")。
   不要传 `index_artifacts/<x>/chunks.jsonl` —— 那是索引侧的合并文件,
   **不含 fragments**, 会让 fragment 索引为 0、gold 全部解析失败
   (实测报 `GoldResolveError: 文本反查无结果`, 日志显示"跳过空文本: 644")。

设计要点
--------
1. **一次检索, 五阶段复用**。不为每个指标重跑检索 —— dense/sparse 只调一次,
   融合与重排在其结果上做。这样既快又保证各阶段用的是同一批候选。

2. **分阶段上报**: dense / sparse / union / rrf / rerank。
   目的是看出"每一层带来了多少增量", 而不是只看最终数字。

3. **每次运行落 manifest.json**。索引一变历史指标就不可比, 必须留指纹。

注意: 运行前必须停掉 FastAPI —— Qdrant local 模式会锁 qdrant_data。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# 允许 `python -m eval.run_eval` 与直接运行两种方式
_HERE = Path(__file__).resolve().parent
_PIPELINE = _HERE.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from eval.metrics import aggregate, score_query
from eval.qrels import QrelsItem, load_qrels, summarize
from eval.resolve import (
    GoldResolveError,
    build_fragment_index_multi,
    chunks_sha256,
    resolve_all,
)

STAGES = ["dense", "sparse", "union", "rrf", "rerank"]


def required_stages(stages: List[str]) -> set:
    """为了跑出请求的阶段，实际【必须计算】的底层检索阶段。

    `union` / `rrf` / `rerank` 都拿 dense + sparse 当输入，所以只请求下游阶段时，
    底层阶段也必须算出来 —— 不管它们有没有出现在 `--stages` 里。

    ⚠️ 不做这件事会**静默**出错：`--stages rrf` 会拿两个空列表做融合，
    指标全 0 却不报错，看起来像「模型效果差」。实测踩过这个坑。
    """
    needed = set(stages)
    if {"union", "rrf", "rerank"} & set(stages):
        needed |= {"dense", "sparse"}
    return needed


def _chunk_id_of(hit: Any) -> str:
    return str((hit.payload or {}).get("chunk_id") or "")


def _union_order(dense: Sequence[Any], sparse: Sequence[Any]) -> List[str]:
    """union 阶段: 按"两路都出现优先、再按各自排名"排序。

    这不是生产用的融合策略(RRF 才是), 只用来回答"两路的并集覆盖了多少"。
    """
    seen: Dict[str, int] = {}
    for rank, h in enumerate(dense, 1):
        cid = _chunk_id_of(h)
        if cid:
            seen[cid] = seen.get(cid, 0) + rank
    for rank, h in enumerate(sparse, 1):
        cid = _chunk_id_of(h)
        if cid:
            seen[cid] = seen.get(cid, 0) + rank
    return [cid for cid, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def evaluate(
    items: Sequence[QrelsItem],
    gold_map: Dict[str, Set[str]],
    retriever: Any,
    reranker: Any,
    candidate_k: int = 20,
    final_top_k: int = 5,
    rrf_k: int = 60,
    stages: Sequence[str] = STAGES,
    ks: Sequence[int] = (5, 10, 20),
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """跑评估。返回 (逐条结果, 按阶段聚合的指标)。"""
    from hybrid_retriever import HybridRetriever

    per_query: List[Dict[str, Any]] = []
    by_stage: Dict[str, List[Dict[str, Optional[float]]]] = {s: [] for s in stages}
    by_stage_cat: Dict[str, Dict[str, List[Dict[str, Optional[float]]]]] = {
        s: {} for s in stages
    }

    for i, item in enumerate(items, 1):
        q = item.query
        gold = gold_map.get(item.query_id, set())
        row: Dict[str, Any] = {
            "query_id": item.query_id,
            "query": q,
            "category": item.category,
            "split": item.split,
            "gold_size": len(gold),
            "gold_chunk_ids": sorted(gold),
            "stages": {},
        }

        # dense / sparse 是 union / rrf / rerank 的输入 —— 见 required_stages()。
        needed = required_stages(stages)
        t0 = time.perf_counter()
        dense = (
            retriever.dense_search(q, candidate_k)
            if "dense" in needed else []
        )
        sparse, _, unknown = (
            retriever.sparse_search(q, candidate_k)
            if "sparse" in needed else ([], [], [])
        )
        fused = (
            HybridRetriever.rrf_fuse([dense, sparse], k=rrf_k, limit=candidate_k)
            if ("rrf" in stages or "rerank" in stages) else []
        )
        reranked = []
        if "rerank" in stages and reranker is not None and fused:
            try:
                reranked = reranker.rerank(q, fused, top_k=final_top_k)
            except Exception as exc:      # 单条失败不该中断整轮
                row["rerank_error"] = str(exc)[:200]
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        stage_ids: Dict[str, List[str]] = {
            "dense": [_chunk_id_of(h) for h in dense],
            "sparse": [_chunk_id_of(h) for h in sparse],
            "union": _union_order(dense, sparse),
            "rrf": [_chunk_id_of(h) for h in fused],
            "rerank": [_chunk_id_of(h) for h in reranked],
        }

        for stage in stages:
            ids = [x for x in stage_ids.get(stage, []) if x]
            scores = score_query(ids, gold, ks=ks)
            row["stages"][stage] = {
                "top_ids": ids[:10],
                "scores": scores,
            }
            by_stage[stage].append(scores)
            by_stage_cat[stage].setdefault(item.category, []).append(scores)

        row["unknown_terms"] = unknown[:10]
        row["elapsed_ms"] = elapsed_ms
        per_query.append(row)

        if i % 10 == 0 or i == len(items):
            print("  已评估 %d/%d" % (i, len(items)), flush=True)

    summary: Dict[str, Dict[str, float]] = {}
    for stage in stages:
        summary[stage] = aggregate(by_stage[stage])
        for cat, rows in by_stage_cat[stage].items():
            summary[f"{stage}::{cat}"] = aggregate(rows)
    return per_query, summary


def build_manifest(
    chunks_paths: List[Path],
    artifact_dir: Path,
    dense_collection: str,
    sparse_collection: str,
    client: Any,
) -> Dict[str, Any]:
    """索引快照 —— 索引一变历史指标就不可比。"""
    manifest: Dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunks_jsonl": [str(p) for p in chunks_paths],
        # 逐份文件单独记 hash: 索引重建后任何一份变了都可见
        "chunks_sha256": {str(p): chunks_sha256(p) for p in chunks_paths},
        "artifact_dir": str(artifact_dir),
        "dense_collection": dense_collection,
        "sparse_collection": sparse_collection,
    }
    try:
        d = client.get_collection(dense_collection)
        manifest["dense_points"] = getattr(d, "points_count", None)
    except Exception as exc:
        manifest["dense_error"] = str(exc)[:200]
    try:
        s = client.get_collection(sparse_collection)
        manifest["sparse_points"] = getattr(s, "points_count", None)
    except Exception as exc:
        manifest["sparse_error"] = str(exc)[:200]
    mf = artifact_dir / "index_manifest.json"
    if mf.exists():
        try:
            manifest["index_manifest"] = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return manifest


def write_report(
    out_dir: Path,
    summary: Dict[str, Dict[str, float]],
    per_query: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    qrels_summary: Dict[str, Any],
    resolve_notes: List[str],
    stages: Sequence[str] = STAGES,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    with io.open(out_dir / "per_query.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines: List[str] = []
    lines.append("# 检索评估报告\n")
    lines.append("生成时间: %s\n" % manifest.get("built_at"))
    lines.append("## 配置\n")
    lines.append("```text")
    for k, v in config.items():
        lines.append("%-18s %s" % (k, v))
    lines.append("```\n")
    lines.append("## 索引快照\n")
    lines.append("```text")
    cs = manifest.get("chunks_sha256")
    if isinstance(cs, dict):
        # 多份 chunks.jsonl: 报告里显示合并指纹, 完整逐份 hash 在 manifest.json
        combined = "|".join(cs.values())
        lines.append("chunks_sha256      %s (x%d 份)" % (combined[:16], len(cs)))
    else:
        lines.append("chunks_sha256      %s" % str(cs)[:16])
    lines.append("dense_points       %s" % manifest.get("dense_points"))
    lines.append("sparse_points      %s" % manifest.get("sparse_points"))
    lines.append("```\n")
    lines.append("## 评估集\n")
    lines.append("```text")
    for k, v in qrels_summary.items():
        lines.append("%-18s %s" % (k, v))
    lines.append("```\n")

    lines.append("## 分阶段指标\n")
    lines.append("| 阶段 | hit@5 | recall@5 | recall@20 | MRR@20 | nDCG@5 | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for stage in stages:
        s = summary.get(stage) or {}
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %d |" % (
            stage,
            s.get("hit@5", 0), s.get("recall@5", 0), s.get("recall@20", 0),
            s.get("mrr@20", 0), s.get("ndcg@5", 0), int(s.get("hit@5__n", 0)),
        ))
    skipped = [s for s in STAGES if s not in stages]
    if skipped:
        lines.append("")
        lines.append("> 本次未运行的阶段: %s" % ", ".join(skipped))
    lines.append("")

    # ---- 失败率告警 ----
    # 为什么必须有这一段 (2026-09-20 实测踩到):
    # rerank 单条失败会被 try/except 吞掉, 该 query 的 rerank 阶段记 0 分。
    # 于是**配额耗尽(403)看起来像"rerank 效果极差"**:
    # 实测 candidate_k=50 时 26/39 条 403 → hit@5 显示 0.294;
    # candidate_k=100 时 39/39 条 403 → 显示 0.000, 而真实情况是"根本没跑"。
    # 不把失败率顶到报告最显眼处, 这类错误必然被误读成模型效果。
    err_stats: Dict[str, int] = {}
    for row in per_query:
        err = row.get("rerank_error")
        if err:
            key = "403 配额/鉴权" if "403" in str(err) else str(err)[:60]
            err_stats[key] = err_stats.get(key, 0) + 1
    if err_stats:
        total = len(per_query)
        failed = sum(err_stats.values())
        lines.append("## ⚠️ 阶段失败告警\n")
        lines.append("```text")
        lines.append("rerank 失败的 query: %d / %d (%.0f%%)" % (
            failed, total, 100.0 * failed / max(total, 1)))
        for k, v in sorted(err_stats.items(), key=lambda x: -x[1]):
            lines.append("  %-24s %d 次" % (k, v))
        lines.append("")
        lines.append("受影响阶段的指标**不可信** —— 失败的 query 按 0 分计入,")
        lines.append("会把'接口没跑通'显示成'模型效果差'。")
        lines.append("```\n")
    lines.append("")

    lines.append("## 分阶段 × 类别\n")
    cats = sorted({k.split("::", 1)[1] for k in summary if "::" in k})
    if cats:
        lines.append("| 阶段 | 类别 | hit@5 | recall@5 | recall@20 | MRR@20 | n |")
        lines.append("|---|---|---|---|---|---|---|")
        for stage in stages:
            for cat in cats:
                s = summary.get("%s::%s" % (stage, cat)) or {}
                lines.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %d |" % (
                    stage, cat,
                    s.get("hit@5", 0), s.get("recall@5", 0), s.get("recall@20", 0),
                    s.get("mrr@20", 0), int(s.get("hit@5__n", 0)),
                ))
        lines.append("")

    if resolve_notes:
        lines.append("## 解析提示\n")
        lines.append("```text")
        for n in resolve_notes[:20]:
            lines.append(n)
        lines.append("```\n")

    lines.append("## 每条 query 的排名变化\n")
    lines.append("```text")
    for row in per_query:
        g = row["gold_size"]
        marks = " ".join(
            "%s=%s" % (st, (row["stages"].get(st, {}).get("scores", {}) or {}).get("mrr@20"))
            for st in STAGES if st in row["stages"]
        )
        lines.append("%-10s gold=%-3d %s" % (row["query_id"], g, marks))
    lines.append("```")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class _StubHit:
    def __init__(self, chunk_id: str) -> None:
        self.payload = {"chunk_id": chunk_id}


class _StubRetriever:
    """dry-run 用的桩检索器: 把该 query 的 gold 直接当检索结果返回。

    用途是验证 harness 整条链路(解析 → 打分 → 报告)本身是通的,
    而且指标必须完美 —— 若 recall@1 不是 1.0, 说明 harness 接错了。
    这比单测更彻底: 它跑的是真实的 run_eval 代码路径。

    桩按 query 原文找回对应 gold, 所以指标能一一对上。
    """

    def __init__(self, qrels_items: Sequence[Any],
                 gold_map: Dict[str, Set[str]]) -> None:
        self._by_query: Dict[str, Set[str]] = {}
        for item in qrels_items:
            self._by_query[item.query] = gold_map.get(item.query_id, set())

    def dense_search(self, query: str, limit: int, section: Any = None):  # noqa: ANN201
        return self._hits(query, limit)

    def sparse_search(self, query: str, limit: int, section: Any = None):  # noqa: ANN201
        return self._hits(query, limit), [], []

    def _hits(self, query: str, limit: int) -> List[_StubHit]:
        gold = self._by_query.get(query, set())
        return [_StubHit(c) for c in sorted(gold)[:limit]]


def _build_retriever(
    args: Any,
    items: Sequence[Any],
    gold_map: Dict[str, Set[str]],
    stages: Sequence[str],
):
    """构造检索器。dry-run 时返回桩, 否则连真实 Qdrant。"""
    if getattr(args, "dry_run", False):
        return _StubRetriever(items, gold_map), None, None

    from qdrant_client import QdrantClient
    from embeddings import build_embeddings
    from hybrid_retriever import HybridRetriever

    client = QdrantClient(path=args.qdrant_path)
    retriever = HybridRetriever(
        client=client,
        embeddings=build_embeddings("qwen"),
        dense_collection=args.dense_collection,
        sparse_collection=args.sparse_collection,
        artifact_dir=Path(args.artifact_dir),
        rrf_k=args.rrf_k,
    )
    reranker = None
    if "rerank" in stages:
        from reranker import DashScopeReranker
        reranker = DashScopeReranker.from_env()
    return retriever, reranker, client


def main() -> None:
    ap = argparse.ArgumentParser(description="分阶段检索评估")
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--chunks", required=True, nargs="+",
                    help="构建索引时的 chunks.jsonl (可传多份; 必须是带 fragment"
                         " text 的原始 dump, 瘦身后的索引 sidecar 无法做文本反查)")
    ap.add_argument("--artifact-dir", required=True,
                    help="含 vocab.json / bm25_stats.json 的目录")
    ap.add_argument("--qdrant-path", default="qdrant_data")
    ap.add_argument("--dense-collection", default="phase0_dense")
    ap.add_argument("--sparse-collection", default="phase0_sparse")
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--final-top-k", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--stages", default=",".join(STAGES),
                    help="逗号分隔, 默认全部五阶段")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="用桩检索器验证 harness 本身（不调 API, 指标应完美）")
    ap.add_argument("--out-dir", default="eval/runs")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.no_rerank and "rerank" in stages:
        stages.remove("rerank")
    for s in stages:
        if s not in STAGES:
            raise SystemExit("未知阶段: %s (可选 %s)" % (s, STAGES))

    print("=" * 74)
    print("分阶段检索评估")
    print("=" * 74)

    items = load_qrels(args.qrels)
    qsum = summarize(items)
    print("qrels      : %d 条 (%s)" % (qsum["total"], qsum["by_split"]))
    print("gold 片段  : %d 条, 平均 %.2f 条/query" % (
        qsum["gold_spans"], qsum["avg_gold_per_query"]))
    print()

    chunks_paths = [Path(p) for p in args.chunks]
    for p in chunks_paths:
        if not p.exists():
            # 不静默跳过: 缺文件意味着部分文档的 gold 一定会解析失败
            raise GoldResolveError(f"chunks.jsonl 不存在: {p}")
    artifact_dir = Path(args.artifact_dir)
    index = build_fragment_index_multi(chunks_paths)
    print("fragment 索引: %d 个 / %d 个 (文档,块) 键" % (index.size, index.keys))
    print("  跳过 image : %d" % index.skipped_image)
    print("  跳过空文本 : %d" % index.skipped_empty)
    print()

    print("解析 gold ...")
    gold_map, reports = resolve_all(items, index)
    notes = [r.note for r in reports if r.note]
    multi = sum(1 for r in reports if len(r.chunk_ids) > 1)
    print("  解析成功 %d 条, 其中 %d 条命中多个 chunk" % (len(reports), multi))
    print()

    from qdrant_client import QdrantClient
    from embeddings import build_embeddings
    from hybrid_retriever import HybridRetriever

    retriever, reranker, client = _build_retriever(args, items, gold_map, stages)
    if args.dry_run:
        print("模式       : DRY-RUN（桩检索器, 指标应完美）")
    elif reranker is not None:
        print("reranker   :", reranker.model)

    print("开始评估 (阶段: %s) ..." % ", ".join(stages))
    t0 = time.perf_counter()
    per_query, summary = evaluate(
        items, gold_map, retriever, reranker,
        candidate_k=args.candidate_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        stages=stages,
    )
    elapsed = time.perf_counter() - t0

    if client is not None:
        manifest = build_manifest(
            chunks_paths, artifact_dir,
            args.dense_collection, args.sparse_collection, client,
        )
    else:
        manifest = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "mode": "dry_run",
            "chunks_jsonl": [str(p) for p in chunks_paths],
            "chunks_sha256": {str(p): chunks_sha256(p) for p in chunks_paths},
        }
    config = {
        "qrels": args.qrels,
        "candidate_k": args.candidate_k,
        "final_top_k": args.final_top_k,
        "rrf_k": args.rrf_k,
        "stages": stages,
        "queries": len(items),
        "elapsed_sec": round(elapsed, 1),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / ts
    report = write_report(out_dir, summary, per_query, manifest, config, qsum,
                          notes, stages=stages)

    print()
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print("%-10s %-9s %-11s %-11s %-9s %s" % (
        "阶段", "hit@5", "recall@5", "recall@20", "MRR@20", "n"))
    print("-" * 74)
    for stage in stages:
        s = summary.get(stage) or {}
        print("%-10s %-9.3f %-11.3f %-11.3f %-9.3f %d" % (
            stage, s.get("hit@5", 0), s.get("recall@5", 0),
            s.get("recall@20", 0), s.get("mrr@20", 0), int(s.get("hit@5__n", 0))))
    print()

    # 失败率告警: 否则配额/鉴权失败会被误读成"该阶段效果差"
    n_fail = sum(1 for r in per_query if r.get("rerank_error"))
    if n_fail:
        first = next(str(r["rerank_error"]) for r in per_query if r.get("rerank_error"))
        print("!" * 74)
        print("⚠️  rerank 失败 %d / %d 条 (%.0f%%) —— rerank 行的指标不可信"
              % (n_fail, len(per_query), 100.0 * n_fail / max(len(per_query), 1)))
        print("    首个错误: %s" % first[:160])
        print("    失败的 query 按 0 分计入, 会把'接口没跑通'显示成'效果差'。")
        print("!" * 74)
        print()

    print("报告:", report)
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
