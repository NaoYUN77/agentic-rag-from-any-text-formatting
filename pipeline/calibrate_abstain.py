r"""标定拒答阈值。

**为什么需要这个脚本**：拒答阈值是「语料 + 嵌入模型」绑定的，换任一个都必须重算。
把它写死在代码里就会变成一个没人敢动的魔法数字 —— 而且换了语料之后**它是错的**，
却不会有人发现（症状是「明明语料里有答案却被拒答」，很难归因到这里）。

这个脚本把标定做成一条命令：读 qrels 的负样本，跑一遍 dense 检索，
按「**误拒为 0 的前提下抓住最多负样本**」给出推荐阈值。

用法
----
    cd pipeline
    .\.venv_rag\Scripts\python.exe calibrate_abstain.py
    .\.venv_rag\Scripts\python.exe calibrate_abstain.py --write      # 写入 .abstain_threshold

之后启动服务时：
    export RAG_ABSTAIN_THRESHOLD=0.5285        # 脚本会打印这一行
或直接让服务读文件（见 rag_server 的 ABSTAIN_THRESHOLD 解析）。

判据与评估器**共用同一个函数**（`eval.run_eval.abstention_analysis`），
所以两边不会出现"定义漂移"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from eval.qrels import load_qrels                    # noqa: E402
from eval.run_eval import abstention_analysis        # noqa: E402

THRESHOLD_FILE = _HERE / ".abstain_threshold"


def _build_retriever(args) -> tuple:
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
    )
    return retriever, client


def main() -> int:
    ap = argparse.ArgumentParser(description="标定拒答阈值")
    ap.add_argument("--qrels", default="eval/qrels/phase0_url_draft.jsonl")
    ap.add_argument("--artifact-dir", default="index_artifacts/phase0_capped")
    ap.add_argument("--dense-collection", default="phase0_capped_dense")
    ap.add_argument("--sparse-collection", default="phase0_capped_sparse")
    ap.add_argument("--qdrant-path", default="qdrant_data")
    ap.add_argument("--write", action="store_true",
                    help="把推荐阈值写入 .abstain_threshold（服务会读它）")
    ap.add_argument("--json-out", default=None,
                    help="把完整结果写成 JSON（分布 + 扫描表 + 推荐值）")
    args = ap.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        key_file = _HERE / ".dashscope_key"
        if key_file.exists():
            os.environ["DASHSCOPE_API_KEY"] = key_file.read_text(
                encoding="utf-8").strip()
        else:
            raise SystemExit("缺少 DASHSCOPE_API_KEY")

    items = load_qrels(args.qrels)
    positives = [i for i in items if i.category != "unanswerable"]
    negatives = [i for i in items if i.category == "unanswerable"]
    print("qrels        : %s" % args.qrels)
    print("  可回答      : %d 条" % len(positives))
    print("  负样本      : %d 条" % len(negatives))
    if not negatives:
        raise SystemExit(
            "qrels 里没有 unanswerable 负样本 —— 无法标定拒答阈值。\n"
            "拒答阈值必须由负样本标定；没有负样本就不该启用拒答。"
        )
    print()

    retriever, client = _build_retriever(args)
    print("索引         : %s" % args.dense_collection)
    print("嵌入模型     : %s" % getattr(retriever.embeddings, "model", "?"))
    print()
    print("跑 dense 检索（每条只取 top-1 分数）...")

    rows = []
    for i, item in enumerate(items, 1):
        try:
            hits = retriever.dense_search(item.query, 1)
        except Exception as exc:                      # 单条失败不该中断标定
            print("  [%s] 失败: %s" % (item.query_id, str(exc)[:80]))
            continue
        rows.append({
            "query_id": item.query_id,
            "category": item.category,
            "top_dense_score": (
                float(hits[0].dense_score)
                if hits and hits[0].dense_score is not None else None
            ),
        })
        if i % 10 == 0 or i == len(items):
            print("  已处理 %d/%d" % (i, len(items)))
    print()

    result = abstention_analysis(rows)
    if result is None:
        raise SystemExit("样本不足，无法给出阈值（需要正负样本都有）。")

    pos, neg = result["positive"], result["negative"]
    print("=" * 74)
    print("dense top-1 相似度分布")
    print("=" * 74)
    fmt = "  %-8s n=%-3d min=%.3f  p25=%.3f  p50=%.3f  p75=%.3f  max=%.3f"
    print(fmt % ("可回答", pos["n"], pos["min"], pos["p25"],
                 pos["p50"], pos["p75"], pos["max"]))
    print(fmt % ("负样本", neg["n"], neg["min"], neg["p25"],
                 neg["p50"], neg["p75"], neg["max"]))
    print()

    rec = result.get("recommended")
    if not rec:
        print("⚠️  没有任何切点能做到「不误拒」—— 两类分布完全重叠。")
        print("    建议**不要启用拒答**：任何阈值都会误伤可回答的问题。")
        return 1

    print("=" * 74)
    print("推荐阈值")
    print("=" * 74)
    print("  %.4f" % rec["threshold"])
    print("    -> 正确拒答 %d/%d 条负样本，误拒 %d/%d 条可回答"
          % (rec["caught_neg"], neg["n"], rec["wrong_pos"], pos["n"]))
    print()
    print("  含义：top-1 相似度低于 %.4f 时判为「语料里没有答案」。"
          % rec["threshold"])
    if not result.get("separable"):
        print()
        print("  ⚠️ 两类分布**有重叠**（负样本 max %.3f > 可回答 min %.3f）——"
              % (neg["max"], pos["min"]))
        print("     单靠相似度阈值无法完全分开，只能拒掉一部分。"
              "剩下的负样本仍会被硬答。")
    print()
    print("  启动服务时这样设：")
    print("    export RAG_ABSTAIN_THRESHOLD=%.4f" % rec["threshold"])

    if args.write:
        THRESHOLD_FILE.write_text("%.6f\n" % rec["threshold"], encoding="utf-8")
        print()
        print("  已写入 %s（服务在未设环境变量时会读它）"
              % THRESHOLD_FILE.name)

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(json.dumps({
            "qrels": args.qrels,
            "dense_collection": args.dense_collection,
            "embedding_model": getattr(retriever.embeddings, "model", None),
            "positive": pos, "negative": neg,
            "recommended": rec,
            "separable": result.get("separable"),
            "sweep": result["sweep"],
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  明细已写入 %s" % out)

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
