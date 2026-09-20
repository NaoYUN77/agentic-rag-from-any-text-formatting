r"""排序层参数扫描（复用已有索引，不重建）。

为什么单独做这个
----------------
`run_eval` 里的接线是:

    dense  = dense_search(q, candidate_k)
    sparse = sparse_search(q, candidate_k)
    fused  = rrf_fuse([dense, sparse], k=rrf_k, limit=candidate_k)  # 融合后截断
    final  = reranker.rerank(q, fused, top_k=final_top_k)

**`candidate_k` 同时控制"每路深度"和"rerank 能看到的候选数"**。
实测 `ce_02` 的 gold 在融合后正好排第 21 名 —— 被 `candidate_k=20` 挡在
rerank 之外, 于是全阶段 0.0。所以 candidate_k 是**有明确预测**的杠杆。

rrf_k 则是 RRF 的平滑常数: 越小越强调头部排名。

本脚本只改这两个参数, 语料与索引都不动, 因此各变体严格可比。

用法
----
    export DASHSCOPE_API_KEY="$(cat .dashscope_key)"
    ./.venv_rag/Scripts/python.exe sweep_ranking.py \
        --artifact-dir index_artifacts/phase0_sweep_t800 \
        --dense-collection phase0_sweep_t800_dense \
        --sparse-collection phase0_sweep_t800_sparse \
        --grid 20:60,50:60,50:10,100:60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sweep_chunk_params import _parse_metrics  # noqa: E402

PY = str(_HERE / ".venv_rag" / "Scripts" / "python.exe")
QRELS = "eval/qrels/phase0_url_draft.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--dense-collection", required=True)
    ap.add_argument("--sparse-collection", required=True)
    ap.add_argument("--corpus-dir", default=None,
                    help="chunks.jsonl 所在语料目录 (默认由 artifact-dir 推断)")
    ap.add_argument("--grid", default="20:60,50:60,50:10,100:60",
                    help="candidate_k:rrf_k 的逗号分隔列表")
    ap.add_argument("--stages", default=None,
                    help="只跑这些阶段 (如 dense,sparse,union,rrf)。"
                         "rerank 配额耗尽时用它拿到有效结论。")
    ap.add_argument("--json-out", default="sweep_ranking.json")
    args = ap.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        kf = _HERE / ".dashscope_key"
        if kf.exists():
            os.environ["DASHSCOPE_API_KEY"] = kf.read_text(encoding="utf-8").strip()
        else:
            raise SystemExit("缺少 DASHSCOPE_API_KEY")
    env = dict(os.environ)

    # chunks 路径: 优先显式给出, 否则找同名 sweep 语料目录
    # 目录命名约定: index_artifacts/phase0_sweep_<tag> <-> experiments_sweep/<tag>
    if args.corpus_dir:
        chunk_files = [str(p) for p in sorted(Path(_HERE, args.corpus_dir).glob("*/chunks.jsonl"))]
    else:
        name = Path(args.artifact_dir).name          # 例如 phase0_sweep_t800
        tag = name.split("phase0_", 1)[-1]           # -> sweep_t800
        tag = tag.split("sweep_", 1)[-1]             # -> t800
        chunk_files = [str(p) for p in sorted(
            Path(_HERE, "experiments_sweep", tag).glob("*/chunks.jsonl"))]
        print("自动推断语料目录: experiments_sweep/%s" % tag)
    if not chunk_files:
        raise SystemExit("找不到 chunks.jsonl, 请用 --corpus-dir 指定")
    print("chunks 文件 %d 个" % len(chunk_files))

    grid: List[tuple] = []
    for item in args.grid.split(","):
        ck, rk = item.split(":")
        grid.append((int(ck), int(rk)))

    results = []
    for ck, rk in grid:
        print("=" * 78)
        print("candidate_k=%d  rrf_k=%d" % (ck, rk))
        print("=" * 78)
        cmd = [PY, "-m", "eval.run_eval", "--qrels", QRELS,
               "--chunks", *chunk_files,
               "--artifact-dir", args.artifact_dir,
               "--dense-collection", args.dense_collection,
               "--sparse-collection", args.sparse_collection,
               "--candidate-k", str(ck), "--rrf-k", str(rk),
               "--out-dir", "eval/runs"]
        if args.stages:
            cmd += ["--stages", args.stages]
        proc = subprocess.run(cmd, cwd=str(_HERE), env=env, capture_output=True,
                              encoding="utf-8", errors="replace", timeout=1800)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            print("  失败: %s" % out[-1500:])
            continue
        m = re.search(r"报告:\s*(\S+)", out)
        report = m.group(1).strip() if m else None
        metrics = {}
        if report:
            p = Path(_HERE, report)
            if p.exists():
                metrics = _parse_metrics(p.read_text(encoding="utf-8"))
        print("  %s" % report)
        results.append({"candidate_k": ck, "rrf_k": rk,
                        "report": report, "metrics": metrics})

    # ---------------- 汇总 ----------------
    stages = ["dense", "sparse", "union", "rrf", "rerank"]
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    for metric in ("hit@5", "recall@20", "MRR@20", "nDCG@5"):
        print("\n%s" % metric)
        print("  %-16s %s" % ("cand:rrf", "  ".join("%-8s" % s for s in stages)))
        for r in results:
            vals = []
            for s in stages:
                v = r["metrics"].get(s, {}).get(metric)
                vals.append("%-8s" % ("%.3f" % v if v is not None else "-"))
            print("  %-16s %s" % ("%d:%d" % (r["candidate_k"], r["rrf_k"]),
                                  "  ".join(vals)))

    Path(_HERE / args.json_out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n明细已写入 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
