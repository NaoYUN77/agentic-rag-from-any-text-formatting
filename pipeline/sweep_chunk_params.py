r"""切块参数 / 解析器变体 的端到端扫描。

目的
----
之前所有关于「chunk_tokens=800 是否合适」的结论都只有**结构性**证据
（chunk 数、token 分布、斜率），没有**检索指标**。现在有真实语义
embedding 了, 可以量出切块参数对召回的实际影响。

用法
----
    export DASHSCOPE_API_KEY="$(cat .dashscope_key)"
    ./.venv_rag/Scripts/python.exe sweep_chunk_params.py \
        --variants 600,800,1200

每个变体做三件事（全部落到独立目录, 互不覆盖）:

    1. 重建语料   experiments_sweep/<tag>/<doc>/
    2. 建索引     phase0_sweep_<tag>_dense / _sparse + index_artifacts/phase0_sweep_<tag>/
    3. 跑评估     读上面的 collection, 输出 eval/runs/<ts>/

设计取舍
--------
- **不删任何既有目录**: 每个变体用新路径。原因: 本沙箱的 safe-delete shim
  会拦截 `shutil.rmtree`, 而且 FAIL_CLOSED 并没有真的兜住
  (实测 2026-09-20 删掉过 experiments_rebuilt/)。
- **复用 rebuild_phase0_index 与 eval.run_eval**, 不重写逻辑 ——
  扫描用的必须就是生产路径, 否则结论不可迁移。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

PY = str(_HERE / ".venv_rag" / "Scripts" / "python.exe")
QRELS = "eval/qrels/phase0_url_draft.jsonl"


def _run(cmd: List[str], env: Dict[str, str]) -> str:
    proc = subprocess.run(
        cmd, cwd=str(_HERE), env=env, capture_output=True,
        encoding="utf-8", errors="replace", timeout=1800,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError("命令失败: %s\n%s" % (" ".join(cmd), out[-3000:]))
    return out


def _parse_metrics(report_md: str) -> Dict[str, Dict[str, float]]:
    """从 report.md 的「分阶段指标」表里抽出各阶段指标。

    坑: 报告里还有一张「分阶段 × 类别」表, 阶段名会重复出现。
    若全文扫行, 后出现的 unanswerable 行(全 0)会把真实值**覆盖**掉。
    所以必须先切出「分阶段指标」这一节再解析。
    """
    # 只取 "## 分阶段指标" 到下一个 "## " 之间的内容
    section = report_md
    m = re.search(r"^##\s*分阶段指标\s*$(.*?)(?=^##\s|\Z)",
                  report_md, re.MULTILINE | re.DOTALL)
    if m:
        section = m.group(1)

    rows: Dict[str, Dict[str, float]] = {}
    for line in section.splitlines():
        hit = re.match(
            r"^\|\s*(dense|sparse|union|rrf|rerank)\s*\|(.*)\|\s*$", line)
        if not hit:
            continue
        stage = hit.group(1)
        if stage in rows:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+", hit.group(2))
        if len(nums) < 5:
            continue
        rows[stage] = {
            "hit@5": float(nums[0]), "recall@5": float(nums[1]),
            "recall@20": float(nums[2]), "MRR@20": float(nums[3]),
            "nDCG@5": float(nums[4]),
        }
    return rows


def run_variant(tag: str, chunk_tokens: int,
                env: Dict[str, str]) -> Dict:
    corpus = f"experiments_sweep/{tag}"
    art_dir = f"index_artifacts/phase0_sweep_{tag}"
    dense = f"phase0_sweep_{tag}_dense"
    sparse = f"phase0_sweep_{tag}_sparse"

    print("=" * 78)
    print("变体 %s : chunk_tokens=%d" % (tag, chunk_tokens))
    print("=" * 78)

    print("  [1/3] 重建语料 ...")
    out = _run([PY, "rebuild_phase0_index.py", "--backend", "qwen",
                "--chunk-tokens", str(chunk_tokens),
                "--out-dir", corpus,
                "--artifact-dir", art_dir,
                "--dense-collection", dense,
                "--sparse-collection", sparse], env)
    m = re.search(r'"chunks_loaded":\s*(\d+)', out)
    chunks = int(m.group(1)) if m else -1
    m = re.search(r'"parent_count":\s*(\d+)', out)
    parents = int(m.group(1)) if m else -1
    print("        chunks=%d parents=%d" % (chunks, parents))

    print("  [2/3] 跑评估 ...")
    chunk_files = [str(p) for p in sorted(Path(_HERE, corpus).glob("*/chunks.jsonl"))]
    out = _run([PY, "-m", "eval.run_eval", "--qrels", QRELS,
                "--chunks", *chunk_files,
                "--artifact-dir", art_dir,
                "--dense-collection", dense,
                "--sparse-collection", sparse,
                "--out-dir", "eval/runs"], env)
    m = re.search(r"报告:\s*(\S+)", out)
    report_path = m.group(1).strip() if m else None
    print("        报告: %s" % report_path)

    metrics = {}
    if report_path:
        p = Path(_HERE, report_path, "report.md")
        if not p.exists():
            p = Path(_HERE, report_path)
        if p.exists():
            metrics = _parse_metrics(p.read_text(encoding="utf-8"))

    print("  [3/3] 完成\n")
    return {"tag": tag, "chunk_tokens": chunk_tokens,
            "chunks": chunks, "parents": parents,
            "report": report_path, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="600,800,1200",
                    help="逗号分隔的 chunk_tokens 列表")
    ap.add_argument("--json-out", default="sweep_chunk_params.json")
    args = ap.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        key_file = _HERE / ".dashscope_key"
        if key_file.exists():
            os.environ["DASHSCOPE_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
        else:
            raise SystemExit("缺少 DASHSCOPE_API_KEY")
    env = dict(os.environ)

    sizes = [int(s) for s in args.variants.split(",") if s.strip()]
    results = []
    for size in sizes:
        tag = "t%d" % size
        results.append(run_variant(tag, size, env))

    # ---------------- 汇总 ----------------
    print("=" * 78)
    print("汇总 (qrels=%s)" % QRELS)
    print("=" * 78)
    stages = ["dense", "sparse", "union", "rrf", "rerank"]
    for metric in ("hit@5", "recall@5", "recall@20", "MRR@20", "nDCG@5"):
        print("\n%s" % metric)
        print("  %-8s %-6s %s" % ("chunk", "chunks", "  ".join("%-8s" % s for s in stages)))
        for r in results:
            vals = []
            for s in stages:
                v = r["metrics"].get(s, {}).get(metric)
                vals.append("%-8s" % ("%.3f" % v if v is not None else "-"))
            print("  %-8d %-6d %s" % (r["chunk_tokens"], r["chunks"], "  ".join(vals)))

    Path(_HERE / args.json_out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n明细已写入 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
