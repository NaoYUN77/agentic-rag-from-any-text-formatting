r"""验证 qrels 的 gold 在**重建后的语料**上还能不能解析。

为什么单独做这一步
------------------
gold 是锚在**文本片段**上的（不是 chunk_id / point_id）。
这样设计的目的就是"索引重建后 gold 依然有效" —— 因为 id 里含位置序号
(`c0001`), 一重建就全变。

所以"重建之后 gold 还能解析"是评估链路能不能用的**前置条件**。
它不需要 embedding、不需要 API Key, 完全可以离线验证。

本脚本对同一份 qrels 分别在**旧语料**和**新语料**上解析,
逐条列出结果 —— 任何一条解析失败都会显式报出来, 不静默跳过。

用法
----
    cd pipeline
    .\.venv_rag\Scripts\python.exe check_qrels_resolution.py
    .\.venv_rag\Scripts\python.exe check_qrels_resolution.py --qrels eval/qrels/phase0_url_draft.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.resolve import (  # noqa: E402
    FragmentIndex,
    build_fragment_index,
    build_fragment_index_multi,
    GoldResolveError,
)
from eval.qrels import load_qrels  # noqa: E402

DOCS = [
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


def merge_index(root: Path) -> Tuple[FragmentIndex, int]:
    """把 10 篇的 fragment 合成一个索引。

    构造逻辑统一在 eval.resolve 里 (build_fragment_index_multi),
    这里只负责挑文件 —— 否则会像之前那样绕开 text 归一化。
    """
    paths = [root / doc / "chunks.jsonl" for doc in DOCS]
    n_docs = sum(1 for p in paths if p.exists())
    return build_fragment_index_multi(paths), n_docs


def check(root: Path, items) -> dict:
    idx, n_docs = merge_index(root)
    rows = []
    for it in items:
        merged = set()
        errs: List[str] = []
        for gi, g in enumerate(it.gold):
            snippet = (g.text or "").strip()
            try:
                hits = idx.search(snippet)
                if not hits:
                    raise GoldResolveError("文本反查无结果")
                merged |= {h.chunk_id for h in hits}
            except Exception as exc:  # noqa: BLE001
                errs.append(f"gold[{gi}] {type(exc).__name__}: {str(exc)[:70]}")
        rows.append({
            "query_id": it.query_id,
            "n_chunks": len(merged),
            "ok": not errs,
            "errors": errs,
        })
    return {"root": str(root), "docs": n_docs, "fragments": idx.size,
            "keys": idx.keys, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="eval/qrels/phase0_url_draft.jsonl")
    ap.add_argument("--old-root", default="experiments")
    ap.add_argument("--new-root", default="experiments_rebuilt")
    args = ap.parse_args()

    items = load_qrels(args.qrels)
    print("=" * 84)
    print(f"qrels: {args.qrels}   ({len(items)} 条)")
    print("=" * 84)
    print()

    results = {}
    for tag, root in (("OLD", Path(args.old_root)), ("NEW", Path(args.new_root))):
        if not root.exists():
            print(f"[{tag}] 目录不存在: {root}")
            continue
        results[tag] = check(root, items)

    # 对照表
    print(f"  {'query_id':<10}{'OLD':>16}{'NEW':>16}")
    print("  " + "-" * 42)
    old = {r["query_id"]: r for r in results.get("OLD", {}).get("rows", [])}
    new = {r["query_id"]: r for r in results.get("NEW", {}).get("rows", [])}
    for qid in [r["query_id"] for r in results.get("OLD", {}).get("rows", [])]:
        o, n = old.get(qid), new.get(qid)
        os_ = f"{o['n_chunks']} chunks" if o and o["ok"] else ("FAIL" if o else "-")
        ns_ = f"{n['n_chunks']} chunks" if n and n["ok"] else ("FAIL" if n else "-")
        mark = "" if (o and n and o["ok"] and n["ok"]) else "   <<<"
        print(f"  {qid:<10}{os_:>16}{ns_:>16}{mark}")

    print()
    for tag, r in results.items():
        n_ok = sum(1 for x in r["rows"] if x["ok"])
        print(f"  [{tag}] {r['root']}  docs={r['docs']}  "
              f"fragments={r['fragments']}  keys={r['keys']}  "
              f"gold ok={n_ok}/{len(r['rows'])}")
        for x in r["rows"]:
            for e in x["errors"]:
                print(f"        ! {x['query_id']}  {e}")

    ok_all = all(x["ok"] for r in results.values() for x in r["rows"])
    print()
    print("结果:", "PASS — gold 在新旧语料上都能解析" if ok_all
          else "FAIL — 有 gold 解析不了, 见上面 !")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
