"""混合分块: 固定长度定上限 + 在窗口内找语义断点。

对比三种策略:
    A. 纯固定长度      到 max_chars 就切, 不看语义
    B. 纯语义分块      LangChain SemanticChunker, 只看语义不看长度
    C. 混合            长度定上限, 在边界附近挑语义断点

评价指标:
    块大小分布(块太大/太小都是问题)
    切点处的平均语义距离(越高说明切得越"在断点上")

运行:
    python hybrid_chunk_demo.py                # 用 corpus/redhat_p1-20.md
    python hybrid_chunk_demo.py --max-chars 800 --window 4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np

from pdf_loader import load_markdown
from semantic_chunker_demo import CN_SENTENCE_SPLIT_REGEX, build_embeddings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, text) if s.strip()]


def unit(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# --------------------------------------------------------------------------
# A. 纯固定长度
# --------------------------------------------------------------------------
def fixed_chunks(sents: List[str], max_chars: int) -> List[Tuple[int, int]]:
    """返回 [(起始句下标, 结束句下标)] 的列表(左闭右开)。"""
    out, start, size = [], 0, 0
    for i, s in enumerate(sents):
        if size + len(s) > max_chars and i > start:
            out.append((start, i))
            start, size = i, len(s)
        else:
            size += len(s)
    if start < len(sents):
        out.append((start, len(sents)))
    return out


# --------------------------------------------------------------------------
# C. 混合: 长度定上限 + 窗口内挑语义断点
# --------------------------------------------------------------------------
def hybrid_chunks(
    sents: List[str],
    vecs: np.ndarray,
    max_chars: int,
    window: int = 4,
) -> List[Tuple[int, int]]:
    """贪心累积到接近上限, 然后往回看 window 句, 挑语义距离最大的位置切。

    切点候选 i 表示"切在第 i 句之后"(即下一块从 i+1 开始)。
    """
    n = len(sents)
    out: List[Tuple[int, int]] = []
    start = 0

    while start < n:
        # ① 按长度贪心往后吃, 得到硬边界 end
        end, size = start, 0
        while end < n and (size + len(sents[end]) <= max_chars or end == start):
            size += len(sents[end])
            end += 1

        if end >= n:
            out.append((start, n))
            break

        # ② 在块的【尾部窗口】里找语义距离最大的切点
        #    切点 i 表示在第 i 句后切, 下一块从 i+1 开始
        #
        #    两个保护条件(第一版没加, 结果出现了 38 字的碎块):
        #      a. 块本身至少 3 句, 否则不往回挪
        #      b. 窗口不超过块长的一半, 免得切点退到块首
        span = end - start
        if span < 3:
            cut = end
        else:
            w = min(window, span // 2)
            lo = end - w
            best_i, best_d = end - 1, -1.0
            for i in range(lo, end):
                if i + 1 >= n:
                    continue
                d = 1.0 - unit(vecs[i], vecs[i + 1])
                if d > best_d:
                    best_d, best_i = d, i
            cut = best_i + 1
        out.append((start, cut))
        start = cut

    return out


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# D. 语义断点优先 + 长度双向约束
# --------------------------------------------------------------------------
def semantic_first_chunks(
    sents: List[str],
    vecs: np.ndarray,
    max_chars: int,
    min_chars: int = 200,
    pct: float = 90.0,
) -> List[Tuple[int, int]]:
    """先用语义断点定切点, 再用长度做双向约束。

    和 hybrid_chunks 的区别:
        hybrid_chunks    长度先, 语义只在块尾窗口微调 -> 看不见块内的语义跳跃
        semantic_first   语义先, 长度只做兜底和合并   -> 能发现任意位置的跳跃
    """
    n = len(sents)
    if n < 2:
        return [(0, n)]

    dists = np.array([1.0 - unit(vecs[i], vecs[i + 1]) for i in range(n - 1)])
    thr = float(np.percentile(dists, pct))
    # 候选切点: 切在第 i 句之后 -> 下一块从 i+1 开始
    cands = [i + 1 for i, d in enumerate(dists) if d > thr]

    out: List[Tuple[int, int]] = []
    start = 0
    for c in cands + [n]:
        if c <= start:
            continue
        length = sum(len(s) for s in sents[start:c])

        # ① 太短 -> 跳过这个切点, 让它和后面的内容合并
        if length < min_chars and c < n:
            continue

        # ② 太长 -> 先按长度硬切几刀
        while sum(len(s) for s in sents[start:c]) > max_chars:
            size, k = 0, start
            while k < c and (size + len(sents[k]) <= max_chars or k == start):
                size += len(sents[k])
                k += 1
            out.append((start, k))
            start = k

        # ③ 剩下的收尾
        if start < c:
            out.append((start, c))
            start = c

    if start < n:
        out.append((start, n))
    return out


# 评价
# --------------------------------------------------------------------------
def report(
    name: str,
    spans: List[Tuple[int, int]],
    sents: List[str],
    vecs: np.ndarray,
    all_dists: np.ndarray,
) -> None:
    sizes = [sum(len(s) for s in sents[a:b]) for a, b in spans]
    cuts = []
    for a, b in spans[:-1]:
        if b < len(sents):
            cuts.append(1.0 - unit(vecs[b - 1], vecs[b]))
    global_mean = float(all_dists.mean())
    cut_mean = float(np.mean(cuts)) if cuts else 0.0
    print("  %-14s 块数 %3d   字数 min %5d 均值 %6.0f max %5d   切点均值距离 %.4f"
          % (name, len(spans), min(sizes), sum(sizes) / len(sizes), max(sizes), cut_mean))
    return cut_mean, global_mean


def main() -> None:
    ap = argparse.ArgumentParser(description="混合分块演示")
    ap.add_argument("--md", default=os.path.join("corpus", "redhat_p1-20.md"))
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--backend", default="qwen")
    args = ap.parse_args()

    text = load_markdown(args.md)
    sents = split_sentences(text)
    print("语料: %s" % args.md)
    print("句数: %d   总字数: %d" % (len(sents), sum(len(s) for s in sents)))
    print("参数: max_chars=%d  window=%d" % (args.max_chars, args.window))
    print()

    print("正在计算句子向量 ...")
    emb = build_embeddings(args.backend)
    vecs = np.array(emb.embed_documents(sents), dtype=np.float64)
    all_dists = np.array([1.0 - unit(vecs[i], vecs[i + 1]) for i in range(len(sents) - 1)])
    print("  向量 shape = %s" % (vecs.shape,))
    print()

    print("=" * 78)
    print("三种策略对比")
    print("=" * 78)
    print("  全局相邻距离均值: %.4f  (切点距离高于它, 说明切在了语义断点上)" % all_dists.mean())
    print()

    a = fixed_chunks(sents, args.max_chars)
    report("A. 纯固定长度", a, sents, vecs, all_dists)

    hy = hybrid_chunks(sents, vecs, args.max_chars, args.window)
    report("C. 长度优先+语义微调", hy, sents, vecs, all_dists)

    sf = semantic_first_chunks(sents, vecs, args.max_chars, min_chars=200, pct=90)
    report("D. 语义优先+长度约束", sf, sents, vecs, all_dists)

    print()
    print("=" * 78)
    print("块大小分布(混合方案)")
    print("=" * 78)
    sizes = sorted(sum(len(s) for s in sents[a:b]) for a, b in hy)
    print("  " + " ".join(str(x) for x in sizes))
    print()
    print("  前 3 块的切点位置与语义距离:")
    for a_, b_ in hy[:3]:
        if b_ < len(sents):
            d = 1.0 - unit(vecs[b_ - 1], vecs[b_])
            print("    切在第 %d / %d 句之间   距离 %.4f" % (b_, len(sents), d))
            print("      上一块结尾: %s" % sents[b_ - 1][-38:])
            print("      下一块开头: %s" % sents[b_][:38])


if __name__ == "__main__":
    main()


