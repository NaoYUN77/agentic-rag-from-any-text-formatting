"""单句嵌入 vs 窗口嵌入: 哪种算出的距离更能识别语义边界?

设计:
    标注(弱)     Markdown 标题起始的句子之前 -> 视为"真断点"
                其余相邻位置              -> 视为"非断点"
    对照         两种机制在同一批位置上各算一遍距离
    指标         AUC (0.5=随机, 1.0=完美区分)

注意: 标题边界是【结构边界】, 不等于语义边界 —— 这是弱标注,
      但比完全没有标注强, 足够做相对比较。

运行:
    python embedding_mechanism_compare.py
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

import numpy as np

from pdf_loader import load_markdown
from embeddings import CN_SENTENCE_SPLIT_REGEX, build_embeddings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def unit(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def build_window_texts(sents: List[str], radius: int) -> List[str]:
    """每个句子配一个以它为中心的窗口文本。"""
    n = len(sents)
    out = []
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        out.append(" ".join(sents[lo:hi]))
    return out


def auc_of(scores: np.ndarray, labels: np.ndarray) -> float:
    """手算 AUC (避免额外依赖): 正样本分数高于负样本的比例。"""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 用秩和公式, 处理并列用平均秩
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    r_pos = ranks[labels].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    path = os.path.join("corpus", "redhat_p1-20.md")
    raw = open(path, encoding="utf-8", errors="replace").read()
    text = load_markdown(path)
    sents = [s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, text) if s.strip()]
    n = len(sents)

    print("语料: %s" % path)
    print("句数: %d" % n)
    print()

    # ---------- 标注: 标题起始句之前的位置 = 断点 ----------
    # 位置 i 表示"第 i 句和第 i+1 句之间"
    labels = np.zeros(n - 1, dtype=bool)
    title_pos = []
    for i in range(1, n):
        if sents[i].lstrip().startswith("#"):
            labels[i - 1] = True          # 边界在 i-1 和 i 之间
            title_pos.append(i)
    print("=== 标注(弱) ===")
    print("  以 # 开头的句子: %d 个" % len(title_pos))
    print("  标记为真断点的位置: %d 个 / 共 %d 个位置" % (labels.sum(), n - 1))
    print("  => 正样本占比 %.1f%%" % (100 * labels.mean()))
    print()
    print("  抽样几个被标为断点的位置:")
    for i in title_pos[:5]:
        print("    位置 %3d|%3d  下一句开头: %s" % (i - 1, i, sents[i][:44].replace("\n", " ")))
    print()

    # ---------- 两种机制算距离 ----------
    print("正在计算向量 ...")
    emb = build_embeddings("qwen")
    vecs_sent = np.array(emb.embed_documents(sents), dtype=np.float64)
    win_texts = build_window_texts(sents, radius=1)
    vecs_win = np.array(emb.embed_documents(win_texts), dtype=np.float64)
    print("  单句向量 %s   窗口向量 %s" % (vecs_sent.shape, vecs_win.shape))
    print()

    d_sent = np.array([1.0 - unit(vecs_sent[i], vecs_sent[i + 1]) for i in range(n - 1)])
    d_win = np.array([1.0 - unit(vecs_win[i], vecs_win[i + 1]) for i in range(n - 1)])

    # ---------- 对比 ----------
    print("=" * 78)
    print("对比: 哪种机制算出的距离更能识别标题边界?")
    print("=" * 78)
    print("  %-14s %10s %10s %10s %8s" % ("机制", "断点均值", "非断点均值", "差距", "AUC"))
    print("  " + "-" * 62)
    for name, d in [("单句嵌入", d_sent), ("窗口嵌入", d_win)]:
        mp, mn = float(d[labels].mean()), float(d[~labels].mean())
        a = auc_of(d, labels)
        print("  %-14s %10.4f %10.4f %10.4f %8.4f" % (name, mp, mn, mp - mn, a))
    print()
    print("  AUC 含义: 0.5 = 完全随机, 1.0 = 完美区分")
    print("  差距 = 断点平均距离 - 非断点平均距离, 越大说明信号越强")
    print()

    # ---------- 换个粒度再验一次: 顶级标题 ----------
    labels2 = np.zeros(n - 1, dtype=bool)
    for i in range(1, n):
        if sents[i].lstrip().startswith("# "):
            labels2[i - 1] = True
    if labels2.sum() >= 3:
        print("=== 复核: 只用【一级标题】(# )作标注 ===")
        print("  正样本: %d 个" % labels2.sum())
        for name, d in [("单句嵌入", d_sent), ("窗口嵌入", d_win)]:
            a = auc_of(d, labels2)
            print("  %-14s AUC = %.4f" % (name, a))
        print()

    print("=" * 78)
    print("说明")
    print("=" * 78)
    print("  1. 这是【弱标注】: 标题是结构边界, 不一定等于语义边界")
    print("  2. AUC 只反映'能否排序', 不反映绝对距离大小")
    print("  3. 两种机制的距离尺度不同, 所以只能比 AUC, 不能比绝对值")


if __name__ == "__main__":
    main()
