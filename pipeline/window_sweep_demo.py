"""边界窗口大小对分块质量的影响。

验证一个设计: 先用固定长度切块, 再在【边界窗口】里做语义微调。
窗口越大, 越可能找到真正的语义断点, 但块数也会变多。

对比不同窗口宽度下的:
    块数 / 块大小分布 / 切点处平均语义距离 / 分块阶段的嵌入成本

运行:
    python window_sweep_demo.py
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

import numpy as np

from pdf_loader import load_markdown
from semantic_chunker_demo import CN_SENTENCE_SPLIT_REGEX, build_embeddings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def sentences_of(text: str) -> List[str]:
    return [s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, text) if s.strip()]


def unit(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def char_window_chunks(
    sents: List[str],
    vecs: np.ndarray,
    max_chars: int,
    window_chars: int,
) -> List[Tuple[int, int]]:
    """固定长度粗切 + 在【边界前 window_chars 字】范围内找语义断点。

    window_chars 就相当于 OpenAI 那种"块间重叠"的宽度:
        窗口 = 0      -> 退化成纯固定长度
        窗口越大       -> 切点能挪的距离越远, 越可能找到真断点
    """
    n = len(sents)
    out: List[Tuple[int, int]] = []
    start = 0

    while start < n:
        # ① 贪心吃到 max_chars, 得到硬边界 end
        end, size = start, 0
        while end < n and (size + len(sents[end]) <= max_chars or end == start):
            size += len(sents[end])
            end += 1

        if end >= n:
            out.append((start, n))
            break

        if window_chars <= 0:
            cut = end
        else:
            # ② 把 window_chars 折算成"从 end 往回数几句"
            acc, lo = 0, end
            while lo > start + 1 and acc < window_chars:
                lo -= 1
                acc += len(sents[lo])

            # ③ 在 [lo, end) 里挑距离最大的位置
            #    保护: 不许切点落在块首(否则会切出极小的块)
            lo = max(lo, start + 1)
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


def main() -> None:
    path = os.path.join("corpus", "redhat_p1-20.md")
    text = load_markdown(path)
    sents = sentences_of(text)
    n = len(sents)
    total = sum(len(s) for s in sents)

    print("语料: %s" % path)
    print("句数 %d   总字数 %d" % (n, total))
    print()

    print("正在计算句子向量 ...")
    emb = build_embeddings("qwen")
    vecs = np.array(emb.embed_documents(sents), dtype=np.float64)
    dists = np.array([1.0 - unit(vecs[i], vecs[i + 1]) for i in range(n - 1)])
    print("  全局相邻距离均值 = %.4f" % dists.mean())
    print()

    MAXC = 800
    print("=" * 82)
    print("边界窗口宽度 sweep (max_chars=%d)" % MAXC)
    print("=" * 82)
    print("  窗口   块数   字数 min/均值/max        切点均值距离   相对全局")
    print("  " + "-" * 74)

    for w in [0, 100, 200, 300, 400, 600]:
        spans = char_window_chunks(sents, vecs, MAXC, w)
        sizes = [sum(len(s) for s in sents[a:b]) for a, b in spans]
        cuts = []
        for a, b in spans[:-1]:
            if b < n:
                cuts.append(1.0 - unit(vecs[b - 1], vecs[b]))
        cm = float(np.mean(cuts)) if cuts else 0.0
        print("  %4d   %4d   %5d / %5.0f / %5d      %.4f        %.2fx"
              % (w, len(spans), min(sizes), sum(sizes) / len(sizes), max(sizes),
                 cm, cm / dists.mean()))

    print()
    print("  * 窗口=0 即纯固定长度;  相对全局 = 切点距离 / 全局均值")
    print("  * 相对全局 1.00x 表示切点是随机的, 越高说明越切在语义断点上")


if __name__ == "__main__":
    main()
