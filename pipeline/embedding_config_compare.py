"""嵌入配置对比: 成本 + 分块效果。

三种配置:
    A. 单句嵌入                每句一个向量
    B. 窗口嵌入 (radius=1)     前后各 1 句拼起来
    C. 单句 + metadata 前缀     每句前面拼上 文件/页码/章节

度量:
    成本    输入字符数、调用次数     (纯本地计算)
    效果    拿它做切点判定, 输出块数 / 块大小 / 切点示例

运行:
    python embedding_config_compare.py
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

MAX_CHARS = 800
WINDOW_CHARS = 200


def unit(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def build_metadata_prefixes(sents: List[str]) -> List[str]:
    """给每句生成 metadata 前缀, 模拟真实管线里的文件/页码/章节。"""
    out, header, page = [], "Red Hat 负载平衡器管理", 1
    for i, s in enumerate(sents):
        t = s.lstrip()
        if t.startswith("#"):
            header = t.lstrip("#").strip().replace("\n", " ")[:28]
        if i > 0 and i % 14 == 0:
            page += 1
        out.append("file: redhat-lb.pdf | page: %d | section: %s" % (page, header))
    return out


def window_texts(sents: List[str], radius: int = 1) -> List[str]:
    n = len(sents)
    return [
        " ".join(sents[max(0, i - radius): min(n, i + radius + 1)])
        for i in range(n)
    ]


def chunks_by_window(
    sents: List[str], vecs: np.ndarray, max_chars: int, window_chars: int
) -> List[Tuple[int, int]]:
    """固定长度粗切 + 在边界窗口内挑语义断点(单句向量)。"""
    n = len(sents)
    out: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end, size = start, 0
        while end < n and (size + len(sents[end]) <= max_chars or end == start):
            size += len(sents[end])
            end += 1
        if end >= n:
            out.append((start, n))
            break
        acc, lo = 0, end
        while lo > start + 1 and acc < window_chars:
            lo -= 1
            acc += len(sents[lo])
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
    sents = [s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, text) if s.strip()]
    n = len(sents)
    total_chars = sum(len(s) for s in sents)

    meta = build_metadata_prefixes(sents)
    win = window_texts(sents, 1)
    sent_with_meta = ["%s\n\n%s" % (m, s) for m, s in zip(meta, sents)]

    # ---------- 成本(纯本地计算, 不用调 API) ----------
    print("语料: %d 句, 总 %d 字" % (n, total_chars))
    print()
    print("=" * 82)
    print("[1] 成本对比 (分块阶段的嵌入输入量)")
    print("=" * 82)
    rows = [
        ("A. 单句嵌入", n, sum(len(s) for s in sents)),
        ("B. 窗口嵌入(radius=1)", n, sum(len(s) for s in win)),
        ("C. 单句 + metadata", n, sum(len(s) for s in sent_with_meta)),
        ("D. 窗口 + metadata", n, sum(len(s) for s in win) + sum(len(m) + 2 for m in meta)),
    ]
    print("  %-24s %8s %12s %10s" % ("配置", "调用次数", "输入字符数", "相对 A"))
    print("  " + "-" * 60)
    base = rows[0][2]
    for name, calls, chars in rows:
        print("  %-24s %8d %12d %9.2fx" % (name, calls, chars, chars / base))
    print()
    print("  metadata 前缀平均长度: %.1f 字/句" % (sum(len(m) for m in meta) / n))
    print()

    # ---------- 效果(需要调 API) ----------
    print("正在计算向量 (三种配置) ...")
    emb = build_embeddings("qwen")
    v_sent = np.array(emb.embed_documents(sents), dtype=np.float64)
    v_win = np.array(emb.embed_documents(win), dtype=np.float64)
    v_meta = np.array(emb.embed_documents(sent_with_meta), dtype=np.float64)
    print()
    print("=" * 82)
    print("[2] 分块效果对比 (max_chars=%d, 窗口=%d)" % (MAX_CHARS, WINDOW_CHARS))
    print("=" * 82)
    print("  %-24s %6s %10s %10s %12s" % ("配置", "块数", "min", "均值", "切点均距"))
    print("  " + "-" * 66)
    for name, vecs in [
        ("A. 单句嵌入", v_sent),
        ("B. 窗口嵌入(radius=1)", v_win),
        ("C. 单句 + metadata", v_meta),
    ]:
        spans = chunks_by_window(sents, vecs, MAX_CHARS, WINDOW_CHARS)
        sizes = [sum(len(s) for s in sents[a:b]) for a, b in spans]
        cuts = [1.0 - unit(vecs[b - 1], vecs[b]) for a, b in spans[:-1] if b < n]
        print("  %-24s %6d %10d %10.0f %12.4f"
              % (name, len(spans), min(sizes), sum(sizes) / len(sizes),
                 float(np.mean(cuts)) if cuts else 0.0))
    print()
    print("  注: 三种配置的距离尺度不同, 切点均距不能横向比大小")
    print()

    # ---------- 展示单句嵌入的实际分块结果 ----------
    spans = chunks_by_window(sents, v_sent, MAX_CHARS, WINDOW_CHARS)
    print("=" * 82)
    print("[3] 单句嵌入的分块结果 (前 5 块)")
    print("=" * 82)
    for k, (a, b) in enumerate(spans[:5], 1):
        seg = "".join(sents[a:b])
        print("  块%d  句 %d~%d   %d 字" % (k, a, b - 1, len(seg)))
        print("       开头: %s" % seg[:56].replace("\n", " "))
        if b < n:
            d = 1.0 - unit(v_sent[b - 1], v_sent[b])
            print("       切点距离 %.4f   下一块从: %s" % (d, sents[b][:36].replace("\n", " ")))
        print()


if __name__ == "__main__":
    main()
