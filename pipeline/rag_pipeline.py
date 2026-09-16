"""完整的 RAG 入库流程。

    语料清洗(MinerU) -> 分块(固定长度 + 语义微调) -> 向量化 -> 存入 Qdrant

分块策略(沿用之前的实测结论):
    固定长度粗切(max_chars)
      + 在边界前 window_chars 范围内挑语义断点
      + 切点距离用【单句嵌入】算(成本 1/3, 且无位置错位)

向量化:
    入库时嵌入的是 chunk 全文(检索单元), 维度 1024

用法:
    python rag_pipeline.py --md corpus/redhat_p1-20.md
    python rag_pipeline.py --pdf 文件.pdf --pdf-pages 1-20
    python rag_pipeline.py --query "如何配置 keepalived 的故障转移"
    python rag_pipeline.py --rebuild
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from pdf_loader import load_markdown
from preprocess import parse_markdown, summarize
from semantic_chunker_demo import CN_SENTENCE_SPLIT_REGEX, build_embeddings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QDRANT_PATH = "qdrant_data"
COLLECTION = "redhat"
VECTOR_SIZE = 1024
CHARS_PER_PAGE = 900          # 估算页码用(语料没有显式分页标记)


def unit(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# --------------------------------------------------------------------------
# [1] 加载 + 清洗
# --------------------------------------------------------------------------
def load_corpus(args) -> Tuple[str, str]:
    if args.md:
        raw = open(args.md, encoding="utf-8", errors="replace").read()
        text = load_markdown(args.md) if not args.no_clean else raw
        name = os.path.basename(args.md)
        print("  来源: %s" % args.md)
        print("  清洗: %d 字 -> %d 字" % (len(raw), len(text)))
        return text, name
    if args.pdf:
        from pdf_loader import clean as clean_md
        from pdf_loader import extract

        print("  调用 MinerU 解析: %s" % args.pdf)
        raw = extract(args.pdf, out_path=args.pdf_md, pages=args.pdf_pages,
                      language=args.pdf_lang)
        text = raw if args.no_clean else clean_md(raw)
        print("  清洗: %d 字 -> %d 字" % (len(raw), len(text)))
        return text, os.path.basename(args.pdf)
    raise SystemExit("请用 --md 或 --pdf 指定语料")


# --------------------------------------------------------------------------
# [2] 分块: 固定长度 + 边界窗口内挑语义断点
# --------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, text) if s.strip()]


def prepare_sentences(md: str, preprocess: bool = True):
    """把 Markdown 变成 [(句子, 所属章节), ...]。

    preprocess=True 时先做内容分类和结构提取:
        - 剥离 Markdown 标记(标题不进正文)
        - 目录/版权/frontmatter 不参与分块
        - 每句带上干净的章节路径

    preprocess=False 时退回到旧行为(整篇按标点分句), 用于对照。
    """
    if not preprocess:
        return [(s, "文档开头") for s in split_sentences(md)], None

    blocks = parse_markdown(md)
    stat = summarize(blocks)
    pairs = []
    for b in blocks:
        if b.kind != "body":
            continue
        for s in split_sentences(b.text):
            pairs.append((s, b.section))
    return pairs, stat


def chunk_by_window(
    sents: List[str], vecs: np.ndarray, max_chars: int, window_chars: int
) -> List[Tuple[int, int]]:
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
        # 边界窗口: 从 end 往回折算成句
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


def chunk_by_section(
    sents: List[str],
    sections: List[str],
    vecs: np.ndarray,
    max_chars: int,
    window_chars: int,
) -> List[Tuple[int, int]]:
    """以【顶层章节】为硬边界, 章内再按长度+语义分块。

    解决的问题(issues/01 的另一面):
        原来的分块只看长度和语义距离, 不看章节边界,
        结果一个块里混了多个章节的内容 ——
        查询 A 章节的主题, 却召回了含 B 章节内容的块。

    做法:
        1. 取 section 路径的第一段作为"顶层章节"(如"第 2 章 KEEPALIVED 概述")
        2. 按顶层章节切成连续区间
        3. 每个区间内部独立跑 chunk_by_window
    """
    n = len(sents)
    if n == 0:
        return []

    tops = [(s or "").split(" > ")[0].strip() for s in sections]
    out: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and tops[end] == tops[start]:
            end += 1
        sub = chunk_by_window(
            sents[start:end], vecs[start:end], max_chars, window_chars
        )
        out.extend((a + start, b + start) for a, b in sub)
        start = end
    return out

# --------------------------------------------------------------------------
# [3] 生成 metadata
# --------------------------------------------------------------------------
def build_metadatas(
    sents: List[str],
    spans: List[Tuple[int, int]],
    file_name: str,
    sections: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """给每个 chunk 生成 metadata: 文件名 / 估算页码 / 章节 / 序号 / 字数

    sections 由预处理阶段提供(干净的层级章节路径)。
    不传时退化为"文档开头", 保证向后兼容。
    """
    if sections is None:
        sections = ["文档开头"] * len(sents)

    # 累计字符位置 -> 估算页码
    pos = []
    acc = 0
    for s in sents:
        pos.append(acc)
        acc += len(s)

    metas = []
    for idx, (a, b) in enumerate(spans):
        body = "".join(sents[a:b])
        page = pos[a] // CHARS_PER_PAGE + 1
        # chunk 跨越的章节取第一句所属的
        metas.append({
            "file_name": file_name,
            "page": int(page),
            "section": sections[a],
            "chunk_index": idx,
            "char_count": len(body),
            "n_sentences": b - a,
        })
    return metas


# --------------------------------------------------------------------------
# [4] 存入 Qdrant
# --------------------------------------------------------------------------
def store(
    client: QdrantClient, chunks: List[str], metas: List[Dict[str, Any]],
    vectors: List[List[float]], rebuild: bool,
) -> None:
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
            print("  已删除旧集合")
        except Exception:
            pass

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("  创建集合 %s (COSINE, %d 维)" % (COLLECTION, VECTOR_SIZE))

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={"text": chunks[i], **metas[i]},
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print("  写入 %d 个点" % len(points))


# --------------------------------------------------------------------------
# [5] 检索
# --------------------------------------------------------------------------
def search(client: QdrantClient, embeddings, query: str, top_k: int,
           section: str | None = None) -> None:
    qv = embeddings.embed_query(query)
    flt = None
    if section:
        from qdrant_client.models import FieldCondition, Filter, MatchText

        flt = Filter(must=[FieldCondition(key="section", match=MatchText(text=section))])

    res = client.query_points(
        collection_name=COLLECTION, query=qv, limit=top_k,
        query_filter=flt, with_payload=True,
    )
    print("  查询: %s" % query)
    if section:
        print("  过滤: section 包含 %r" % section)
    for rank, h in enumerate(res.points, 1):
        p = h.payload
        print("    #%d  %.4f  [%s p%s] %s" % (
            rank, h.score, p.get("section", "?")[:18], p.get("page"),
            p.get("text", "")[:60].replace("\n", " ")))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 入库流程")
    ap.add_argument("--md")
    ap.add_argument("--pdf")
    ap.add_argument("--pdf-pages", default="1-20")
    ap.add_argument("--pdf-lang", default="ch")
    ap.add_argument("--pdf-md", default=os.path.join("corpus", "pipeline_extracted.md"))
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="跳过内容分类和结构提取(用于对照旧行为)")
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--window-chars", type=int, default=400,
                    help="边界窗口宽度(相当于 overlap), 默认 400 = chunk 的一半")
    ap.add_argument("--rebuild", action="store_true", help="重建集合")
    ap.add_argument("--query", action="append", help="检索测试, 可多次")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--only-search", action="store_true", help="只检索, 不入库")
    args = ap.parse_args()

    embeddings = build_embeddings("qwen")

    # 重建时直接清空存储目录 —— 本地模式的 delete_collection 有残留风险
    # (实测: 删了集合重建后, 旧的高 id 点仍然留在索引里)
    if args.rebuild and os.path.isdir(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
        print("已清空 %s" % QDRANT_PATH)

    client = QdrantClient(path=QDRANT_PATH)

    try:
        if not args.only_search:
            print("=" * 74)
            print("[1/5] 加载语料 + 清洗")
            print("=" * 74)
            text, file_name = load_corpus(args)

            use_pre = not args.no_preprocess
            pairs, prep_stat = prepare_sentences(text, preprocess=use_pre)
            sents = [s for s, _ in pairs]
            sent_sections = [sec for _, sec in pairs]

            if prep_stat:
                print("  预处理: 内容分类 + 结构提取")
                k = prep_stat["by_kind"]
                print("    body %d 块 / toc %d / copyright %d / frontmatter %d"
                      % (k.get("body", 0), k.get("toc", 0),
                         k.get("copyright", 0), k.get("frontmatter", 0)))
                print("    正文 %d 字(剔除非正文 %d 字), 涉及 %d 个章节"
                      % (prep_stat["chars_by_kind"].get("body", 0),
                         sum(v for kk, v in prep_stat["chars_by_kind"].items() if kk != "body"),
                         prep_stat["sections"]))
            else:
                print("  预处理: 已跳过(--no-preprocess)")
            print("  分句: %d 句" % len(sents))

            print()
            print("=" * 74)
            print("[2/5] 分块")
            print("=" * 74)
            print("  策略: 固定长度 + 边界窗口内挑语义断点")
            print("  参数: max_chars=%d  window_chars=%d" % (args.max_chars, args.window_chars))
            print("  嵌入: 单句嵌入(仅用于算切点距离)")
            vecs = np.array(embeddings.embed_documents(sents), dtype=np.float64)
            if use_pre:
                spans = chunk_by_section(sents, sent_sections, vecs,
                                         args.max_chars, args.window_chars)
                print("  边界: 以顶层章节为硬边界(块不跨章)")
            else:
                spans = chunk_by_window(sents, vecs, args.max_chars, args.window_chars)
            chunks = ["".join(sents[a:b]) for a, b in spans]
            sizes = [len(c) for c in chunks]
            print("  结果: %d 个 chunk, 字数 min=%d 均值=%d max=%d"
                  % (len(chunks), min(sizes), sum(sizes) // len(sizes), max(sizes)))

            print()
            print("=" * 74)
            print("[3/5] 生成 metadata")
            print("=" * 74)
            metas = build_metadatas(sents, spans, file_name, sent_sections)
            for m in metas[:3]:
                print("  块%d: %s" % (m["chunk_index"], m))
            print("  ... 共 %d 条" % len(metas))

            print()
            print("=" * 74)
            print("[4/5] 嵌入 chunk + 存入 Qdrant")
            print("=" * 74)
            print("  嵌入 %d 个 chunk ..." % len(chunks))
            cvecs = embeddings.embed_documents(chunks)
            store(client, chunks, metas, cvecs, args.rebuild)

        print()
        print("=" * 74)
        print("[5/5] 检索验证")
        print("=" * 74)
        info = client.get_collection(COLLECTION)
        print("  集合 %s: %d 个点" % (COLLECTION, info.points_count))
        print()

        queries = args.query or [
            "如何配置 keepalived 的故障转移",
            "HAProxy 支持哪些负载均衡算法",
            "VRRP 版本不匹配会有什么问题",
        ]
        for q in queries:
            search(client, embeddings, q, args.top_k)

        # 演示 metadata 过滤
        print("=" * 74)
        print("metadata 过滤示例")
        print("=" * 74)
        search(client, embeddings, "配置步骤", args.top_k, section="KEEPALIVED")

    finally:
        client.close()


if __name__ == "__main__":
    main()



