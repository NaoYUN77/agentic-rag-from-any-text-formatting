r"""独立稀疏检索完整链路: PDF/Markdown -> 清洗 -> 简单分块 -> jieba -> BM25。

这个脚本刻意不接入现有的语义分块和 Qdrant 链路, 目的是一次只看清楚
稀疏检索自己的工程步骤:

    PDF
      -> MinerU/Markdown
      -> 基础清洗
      -> 标题感知的简单段落分块
      -> jieba 分析器
      -> 词表 term -> id
      -> 倒排索引 + BM25 稀疏权重
      -> query 分析 -> posting list -> top-k

这里的“分块”只是为了给 BM25 demo 提供可检索单元, 不是最终的 chunking
策略。后续要接生产链路时, 可以只替换 chunk 来源, 分析器和 BM25 仍然复用。

用法:
    cd pipeline

    # 已有 Markdown
    .\.venv_rag\Scripts\python.exe sparse_pipeline_demo.py `
        --md corpus/redhat_p1-20.md `
        --query "你的问题" `
        --top-k 5

    # PDF: 默认请求 MinerU 解析全部页面
    .\.venv_rag\Scripts\python.exe sparse_pipeline_demo.py `
        --pdf some.pdf `
        --query "你的问题" `
        --out-dir experiments/14_sparse_pipeline_artifacts

    # 复用现有 dense 链路已经切好并写入 Qdrant 的 chunk
    .\.venv_rag\Scripts\python.exe sparse_pipeline_demo.py `
        --qdrant `
        --qdrant-path qdrant_data `
        --collection redhat `
        --query "你的问题"

注意:
    MinerU flash-extract 可能有服务端的页数限制。需要真正处理所有页时,
    应使用本地 MinerU 或带 token 的 extract 模式。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jieba

from pdf_loader import clean as clean_markdown
from pdf_loader import extract as extract_pdf
from pdf_loader import load_markdown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 这些是演示用领域词。真实项目通常从术语表或人工审核结果生成。
DOMAIN_WORDS = [
    "keepalived",
    "haproxy",
    "virtual_ipaddress",
    "virtual_router_id",
    "keepalived.conf",
    "故障转移",
    "负载均衡",
    "虚拟服务器",
    "健康检查",
    "高可用",
    "防火墙标记",
]

for word in DOMAIN_WORDS:
    jieba.add_word(word)


# 只去掉代词、连接词等低信息词。BM25 自己的 IDF 会继续降低常见词权重。
STOPWORDS = {
    "的", "了", "和", "与", "或", "在", "是", "为", "对", "中", "上", "下",
    "这", "那", "有", "及", "等", "一个", "一种", "以及", "可以", "需要",
    "如果", "因为", "所以", "进行", "通过", "使用",
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is",
    "are", "be", "as", "by", "with", "from", "this", "that",
}

# 先保护配置项、文件路径、版本号等技术 token, 再把中文片段交给 jieba。
BLOCK_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/+\-]*|[\u4e00-\u9fff]+")
ASCII_TOKEN_RE = re.compile(r"^[a-z0-9_][a-z0-9_.:/+\-]*$")
CHINESE_TOKEN_RE = re.compile(r"^[\u4e00-\u9fff]+$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
EDGE_CHARS = " \t\r\n.,;:!?()[]{}<>\"'`，。；：！？（）【】《》"


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    order: int
    text: str
    index_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    point_id: Optional[int | str] = None


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    matched_terms: List[str]
    section: str
    text: str


def normalize_text(text: str) -> str:
    """统一全半角、大小写和常见空白。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    return text.lower()


def analyze(text: str, keep_stopwords: bool = False) -> List[str]:
    """jieba + 技术 token 保护构成的稀疏检索分析器。

    这里输出的不是 Hugging Face token id, 而是可供 BM25 使用的词项。
    英文配置项和路径保持完整, 中文片段交给 jieba。
    """
    normalized = normalize_text(text)
    terms: List[str] = []

    for block in BLOCK_RE.findall(normalized):
        if ASCII_TOKEN_RE.fullmatch(block):
            pieces = [block]
        else:
            pieces = list(jieba.cut(block, cut_all=False))

        for raw in pieces:
            term = raw.strip(EDGE_CHARS)
            if not term:
                continue
            if not keep_stopwords and term in STOPWORDS:
                continue
            if term.isdigit() and len(term) < 2:
                continue
            if len(term) == 1 and CHINESE_TOKEN_RE.fullmatch(term):
                continue
            if not (ASCII_TOKEN_RE.fullmatch(term) or CHINESE_TOKEN_RE.fullmatch(term)):
                continue
            terms.append(term)

    return terms


def split_paragraphs(markdown: str) -> List[str]:
    """按空行切段; 标题保留为独立段落。"""
    parts = re.split(r"\n\s*\n", markdown)
    return [part.strip() for part in parts if part.strip()]


def split_oversized(text: str, max_chars: int) -> List[str]:
    """单个超长段落没有自然边界时, 做最简单的字符窗口切分。"""
    if max_chars <= 0:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def build_chunks(markdown: str, source: str, max_chars: int = 800) -> List[Chunk]:
    """标题感知的段落打包。

    这是独立实验分块, 不复用 semantic_chunker_demo / hybrid_chunk_demo。
    目标只是让每个 BM25 文档有稳定的文本边界和 section metadata。
    """
    paragraphs = split_paragraphs(markdown)
    source_id = re.sub(r"[^0-9A-Za-z_-]+", "_", Path(source).stem).strip("_") or "doc"

    chunks: List[Chunk] = []
    section_stack: List[str] = []
    current_parts: List[str] = []
    current_section = ""
    order = 0

    def flush() -> None:
        nonlocal current_parts, current_section, order
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        if not text:
            current_parts = []
            return
        section = current_section or "文档开头"
        index_text = f"{section}\n{text}" if section != "文档开头" else text
        chunks.append(Chunk(
            id=f"{source_id}_{order:04d}",
            source=source,
            section=section,
            order=order,
            text=text,
            index_text=index_text,
            metadata={
                "file_name": source,
                "section": section,
                "chunk_index": order,
                "char_count": len(text),
            },
        ))
        order += 1
        current_parts = []

    for para in paragraphs:
        heading = HEADING_RE.match(para)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while len(section_stack) >= level:
                section_stack.pop()
            section_stack.append(title)
            current_section = " > ".join(section_stack)
            continue

        section = " > ".join(section_stack) if section_stack else "文档开头"
        if current_parts and section != current_section:
            flush()
            current_section = section
        elif not current_parts:
            current_section = section

        if len(para) > max_chars:
            flush()
            for piece in split_oversized(para, max_chars):
                current_parts = [piece]
                current_section = section
                flush()
            continue

        candidate_len = len("\n\n".join(current_parts + [para]))
        if current_parts and candidate_len > max_chars:
            flush()
        current_parts.append(para)

    flush()
    return chunks


class SparseBM25Index:
    """BM25 词表、倒排索引和 sparse vector 的同一份实现。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [analyze(chunk.index_text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        vocabulary = sorted({term for tokens in self.doc_tokens for term in tokens})
        self.term_to_id: Dict[str, int] = {term: i for i, term in enumerate(vocabulary)}
        self.id_to_term: Dict[int, str] = {i: term for term, i in self.term_to_id.items()}

        self.df: Counter[str] = Counter()
        self.postings: Dict[str, List[int]] = defaultdict(list)
        self.doc_sparse: List[Dict[int, float]] = []

        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            sparse: Dict[int, float] = {}
            dl = max(len(tokens), 1)

            for term, freq in counts.items():
                term_id = self.term_to_id[term]
                self.df[term] += 1
                self.postings[term].append(doc_idx)
                tf_weight = (
                    freq * (self.k1 + 1.0)
                    / (freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
                )
                sparse[term_id] = tf_weight

            self.doc_sparse.append(sparse)

        self._idf_cache: Dict[str, float] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.term_to_id)

    @property
    def nonzero_count(self) -> int:
        return sum(len(vector) for vector in self.doc_sparse)

    def idf(self, term: str) -> float:
        if term not in self._idf_cache:
            df = self.df.get(term, 0)
            n = len(self.chunks)
            self._idf_cache[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return self._idf_cache[term]

    def query_sparse(self, query: str) -> Tuple[Dict[int, float], List[str], List[str]]:
        """返回 query sparse vector、已知词项和不在词表中的词项。"""
        tokens = analyze(query)
        counts = Counter(tokens)
        sparse: Dict[int, float] = {}
        known: List[str] = []
        unknown: List[str] = []

        for term, qtf in counts.items():
            if term not in self.term_to_id:
                unknown.append(term)
                continue
            known.append(term)
            sparse[self.term_to_id[term]] = self.idf(term) * qtf

        return sparse, sorted(known), sorted(unknown)

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        query_vector, known, _ = self.query_sparse(query)
        if not query_vector:
            return []

        candidates = set()
        for term in known:
            candidates.update(self.postings.get(term, []))

        hits: List[SearchHit] = []
        for doc_idx in candidates:
            doc_vector = self.doc_sparse[doc_idx]
            score = 0.0
            matched_ids: List[int] = []
            for term_id, query_weight in query_vector.items():
                doc_weight = doc_vector.get(term_id, 0.0)
                if doc_weight:
                    score += query_weight * doc_weight
                    matched_ids.append(term_id)
            if score <= 0:
                continue
            chunk = self.chunks[doc_idx]
            hits.append(SearchHit(
                chunk_id=chunk.id,
                score=score,
                matched_terms=sorted(self.id_to_term[i] for i in matched_ids),
                section=chunk.section,
                text=chunk.text,
            ))

        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:top_k]

    def describe_sparse(self, doc_idx: int) -> Tuple[List[int], List[float]]:
        vector = self.doc_sparse[doc_idx]
        indices = sorted(vector)
        return indices, [vector[i] for i in indices]


def load_markdown_source(
    md_path: Optional[str],
    pdf_path: Optional[str],
    pdf_pages: Optional[str],
    pdf_language: str,
    save_markdown: Optional[str],
    pdf_timeout: int,
) -> Tuple[str, str, str]:
    """返回 source_name、Markdown 和来源类型。"""
    if md_path:
        path = Path(md_path)
        return path.name, load_markdown(str(path), do_clean=True), "markdown"

    if not pdf_path:
        raise ValueError("必须提供 --md 或 --pdf")

    path = Path(pdf_path)
    markdown = extract_pdf(
        str(path),
        out_path=save_markdown,
        pages=pdf_pages,
        language=pdf_language,
        timeout=pdf_timeout,
    )
    return path.name, clean_markdown(markdown), "pdf/mineru"


def load_qdrant_chunks(
    qdrant_path: str,
    collection: str,
    limit: int = 10000,
) -> Tuple[List[Chunk], str, str]:
    """读取现有 Qdrant payload, 复用 dense 链路已经切好的 chunk。"""
    from qdrant_client import QdrantClient

    path = Path(qdrant_path)
    if not path.exists():
        raise FileNotFoundError(f"Qdrant 路径不存在: {path}")

    try:
        client = QdrantClient(path=str(path))
    except RuntimeError as exc:
        raise RuntimeError(
            "Qdrant 本地目录正在被其他进程占用。请先停止 FastAPI 服务, "
            "再读取 chunk。"
        ) from exc

    try:
        if not client.collection_exists(collection):
            raise RuntimeError(f"Qdrant collection 不存在: {collection}")
        points, _ = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    finally:
        client.close()

    points = sorted(
        points,
        key=lambda point: (
            int((point.payload or {}).get("chunk_index", 10**9)),
            str(point.id),
        ),
    )

    chunks: List[Chunk] = []
    for order, point in enumerate(points):
        payload = dict(point.payload or {})
        text = str(payload.get("text", "")).strip()
        if not text:
            continue

        section = str(payload.get("section") or "文档开头")
        source = str(payload.get("file_name") or f"qdrant:{collection}")
        metadata = dict(payload)
        metadata.pop("text", None)
        index_text = f"{section}\n{text}" if section != "文档开头" else text

        chunks.append(Chunk(
            id=str(point.id),
            source=source,
            section=section,
            order=order,
            text=text,
            index_text=index_text,
            metadata=metadata,
            point_id=point.id,
        ))

    if not chunks:
        raise RuntimeError(f"Qdrant collection {collection} 没有可用 chunk")

    return chunks, f"qdrant:{collection}", "qdrant"


def save_artifacts(
    out_dir: Path,
    source: str,
    chunks: Sequence[Chunk],
    index: SparseBM25Index,
    query_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            row = asdict(chunk)
            row["tokens"] = analyze(chunk.index_text)
            indices, values = index.describe_sparse(chunk.order)
            row["sparse_indices"] = indices
            row["sparse_values"] = values
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "vocab.json").open("w", encoding="utf-8") as fh:
        json.dump(index.term_to_id, fh, ensure_ascii=False, indent=2)

    postings = {
        term: [chunks[i].id for i in doc_ids]
        for term, doc_ids in sorted(index.postings.items())
    }
    with (out_dir / "postings.json").open("w", encoding="utf-8") as fh:
        json.dump(postings, fh, ensure_ascii=False, indent=2)

    bm25_stats = {
        "k1": index.k1,
        "b": index.b,
        "avgdl": index.avgdl,
        "idf": {term: index.idf(term) for term in sorted(index.term_to_id)},
    }
    with (out_dir / "bm25_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(bm25_stats, fh, ensure_ascii=False, indent=2)

    summary = {
        "source": source,
        "chunks": len(chunks),
        "vocabulary_size": index.vocab_size,
        "avg_doc_length_tokens": round(index.avgdl, 3),
        "doc_sparse_nonzero": index.nonzero_count,
        "sparse_density": round(
            index.nonzero_count / max(len(chunks) * index.vocab_size, 1), 6
        ),
        "queries_run": query_count,
        "bm25": {"k1": index.k1, "b": index.b},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def store_sparse_collection(
    qdrant_path: str,
    collection: str,
    chunks: Sequence[Chunk],
    index: SparseBM25Index,
    rebuild: bool = False,
) -> int:
    """把 BM25 sparse vector 写入 Qdrant 的独立 sparse collection。"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector, SparseVectorParams

    path = Path(qdrant_path)
    if not path.exists():
        raise FileNotFoundError(f"Qdrant 路径不存在: {path}")

    try:
        client = QdrantClient(path=str(path))
    except RuntimeError as exc:
        raise RuntimeError(
            "Qdrant 本地目录正在被其他进程占用。请先停止 FastAPI 服务, "
            "再写入 sparse collection。"
        ) from exc

    try:
        exists = client.collection_exists(collection)
        if exists and rebuild:
            client.delete_collection(collection)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=collection,
                vectors_config={},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

        points = []
        for doc_idx, chunk in enumerate(chunks):
            indices, values = index.describe_sparse(doc_idx)
            payload = dict(chunk.metadata)
            payload["text"] = chunk.text
            payload["source_point_id"] = chunk.point_id
            points.append(PointStruct(
                id=chunk.point_id if chunk.point_id is not None else doc_idx,
                vector={"sparse": SparseVector(indices=indices, values=values)},
                payload=payload,
            ))

        client.upsert(collection_name=collection, points=points, wait=True)
        return int(client.count(collection_name=collection, exact=True).count)
    finally:
        client.close()


def print_report(
    source: str,
    source_kind: str,
    source_chars: int,
    chunks: Sequence[Chunk],
    index: SparseBM25Index,
    queries: Sequence[str],
    top_k: int,
    show_vocab: int,
) -> None:
    print("=" * 80)
    print("独立 sparse 完整链路")
    print("=" * 80)
    print(f"来源类型   : {source_kind}")
    print(f"来源       : {source}")
    print(f"文本规模   : {source_chars} 字")
    print(f"chunk 数   : {len(chunks)}")
    print(f"词表大小   : {index.vocab_size}")
    print(f"平均长度   : {index.avgdl:.2f} token/chunk")
    print(f"非零权重   : {index.nonzero_count}")
    density = index.nonzero_count / max(len(chunks) * index.vocab_size, 1)
    print(f"稀疏密度   : {density:.6f}")
    print(f"BM25       : k1={index.k1}, b={index.b}")
    print()

    print("--- 词表示例 ---")
    for term, term_id in list(index.term_to_id.items())[:show_vocab]:
        print(f"{term_id:>5}  {term}")
    print()

    if chunks:
        print("--- chunk 与 sparse vector 示例 ---")
        chunk = chunks[0]
        tokens = analyze(chunk.index_text)
        indices, values = index.describe_sparse(0)
        print(f"chunk_id : {chunk.id}")
        print(f"section  : {chunk.section}")
        print(f"text     : {chunk.text[:180]}")
        print(f"tokens   : {tokens[:40]}")
        print(f"indices  : {indices[:40]}")
        print(f"values   : {[round(v, 4) for v in values[:40]]}")
        print()

    for query in queries:
        query_vector, known, unknown = index.query_sparse(query)
        hits = index.search(query, top_k=top_k)
        print("=" * 80)
        print(f"query      : {query}")
        print(f"terms      : {analyze(query)}")
        print(f"known      : {known}")
        print(f"OOV        : {unknown}")
        print(f"query sparse: {sorted(query_vector.items())}")
        print("-" * 80)
        if not hits:
            print("没有命中。")
        for rank, hit in enumerate(hits, 1):
            snippet = re.sub(r"\s+", " ", hit.text)[:160]
            print(
                f"#{rank} score={hit.score:.4f} id={hit.chunk_id} "
                f"section={hit.section}"
            )
            print(f"   terms: {hit.matched_terms[:20]}")
            print(f"   text : {snippet}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立 sparse/BM25 完整链路")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--md", help="Markdown 输入")
    source.add_argument("--pdf", help="PDF 输入, 通过 MinerU 解析")
    source.add_argument("--qdrant", action="store_true",
                        help="复用现有 Qdrant collection 里的 chunk")

    parser.add_argument("--qdrant-path", default="qdrant_data",
                        help="Qdrant 本地存储目录")
    parser.add_argument("--collection", default="redhat",
                        help="Qdrant collection 名称")
    parser.add_argument("--write-sparse", action="store_true",
                        help="把 sparse vector 写入 Qdrant")
    parser.add_argument("--sparse-collection", default="redhat_sparse",
                        help="sparse vector 的目标 collection")
    parser.add_argument("--rebuild-sparse", action="store_true",
                        help="写入前重建 sparse collection")

    parser.add_argument("--pdf-pages", default=None,
                        help="PDF 页码, 如 1-20; 默认解析全部页面")
    parser.add_argument("--pdf-language", default="ch")
    parser.add_argument("--pdf-timeout", type=int, default=900)
    parser.add_argument("--save-markdown", default=None,
                        help="PDF 解析后另存 Markdown")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", default=[],
                        help="查询文本, 可重复传入")
    parser.add_argument("--out-dir", default=None,
                        help="保存 chunks/vocab/postings 等实验产物")
    parser.add_argument("--show-vocab", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qdrant:
        chunks, source, source_kind = load_qdrant_chunks(
            qdrant_path=args.qdrant_path,
            collection=args.collection,
        )
        source_chars = sum(len(chunk.text) for chunk in chunks)
    else:
        source, markdown, source_kind = load_markdown_source(
            md_path=args.md,
            pdf_path=args.pdf,
            pdf_pages=args.pdf_pages,
            pdf_language=args.pdf_language,
            save_markdown=args.save_markdown,
            pdf_timeout=args.pdf_timeout,
        )
        chunks = build_chunks(markdown, source=source, max_chars=args.max_chars)
        source_chars = len(markdown)

    index = SparseBM25Index(chunks)
    print_report(
        source=source,
        source_kind=source_kind,
        source_chars=source_chars,
        chunks=chunks,
        index=index,
        queries=args.query,
        top_k=args.top_k,
        show_vocab=args.show_vocab,
    )

    if args.out_dir:
        out_dir = Path(args.out_dir)
        save_artifacts(
            out_dir=out_dir,
            source=source,
            chunks=chunks,
            index=index,
            query_count=len(args.query),
        )
        print(f"实验产物已写入: {out_dir}")

    if args.write_sparse or args.rebuild_sparse:
        stored = store_sparse_collection(
            qdrant_path=args.qdrant_path,
            collection=args.sparse_collection,
            chunks=chunks,
            index=index,
            rebuild=args.rebuild_sparse,
        )
        print(
            f"sparse collection 已写入: {args.sparse_collection} "
            f"({stored} points)"
        )


if __name__ == "__main__":
    main()
