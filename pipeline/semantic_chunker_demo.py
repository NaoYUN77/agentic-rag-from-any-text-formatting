"""SemanticChunker 语义分块演示。

演示 LangChain `SemanticChunker` 的工作流程:
    句子分割 -> 上下文窗口 -> 嵌入 -> 相邻距离 -> 阈值判定 -> 按断点成块

语料: 10 句人工构造的中文句子, 话题从"检索"渐变为"向量"再渐变为"分块"。
相邻句之间始终有词汇交集, 但首尾已经完全不谈同一件事, 因此可以清楚看到
分块器在哪里落下断点。

运行:
    python semantic_chunker_demo.py                    # 默认: 阿里云百炼 qwen3-vl-embedding
    python semantic_chunker_demo.py --backend local    # 离线哈希(仅验证逻辑, 无语义)
    python semantic_chunker_demo.py --backend openai   # OpenAI 或兼容端点

默认后端需要环境变量 DASHSCOPE_API_KEY(阿里云百炼 API Key)。
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from typing import List, Sequence

import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

# Windows 控制台默认可能是 GBK, 强制 UTF-8 以免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. 语料: 10 句呈现渐变语义链的中文句子
# --------------------------------------------------------------------------
SENTENCES: List[str] = [
    "RAG 的第一步是从外部知识库里检索出与问题相关的文档。",
    "检索质量直接决定了最终回答能拿到什么样的材料。",
    "传统检索依靠 BM25 这类关键词算法给文档打分排序。",
    "BM25 的核心是词频和逆文档频率，本质上是字面匹配。",
    "关键词匹配擅长精确命中，却难以理解同义表达。",
    "为了解决这个问题，现代检索会先把文本转换成语义向量。",
    "向量是嵌入模型把一段文本映射成的固定长度数字数组。",
    "两个向量的夹角越小，说明它们的语义越接近。",
    "有了向量之后，如何切分文本就成了影响检索粒度的关键。",
    "块太大会混入多个主题稀释语义，块太小又会丢失上下文。",
]

# 中文不能用默认的 (?<=[.?!])\s+, 它只认英文标点。
# 也不能反过来要求"句号后必须跟空白" —— PDF/Markdown 抽出来的文本里,
# 句号后经常直接跟汉字(例如 "路由器上运行。所有运行"), 那样会漏切掉近一半句子。
#
# 这里的正则: 在句末标点之后切分, 顺带吃掉紧随的空白, 且要求后面还有非空白内容
# (最后一条是为了避免文本正好以句号结尾时, 切出一个空片段)。
CN_SENTENCE_SPLIT_REGEX = r"(?<=[。！？；])\s*(?=\S)"

DEMO_TEXT = "\n".join(SENTENCES)


# --------------------------------------------------------------------------
# 2. 离线确定性嵌入: 让 demo 在没有网络和 API key 时也能跑通
# --------------------------------------------------------------------------
def split_latin_words(text: str) -> List[str]:
    """抽出文本中的拉丁词和数字。"""
    out: List[str] = []
    buf: List[str] = []
    for ch in text:
        if ch.isascii() and ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


class HashingEmbeddings(Embeddings):
    """基于字符 n-gram 哈希的词法嵌入。

    这**不是**真正的语义模型, 只捕捉字面重叠: 共享词/字越多的句子越相似。
    对这份语料够用 -- 相邻句共享词汇, 跨主题的句子几乎不共享词汇, 因此
    断点位置和真实嵌入模型基本一致。

    真实项目请换用 OpenAI / BGE / Jina 等嵌入模型, 见 build_embeddings()。
    """

    # 高频虚词会稀释区分度, 去掉后相似度更聚焦在实词上
    STOP_CHARS = set("的是了在和与及就都也很把被这那有些会能可以为对从到并但而")

    def __init__(self, dim: int = 1024, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def _tokens(self, text: str) -> List[str]:
        words = [w.lower() for w in split_latin_words(text)]
        chars = [
            c
            for c in text
            if "\u4e00" <= c <= "\u9fff" and c not in self.STOP_CHARS
        ]
        if len(chars) <= 1:
            grams = chars
        else:
            grams = [
                "".join(chars[i : i + self.ngram])
                for i in range(len(chars) - self.ngram + 1)
            ]
        return words + grams

    def _hash_index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = self._tokens(text)
        if not tokens:
            return vec.tolist()
        counts: dict = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for token, tf in counts.items():
            weight = 1.0 + math.log(tf)  # 次线性词频, 防高频词主导
            vec[self._hash_index(token)] += weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


QWEN_MULTIMODAL_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)


class QwenVLEmbeddings(Embeddings):
    """阿里云百炼 qwen3-vl-embedding 的文本嵌入封装。

    这是**真正的语义模型**, 和上面的 HashingEmbeddings(纯词法)不是一回事。
    它走的是百炼「多模态向量」原生接口, 不是 OpenAI 兼容接口:

        POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding

    参数:
        dimension: qwen3-vl-embedding 支持 2560(默认)/2048/1536/1024/768/512/256
        enable_fusion: 这里保持 False, 每个输入各自返回一个向量;
                       True 则把所有输入融合成 1 个向量(图文混合检索用)
        instruct: 可选任务说明, 官方建议用英文, 通常能提升 1%-5%
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        dimension: int = 1024,
        instruct: str | None = None,
        api_url: str = QWEN_MULTIMODAL_ENDPOINT,
        batch_size: int = 10,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.instruct = instruct
        self.api_url = api_url
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self._cache: dict = {}

    def _call(self, contents: List[dict]) -> List[List[float]]:
        parameters: dict = {"dimension": self.dimension}
        if self.instruct:
            parameters["instruct"] = self.instruct
        payload = {
            "model": self.model,
            "input": {"contents": contents},
            "parameters": parameters,
        }
        resp = requests.post(
            self.api_url,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                "百炼接口返回 %s: %s" % (resp.status_code, resp.text[:500])
            )
        embeddings = resp.json()["output"]["embeddings"]
        embeddings.sort(key=lambda e: e["index"])
        return [e["embedding"] for e in embeddings]

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        # demo 各小节会反复嵌入相同的窗口, 缓存可以把付费接口的调用次数
        # 从约 180 次压到 50 次左右。生产环境请把缓存换成持久化的向量库。
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            if len(batch) == 1:
                vectors = self._call([{"text": batch[0]}])
            else:
                # 官方文档只写了"支持单段文本", 多段合并能否被接受没有明说,
                # 所以失败或数量对不上时自动退化为逐条调用, 保证一定能跑通。
                vectors = []
                try:
                    vectors = self._call([{"text": t} for t in batch])
                except (RuntimeError, requests.RequestException):
                    vectors = []
                if len(vectors) != len(batch):
                    vectors = [self._call([{"text": t}])[0] for t in batch]
            for t, v in zip(batch, vectors):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]


def build_embeddings(backend: str) -> Embeddings:
    """按 backend 构造嵌入函数。"""
    if backend == "local":
        return HashingEmbeddings()

    if backend == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit(
                "缺少 DASHSCOPE_API_KEY 环境变量, 请先设置:\n"
                '  $env:DASHSCOPE_API_KEY="sk-你的百炼Key"'
            )
        return QwenVLEmbeddings(
            api_key=api_key,
            model=os.getenv("QWEN_EMBED_MODEL", "qwen3-vl-embedding"),
            dimension=int(os.getenv("QWEN_EMBED_DIM", "1024")),
            instruct=os.getenv("QWEN_EMBED_INSTRUCT") or None,
        )

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        # 兼容 OpenAI 官方以及任何 OpenAI 协议的服务(智谱/通义等)
        kwargs: dict = {
            "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        }
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        if os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        elif os.getenv("ZAI_API_KEY"):
            kwargs["api_key"] = os.environ["ZAI_API_KEY"]
            kwargs["base_url"] = "https://open.bigmodel.cn/api/paas/v4/"
            kwargs["model"] = os.getenv("EMBEDDING_MODEL", "embedding-3")
        return OpenAIEmbeddings(**kwargs)

    raise ValueError("unknown backend: " + backend)


# --------------------------------------------------------------------------
# 3. 诊断: 把中间结果打出来, 方便对照原理
# --------------------------------------------------------------------------
def semantic_sanity_check(embeddings: Embeddings) -> None:
    """用几组对照句快速判断嵌入是"真语义"还是"纯词法"。"""
    pairs = [
        ("跨语言同义", "汽车", "automobile"),
        ("同义改写", "如何部署系统", "安装步骤如下"),
        ("同义改写", "退货流程", "退换货怎么走"),
        ("完全不相关", "如何部署系统", "今天天气不错"),
    ]
    # 去重后一次性批量嵌入, 而不是逐条调用接口:
    # 8 个词 1 次请求, 比 8 次请求少 7 个网络往返。
    unique = list(dict.fromkeys(t for _, a, b in pairs for t in (a, b)))
    vectors = embeddings.embed_documents(unique)
    cache = {t: np.array(v, dtype=np.float64) for t, v in zip(unique, vectors)}
    print("=" * 78)
    print("嵌入质量速检(语义相近应得高分, 不相关应得低分)")
    print("=" * 78)
    print("  向量维度: %d" % len(cache[pairs[0][1]]))
    for label, a, b in pairs:
        va, vb = cache[a], cache[b]
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        cos = float(va @ vb / denom) if denom > 0 else 0.0
        print("  %-10s %-12s | %-12s -> %.3f" % (label, a, b, cos))
    print()


def print_corpus(sentences: List[str], label: str, max_show: int = 12) -> None:
    print("=" * 78)
    print("语料: %s" % label)
    print("=" * 78)
    print("  共 %d 句, %d 字" % (len(sentences), sum(len(s) for s in sentences)))
    if len(sentences) <= max_show:
        for i, s in enumerate(sentences):
            print("  S%-3d %s" % (i + 1, s))
    else:
        half = max_show // 2
        for i in range(half):
            print("  S%-3d %s" % (i + 1, sentences[i][:74]))
        print("  ...  中间省略 %d 句  ..." % (len(sentences) - max_show))
        for i in range(len(sentences) - half, len(sentences)):
            print("  S%-3d %s" % (i + 1, sentences[i][:74]))
    print()


def diagnose(chunker: SemanticChunker, text: str, max_show: int = 20) -> List[float]:
    """打印句子切分结果和相邻距离序列(用私有方法, 仅用于教学展示)。"""
    sentences = chunker._get_single_sentences_list(text)
    print("句子切分结果: 共 %d 句" % len(sentences))
    print()

    distances, sent_dicts = chunker._calculate_sentence_distances(sentences)
    print("上下文窗口示例 (buffer_size=%d, 共 %d 个):" % (chunker.buffer_size, len(sent_dicts)))
    for i, d in enumerate(sent_dicts[:max_show]):
        print("  W%-4d %s" % (i, d["combined_sentence"][:74]))
    if len(sent_dicts) > max_show:
        print("  ... 其余 %d 个窗口省略 ..." % (len(sent_dicts) - max_show))
    print()

    print("相邻窗口的余弦距离 d(i, i+1) —— 共 %d 个:" % len(distances))
    show = min(len(distances), max_show * 2)
    for i, dist in enumerate(distances[:show]):
        bar = "#" * max(int(dist * 60), 1)
        print("  d(%d, %d) = %.4f  %s" % (i, i + 1, dist, bar))
    if len(distances) > show:
        print("  ... 其余 %d 个距离省略 ..." % (len(distances) - show))
    arr = np.array(distances)
    print(
        "  统计: min=%.4f  Q1=%.4f  中位=%.4f  均值=%.4f  Q3=%.4f  max=%.4f"
        % (
            arr.min(),
            np.percentile(arr, 25),
            np.median(arr),
            arr.mean(),
            np.percentile(arr, 75),
            arr.max(),
        )
    )
    print()
    return list(distances)


def run_one(
    embeddings: Embeddings,
    threshold_type: str,
    amount,
    buffer_size: int,
    text: str,
    verbose: bool = False,
) -> List[str]:
    chunker = SemanticChunker(
        embeddings,
        buffer_size=buffer_size,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=amount,
        sentence_split_regex=CN_SENTENCE_SPLIT_REGEX,
    )
    if verbose:
        distances = diagnose(chunker, text)
        threshold, _ = chunker._calculate_breakpoint_threshold(distances)
        print(
            "阈值(%s, amount=%s) = %.4f"
            % (threshold_type, chunker.breakpoint_threshold_amount, threshold)
        )
        print()
    return chunker.split_text(text)


def print_chunks(chunks: List[str], indent: str = "  ", max_text: int = 0) -> None:
    for i, c in enumerate(chunks, 1):
        body = " ".join(c.split())
        if max_text and len(body) > max_text:
            print("%sChunk %d (%d 字): %s ..." % (indent, i, len(body), body[:max_text]))
        else:
            print("%sChunk %d (%d 字): %s" % (indent, i, len(body), body))


# --------------------------------------------------------------------------
# 4. 语料加载: 内置示例 / Markdown / PDF(MinerU)
# --------------------------------------------------------------------------
def load_corpus(args):
    """返回 (语料文本, 来源说明)。"""
    if args.md:
        from pdf_loader import load_markdown

        return load_markdown(args.md, do_clean=not args.no_clean), "Markdown: %s" % args.md

    if args.pdf:
        from pdf_loader import clean as clean_md
        from pdf_loader import extract

        print("正在用 MinerU 解析 PDF ...")
        print("  文件: %s" % args.pdf)
        print("  页码: %s" % args.pdf_pages)
        raw = extract(
            args.pdf,
            out_path=args.pdf_md,
            pages=args.pdf_pages,
            language=args.pdf_lang,
        )
        text = raw if args.no_clean else clean_md(raw)
        print("  原始 %d 字 -> 清洗后 %d 字" % (len(raw), len(text)))
        print("  中间 Markdown 已保存: %s" % args.pdf_md)
        return text, "PDF: %s" % os.path.basename(args.pdf)

    return DEMO_TEXT, "内置 10 句示例"


# --------------------------------------------------------------------------
# 5. 主流程
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SemanticChunker demo")
    parser.add_argument(
        "--backend",
        choices=["local", "qwen", "openai"],
        default="qwen",
        help="backend: qwen=百炼qwen3-vl-embedding(默认), local=离线哈希(无语义), openai=兼容端点",
    )
    parser.add_argument("--buffer-size", type=int, default=1, help="window radius")
    parser.add_argument("--pdf", help="用 MinerU 解析这个 PDF 作为语料")
    parser.add_argument("--pdf-pages", default="1-20", help="PDF 页码范围, 默认 1-20")
    parser.add_argument("--pdf-lang", default="ch", help="PDF 语言, 默认 ch")
    parser.add_argument("--pdf-md", default=os.path.join("corpus", "extracted.md"),
                        help="中间 Markdown 的输出路径")
    parser.add_argument("--md", help="直接使用已有的 Markdown 作为语料")
    parser.add_argument("--no-clean", action="store_true", help="跳过 Markdown 清洗")
    args = parser.parse_args()

    embeddings = build_embeddings(args.backend)
    print("嵌入后端: %s" % args.backend)
    print()
    semantic_sanity_check(embeddings)

    if args.backend == "local":
        print("!" * 78)
        print("警告: 当前用的是本地哈希函数, 它只匹配字面重叠, 完全不理解语义。")
        print("      上面四组语义速检全是 0.000 就是证据。")
        print("      它可以验证管线逻辑, 但切出来的断点不代表真实语义分块效果。")
        print("      要看真实效果, 请改用 --backend qwen。")
        print("!" * 78)
        print()

    corpus_text, corpus_label = load_corpus(args)
    corpus_sentences = [
        s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, corpus_text) if s.strip()
    ]
    n_sent = len(corpus_sentences)
    max_text = 0 if n_sent <= 30 else 100

    print_corpus(corpus_sentences, corpus_label)

    # ---- 5.1 完整走一遍, 打印所有中间结果 ----
    print("=" * 78)
    print("步骤 1-3: 句子分割 -> 上下文窗口 -> 嵌入 -> 相邻距离")
    print("=" * 78)
    chunks = run_one(
        embeddings, "percentile", None, args.buffer_size, corpus_text, verbose=True
    )
    print("=" * 78)
    print("步骤 4-5: 百分位法找断点 -> 成块 (buffer_size=%d)" % args.buffer_size)
    print("=" * 78)
    print("得到 %d 个 chunk:" % len(chunks))
    print_chunks(chunks, max_text=max_text)
    print()

    # ---- 5.2 四种断点判定方法对比 ----
    print("=" * 78)
    print("四种断点判定方法对比 (buffer_size=%d)" % args.buffer_size)
    print("=" * 78)
    for t in ["percentile", "standard_deviation", "interquartile", "gradient"]:
        cs = run_one(embeddings, t, None, args.buffer_size, corpus_text)
        print("[%s] -> %d 个 chunk" % (t, len(cs)))
        print_chunks(cs, max_text=max_text)
        print()

    # ---- 5.3 buffer_size 的影响 ----
    print("=" * 78)
    print("buffer_size 的影响 (percentile)")
    print("=" * 78)
    for bs in [0, 1, 2, 3]:
        cs = run_one(embeddings, "percentile", None, bs, corpus_text)
        print("buffer_size=%d -> %d 个 chunk" % (bs, len(cs)))
        print_chunks(cs, max_text=max_text)
        print()

    # ---- 5.4 阈值松紧的影响 ----
    print("=" * 78)
    print("breakpoint_threshold_amount 的影响 (percentile, buffer_size=1)")
    print("=" * 78)
    for amt in [50, 80, 90, 95, 99]:
        cs = run_one(embeddings, "percentile", float(amt), 1, corpus_text)
        sizes = [len(c) for c in cs]
        print(
            "amount=%-3d -> %2d 个 chunk   字数 min=%d max=%d 均值=%d"
            % (amt, len(cs), min(sizes), max(sizes), sum(sizes) // len(sizes))
        )
    print()
    print("提示: amount 越大, 阈值越高, 断点越少, 块越大。")
    print()

    # ---- 5.5 number_of_chunks: 直接指定块数 ----
    print("=" * 78)
    print("number_of_chunks: 直接指定想要的块数")
    print("=" * 78)
    targets = [2, 3, 4] if n_sent <= 30 else [3, 5, 8]
    for n in targets:
        chunker = SemanticChunker(
            embeddings,
            buffer_size=1,
            number_of_chunks=n,
            sentence_split_regex=CN_SENTENCE_SPLIT_REGEX,
        )
        cs = chunker.split_text(corpus_text)
        sizes = [len(c) for c in cs]
        print(
            "number_of_chunks=%d -> 实际 %2d 个 chunk   字数 min=%d max=%d"
            % (n, len(cs), min(sizes), max(sizes))
        )
        print_chunks(cs, max_text=70)
        print()

    # ---- 6. 要点回顾 ----
    print("=" * 78)
    print("要点回顾")
    print("=" * 78)
    print("1. 句子切分: 中文必须自定义 sentence_split_regex, 默认只认英文标点")
    print("2. 上下文窗口: buffer_size 越大, 窗口越宽, 距离越被平滑, 断点越容易前移")
    print("3. 嵌入: 每个窗口独立嵌入一次, 10 句 = 10 次调用, 这是语义分块慢的原因")
    print("4. 距离: n 个句子只能产出 n-1 个相邻距离")
    print("5. 阈值: 用相对判断(百分位等), 不能用绝对阈值, 因为距离值都很小且集中")
    print("6. 成块: 断点之间的句子合并, 所以 chunk 通常包含多个句子")
    print("7. 局限: 语义分块不管块大小, 超长块需要自己再套一层递归字符分块兜底")


if __name__ == "__main__":
    main()









