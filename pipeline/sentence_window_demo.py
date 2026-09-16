"""SentenceWindow 检索策略演示。

和 SemanticChunker 的思路正好相反:

    SemanticChunker : 嵌入时把上下文拼进去 -> 检索直接返回窗口文本
                      牺牲匹配精度, 换上下文完整性

    SentenceWindow  : 嵌入时只用单句       -> 检索命中后再把邻居拼回来
                      匹配精度优先, 上下文在检索之后才补上

核心是把"用于匹配的文本"和"送给 LLM 的文本"解耦:
    node.text      = 单句          <- 拿去做嵌入
    node.metadata  = 前后 N 句     <- 检索命中后才拼回来

运行:
    python sentence_window_demo.py                  # 真实模型(百炼 qwen3-vl-embedding)
    python sentence_window_demo.py --backend local   # 离线哈希(仅验证逻辑)
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List, Sequence, Tuple

import numpy as np

from semantic_chunker_demo import build_embeddings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. 语料: 24 句, 分成 5 个话题
# --------------------------------------------------------------------------
DOC = """
部署 RAG 系统之前，需要先确认机器的显存和内存是否够用。
建议为嵌入模型预留至少八 GB 的显存，否则批量推理会频繁失败。
如果显存不足，可以把批大小调小，或者直接换用更轻量的模型。
量化到 INT8 能把显存占用压到原来的三分之一左右，但精度会掉一到两个百分点。
生产环境建议把嵌入服务独立部署成一个进程，避免和生成模型抢显存。
原始文档在入库之前必须清洗，去掉页眉页脚、导航栏和重复段落。
扫描版 PDF 需要先做 OCR，否则提取出来的全是乱码。
分片粒度直接决定检索质量，块太大会稀释语义，块太小会丢失上下文。
通用场景可以从五百一十二个 token 起步，再根据评估结果微调。
块与块之间保留百分之十到二十的重叠，可以避免关键信息被切断。
中文场景推荐使用 bge-m3 或者通义的 text-embedding 系列。
bge-m3 支持稠密、稀疏、多向量三路输出，一个模型就能覆盖混合检索。
向量维度越高表达能力越强，但存储和检索开销也会同步上涨。
更换嵌入模型之后，必须重建整个索引，否则新旧向量无法比较。
建库阶段建议调用批量接口，价格通常只有实时接口的一半。
混合检索把关键词召回和语义召回结合起来，效果通常好于单路。
召回阶段可以多取一些候选，比如五十条，再交给重排序模型精筛。
重排序使用交叉编码器，精度高但速度慢，只适合处理少量候选。
相似度分数只能用于排序，不能当作绝对的相关性阈值来用。
如果召回结果明显跑偏，优先检查查询改写和分片粒度，而不是换模型。
没有评估集的调优都是盲调，至少需要准备一百条标注过的查询。
检索阶段关注召回率和命中率，生成阶段关注忠实度和答案相关性。
忠实度衡量答案有没有编造，答案相关性衡量有没有答非所问。
上线之后要持续监控查询分布漂移，定期用新数据更新评估集。
"""
DOC = DOC.strip()

QUERIES = [
    "显存不够用怎么办",
    "长文档应该怎么切分",
    "怎么判断系统答得好不好",
    "为什么换了模型要重建索引",
]

SENTENCE_SPLIT_RE = r"(?<=[。！？；])"


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(SENTENCE_SPLIT_RE, text) if s.strip()]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# 2. 两种索引: 单句索引 vs 窗口索引
# --------------------------------------------------------------------------
class SentenceWindowIndex:
    """SentenceWindow 做法: 嵌入用单句, 邻居存在 metadata 里。"""

    def __init__(self, embeddings, window: int = 1) -> None:
        self.embeddings = embeddings
        self.window = window
        self.sentences: List[str] = []
        self.vectors: np.ndarray = np.zeros((0, 0))

    def build(self, sentences: Sequence[str]) -> None:
        self.sentences = list(sentences)
        # ★ 关键: 嵌入的是单句, 不带任何上下文
        vecs = self.embeddings.embed_documents(self.sentences)
        self.vectors = np.array(vecs, dtype=np.float64)

    def window_text(self, i: int) -> str:
        """MetadataReplacementPostProcessor 干的事: 命中单句后展开成窗口。"""
        lo = max(0, i - self.window)
        hi = min(len(self.sentences), i + self.window + 1)
        return " ".join(self.sentences[lo:hi])

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        q = np.array(self.embeddings.embed_query(query), dtype=np.float64)
        sims = [cos(self.vectors[i], q) for i in range(len(self.sentences))]
        order = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]


class WindowIndex:
    """对照做法: 嵌入时就把邻居拼进去(就是 SemanticChunker 的思路)。"""

    def __init__(self, embeddings, window: int = 1) -> None:
        self.embeddings = embeddings
        self.window = window
        self.sentences: List[str] = []
        self.vectors: np.ndarray = np.zeros((0, 0))

    def build(self, sentences: Sequence[str]) -> None:
        self.sentences = list(sentences)
        texts = []
        for i in range(len(self.sentences)):
            lo = max(0, i - self.window)
            hi = min(len(self.sentences), i + self.window + 1)
            texts.append(" ".join(self.sentences[lo:hi]))
        # ★ 关键: 嵌入的是拼好的窗口
        vecs = self.embeddings.embed_documents(texts)
        self.vectors = np.array(vecs, dtype=np.float64)

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        q = np.array(self.embeddings.embed_query(query), dtype=np.float64)
        sims = [cos(self.vectors[i], q) for i in range(len(self.sentences))]
        order = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]


# --------------------------------------------------------------------------
# 3. 主流程
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SentenceWindow demo")
    parser.add_argument("--backend", choices=["local", "qwen", "openai"], default="qwen")
    parser.add_argument("--window", type=int, default=1, help="窗口半径 N")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    embeddings = build_embeddings(args.backend)
    sentences = split_sentences(DOC)

    print("嵌入后端: %s    窗口半径 N=%d" % (args.backend, args.window))
    print("=" * 78)
    print("【第 1 步】切句: 共 %d 句" % len(sentences))
    print("=" * 78)
    for i, s in enumerate(sentences):
        print("  S%-2d %s" % (i + 1, s))
    print()

    print("=" * 78)
    print("【第 2 步】建两个索引")
    print("=" * 78)
    sw = SentenceWindowIndex(embeddings, window=args.window)
    wi = WindowIndex(embeddings, window=args.window)
    sw.build(sentences)
    print("  SentenceWindowIndex: 嵌入 %d 条【单句】   -> 向量 %s" % (len(sentences), sw.vectors.shape))
    wi.build(sentences)
    print("  WindowIndex        : 嵌入 %d 条【窗口】   -> 向量 %s" % (len(sentences), wi.vectors.shape))
    print()
    print("  两种索引里, 下标 i 对应的向量是:")
    i = 7
    print("    SentenceWindow -> 嵌入(\"%s\")" % sentences[i][:34])
    print("    Window         -> 嵌入(\"%s...\")" % sw.window_text(i)[:34])
    print()

    for query in QUERIES:
        print("=" * 78)
        print("【第 3 步】查询: %s" % query)
        print("=" * 78)
        print("  --- A. SentenceWindow: 嵌入单句, 命中后再展开窗口 ---")
        hits_sw = sw.search(query, args.top_k)
        for rank, (idx, score) in enumerate(hits_sw, 1):
            print("    #%d  相似度 %.4f" % (rank, score))
            print("        用于匹配的单句 : %s" % sentences[idx])
            print("        ↓ MetadataReplacement 展开为窗口(前%d句 + 本句 + 后%d句)" % (args.window, args.window))
            print("        送给 LLM 的文本 : %s" % sw.window_text(idx))
        print()
        print("  --- B. Window: 嵌入时就带上下文(传统做法) ---")
        hits_wi = wi.search(query, args.top_k)
        for rank, (idx, score) in enumerate(hits_wi, 1):
            print("    #%d  相似度 %.4f   命中 S%d: %s" % (rank, score, idx + 1, sentences[idx]))
        print()
        same = [i for i, _ in hits_sw] == [i for i, _ in hits_wi]
        print("  两种做法命中顺序是否一致: %s" % ("是" if same else "否"))
        if not same:
            print("    SentenceWindow top-1 -> S%d" % (hits_sw[0][0] + 1))
            print("    Window         top-1 -> S%d" % (hits_wi[0][0] + 1))
        print()


if __name__ == "__main__":
    main()
