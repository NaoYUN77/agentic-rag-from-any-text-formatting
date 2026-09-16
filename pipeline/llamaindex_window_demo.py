"""LlamaIndex SentenceWindowNodeParser 窗口机制演示。

演示内容:
    [1] 默认句子切分器在中文上失效
    [2] 换成中文切分器后的正确切分
    [3] 每个 node 的内部结构(text / metadata / excluded keys)
    [4] 建索引 + 检索
    [5] MetadataReplacementPostProcessor 替换前后的对比

运行:
    python llamaindex_window_demo.py                  # 真实嵌入(百炼 qwen3-vl-embedding)
    python llamaindex_window_demo.py --mock           # 结构化演示, 不调接口
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, List

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# 语料: 12 句, 三个话题
# --------------------------------------------------------------------------
DOC = (
    "Keepalived 是运行在 LVS 路由器上的守护进程，负责健康检查与故障转移。"
    "它通过 VRRP 协议在主动和备份路由器之间同步虚拟 IP 地址。"
    "当主路由器失效时，备份路由器会在几秒内接管虚拟 IP，服务几乎不中断。"
    "HAProxy 则工作在应用层，为 TCP 和 HTTP 流量做负载均衡。"
    "它支持多种调度算法，比如轮询、最少连接和源地址哈希。"
    "HAProxy 的健康检查可以精确到 HTTP 状态码，因此比四层检查更可靠。"
    "把 Keepalived 和 HAProxy 组合使用，可以同时获得高可用和负载均衡。"
    "部署时要先规划好虚拟 IP，避免与现有网段冲突。"
    "配置文件修改后必须重启服务，新的路由规则才会生效。"
    "生产环境建议保留至少一台备份节点，并定期演练切换流程。"
    "监控方面要重点关注虚拟 IP 的归属变化和健康检查的失败次数。"
    "日志里如果频繁出现状态切换，通常说明健康检查阈值设置得太敏感。"
)


def zh_splitter(text: str) -> List[str]:
    """中文句子切分。LlamaIndex 默认的 NLTK punkt 对中文无效。"""
    return [s.strip() for s in re.split(r"(?<=[。！？；])\s*(?=\S)", text) if s.strip()]


# --------------------------------------------------------------------------
# 把已有的百炼客户端适配成 LlamaIndex 的 BaseEmbedding
# --------------------------------------------------------------------------
class QwenLIEmbedding(BaseEmbedding):
    api_key: str = ""
    dimension: int = 1024
    _client: Any = PrivateAttr()

    def __init__(self, api_key: str, dimension: int = 1024, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, dimension=dimension, **kwargs)
        from semantic_chunker_demo import QwenVLEmbeddings

        self._client = QwenVLEmbeddings(api_key=api_key, dimension=dimension)
        self._client.batch_size = 10

    @classmethod
    def class_name(cls) -> str:
        return "QwenLIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._client.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def build_parser(use_zh: bool, window_size: int) -> SentenceWindowNodeParser:
    kw = dict(window_size=window_size)
    if use_zh:
        kw["sentence_splitter"] = zh_splitter
    return SentenceWindowNodeParser.from_defaults(**kw)


def show_nodes(parser: SentenceWindowNodeParser, title: str) -> List:
    nodes = parser.get_nodes_from_documents([Document(text=DOC)])
    print("=" * 78)
    print(title)
    print("=" * 78)
    print("  输入: %d 字" % len(DOC))
    print("  输出: %d 个 node" % len(nodes))
    print()
    for i, n in enumerate(nodes):
        print("  node[%d]" % i)
        print("    text            = %s" % n.text[:52])
        print("    metadata:window = %s" % n.metadata.get("window", "")[:52])
        if i == 0:
            print("    excluded_embed  = %s" % n.excluded_embed_metadata_keys)
            print("    excluded_llm    = %s" % n.excluded_llm_metadata_keys)
    print()
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="LlamaIndex SentenceWindow demo")
    ap.add_argument("--mock", action="store_true", help="用 MockEmbedding, 不调真实接口")
    ap.add_argument("--window-size", type=int, default=2)
    args = ap.parse_args()

    # ---------- [1] 默认切分器 ----------
    show_nodes(build_parser(False, args.window_size), "[1] 默认句子切分器 (NLTK punkt) 在中文上")

    # ---------- [2] 中文切分器 ----------
    nodes = show_nodes(build_parser(True, args.window_size), "[2] 换成中文切分器之后")

    # ---------- [3] 窗口结构细节 ----------
    i = 4
    print("=" * 78)
    print("[3] 以 node[%d] 为例, 看窗口是怎么构造的" % i)
    print("=" * 78)
    print("  原句            : %s" % nodes[i].text)
    print("  window_size     : %d" % args.window_size)
    print("  窗口覆盖的 node : [%d, %d]" % (
        max(0, i - args.window_size),
        min(i + args.window_size + 1, len(nodes)) - 1,
    ))
    print("  窗口文本        : %s" % nodes[i].metadata["window"])
    print()
    print("  注意: window 和 original_text 都被放进了 excluded 列表,")
    print("        所以嵌入时只用 text(单句), 窗口不会稀释向量。")
    print()

    # ---------- [4] 建索引 + 检索 ----------
    if args.mock:
        embed = MockEmbedding(embed_dim=64)
        print("=" * 78)
        print("[4] 检索  (使用 MockEmbedding, 向量是随机的, 结果无意义)")
        print("=" * 78)
        print("  跳过。要看真实检索效果请去掉 --mock。")
        return

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[!] 未设置 DASHSCOPE_API_KEY, 无法做真实检索。")
        print('    设置方式: $env:DASHSCOPE_API_KEY="sk-..."')
        print("    或加 --mock 只看结构。")
        return

    Settings.embed_model = QwenLIEmbedding(api_key=api_key)
    Settings.llm = None

    print("=" * 78)
    print("[4] 建索引 + 检索")
    print("=" * 78)
    index = VectorStoreIndex(nodes)
    print("  索引节点数: %d" % len(nodes))

    queries = [
        "Keepalived 用什么协议同步虚拟 IP",
        "HAProxy 支持哪些调度算法",
        "怎么避免虚拟 IP 冲突",
    ]
    retriever = index.as_retriever(similarity_top_k=2)

    for q in queries:
        print()
        print("  查询: %s" % q)
        hits = retriever.retrieve(q)
        for r in hits:
            print("    %.4f  %s" % (r.score or 0.0, r.node.text[:56]))

    # ---------- [5] MetadataReplacement 前后对比 ----------
    print()
    print("=" * 78)
    print("[5] MetadataReplacementPostProcessor 替换前后")
    print("=" * 78)
    q = "监控时要关注哪些指标"
    hits = retriever.retrieve(q)
    print("  查询: %s" % q)
    print()
    print("  --- 替换前(送给 LLM 的是单句) ---")
    for r in hits:
        print("    %s" % r.node.text)
    print()
    post = MetadataReplacementPostProcessor(target_metadata_key="window")
    hits2 = post.postprocess_nodes(hits)
    print("  --- 替换后(送给 LLM 的是完整窗口) ---")
    for r in hits2:
        print("    %s" % r.node.get_content())
    print()
    print("  替换是就地进行的: set_content() 会改写 node 本身。")


if __name__ == "__main__":
    main()
