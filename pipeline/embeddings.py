r"""嵌入模型封装。

这里是**生产基础设施**, 不是演示脚本 —— `rag_server` / `rebuild_phase0_index` /
`eval.run_eval` / `ingest.qdrant_indexer` 都从这里取 `build_embeddings()`。

历史: 这些类原先住在 `semantic_chunker_demo.py` 里（那是个演示脚本）。
结果生产代码反过来依赖一个 "demo" 文件, 命名与实际角色不符。2026-09-21
把可复用部分抽到这里, 演示脚本随之删除。

支持的后端（`build_embeddings(backend)`）：

    local    HashingEmbeddings   —— 字符 n-gram 哈希, 零依赖、确定性, **无语义**。
                                    只用于离线验证链路, 不能用来评估检索效果。
    qwen     QwenTextEmbeddings  —— 阿里云百炼通用文本向量（**生产用这个**）
    openai   OpenAIEmbeddings    —— OpenAI 官方或任何兼容端点

⚠️ **百炼有两个不同的 embedding 接口, 不能只换模型名**：

    通用文本向量  /services/embeddings/text-embedding/text-embedding
                  input={"texts": [...]}  响应 output.embeddings[].text_index
                  -> QwenTextEmbeddings（本项目生产使用）

    多模态向量    /services/embeddings/multimodal-embedding/multimodal-embedding
                  input={"contents": [...]}  响应 output.embeddings[].index
                  -> QwenVLEmbeddings（图文混合检索；当前生产路径未使用）

环境变量覆盖：`QWEN_EMBED_MODEL` / `QWEN_EMBED_DIM` / `QWEN_EMBED_INSTRUCT`。
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import List, Sequence

import numpy as np
import requests
from langchain_core.embeddings import Embeddings


# 中文不能用 LangChain 默认的 (?<=[.?!])\s+ —— 它只认英文标点。
# 也不能反过来要求"句号后必须跟空白": PDF/Markdown 抽出来的文本里,
# 句号后经常直接跟汉字（例如 "路由器上运行。所有运行"）, 那样会漏切近一半句子。
#
# 这个正则: 在句末标点之后切分, 顺带吃掉紧随的空白, 且要求后面还有非空白内容
# （最后一条是为了避免文本正好以句号结尾时, 切出一个空片段）。
CN_SENTENCE_SPLIT_REGEX = r"(?<=[。！？；])\s*(?=\S)"


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

    ⚠️ 这**不是**真正的语义模型, 只捕捉字面重叠: 共享词/字越多的句子越相似。
    它的用途是让离线链路（`--backend local`）在没有网络和 API key 时也能跑通,
    **不能用它的检索指标评估效果**。

    真实语义请用 `build_embeddings("qwen")`。
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

# 通用文本向量原生接口（与多模态接口是两个不同的 endpoint）。
QWEN_TEXT_EMBED_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "text-embedding/text-embedding"
)


class QwenTextEmbeddings(Embeddings):
    """阿里云百炼 qwen3.7-text-embedding 的文本嵌入封装（**生产使用**）。

    走百炼「通用文本向量」原生接口：

        POST /api/v1/services/embeddings/text-embedding/text-embedding

    与 `QwenVLEmbeddings`（多模态接口）的区别（**接口不同, 不能只换模型名**）：

    - 请求体 `input` 是 `{"texts": [...]}`，不是 `{"contents": [{"text": ...}]}`
    - 响应字段是 `output.embeddings[].text_index`，不是 `.index`
    - 支持 `text_type` = `query` / `document`（非对称检索：入库用 document，
      检索用 query，官方建议这样区分能提升效果）
    - 单次最多 20 条（qwen3.7-text-embedding），单条最长 128K token

    参数：
        dimension: qwen3.7-text-embedding 支持 2560/2048/1536/1024(默认)/768/512/256
        instruct:  可选任务说明（仅 text_type=query 时生效），建议英文
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.7-text-embedding",
        dimension: int = 1024,
        instruct: str | None = None,
        api_url: str = QWEN_TEXT_EMBED_ENDPOINT,
        batch_size: int = 20,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.instruct = instruct
        self.api_url = api_url
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        # 缓存按 (text_type, text) 分键 —— 同一段文本作为 query / document
        # 可能得到不同向量，不能混用。
        self._cache: dict = {}

    def _call(self, texts: Sequence[str], text_type: str) -> List[List[float]]:
        parameters: dict = {"dimension": self.dimension, "output_type": "dense"}
        parameters["text_type"] = text_type
        if self.instruct and text_type == "query":
            parameters["instruct"] = self.instruct
        payload = {
            "model": self.model,
            "input": {"texts": list(texts)},
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
                "百炼文本向量接口返回 %s: %s" % (resp.status_code, resp.text[:500])
            )
        embeddings = resp.json()["output"]["embeddings"]
        embeddings.sort(key=lambda e: e["text_index"])
        return [e["embedding"] for e in embeddings]

    def _embed_texts(self, texts: Sequence[str], text_type: str) -> List[List[float]]:
        keys = [(text_type, t) for t in dict.fromkeys(texts)]
        missing = [t for (tt, t) in keys if (tt, t) not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._call(batch, text_type)
            if len(vectors) != len(batch):
                # 数量对不上时退化为逐条，保证一定能跑通。
                vectors = [self._call([t], text_type)[0] for t in batch]
            for t, v in zip(batch, vectors):
                self._cache[(text_type, t)] = v
        return [self._cache[(text_type, t)] for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_texts(texts, "document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text], "query")[0]


class QwenVLEmbeddings(Embeddings):
    """阿里云百炼 qwen3-vl-embedding 的多模态嵌入封装。

    走百炼「多模态向量」原生接口, 不是 OpenAI 兼容接口:

        POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding

    ⚠️ **当前生产路径没有调用方**。保留它是因为：本项目把"图片接入多模态
    embedding"列为待办（见 `block_aware_chunking_implementation.md` 的限制清单），
    而这个类记录了多模态接口与文本接口的**结构差异**（`contents` vs `texts`、
    `.index` vs `.text_index`）—— 那是最容易踩错、也最容易忘的地方。
    如果最终决定不做多模态, 应当把它一并删除。

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
        # 缓存能把付费接口的调用次数大幅压低（同一段文本反复嵌入的场景）。
        # 生产环境请把缓存换成持久化的向量库。
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
    """按 backend 构造嵌入函数。

    local  -> HashingEmbeddings（离线, 无语义, 仅验证链路）
    qwen   -> QwenTextEmbeddings（生产）
    openai -> OpenAIEmbeddings（或任何 OpenAI 协议端点）
    """
    if backend == "local":
        return HashingEmbeddings()

    if backend == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit(
                "缺少 DASHSCOPE_API_KEY 环境变量, 请先设置:\n"
                '  $env:DASHSCOPE_API_KEY="sk-你的百炼Key"'
            )
        return QwenTextEmbeddings(
            api_key=api_key,
            model=os.getenv("QWEN_EMBED_MODEL", "qwen3.7-text-embedding"),
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
