r"""RAG 检索服务。

    语料清洗 -> 分块 -> Qdrant dense + sparse -> RRF -> 本服务暴露检索接口

启动:
    cd pipeline
    $env:DASHSCOPE_API_KEY = "sk-..."
    .\.venv_rag\Scripts\python.exe -m uvicorn rag_server:app --host 127.0.0.1 --port 8000

接口:
    GET  /              检索页面
    GET  /api/info      集合信息
    POST /api/search    检索(mode=hybrid|dense|sparse)
    POST /api/answer    检索 + LLM 生成答案与引用
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from embeddings import build_embeddings
from hybrid_retriever import HybridRetriever
from generation import QwenChatGenerator
from reranker import DashScopeReranker, VoyageReranker, build_reranker
from abstention import ABSTAIN_ANSWER, load_threshold, should_abstain

QDRANT_PATH = os.getenv("RAG_QDRANT_PATH", "qdrant_data")
# 默认指向 Phase 0 的当前索引（固定大小切块、无 overlap、
# parent 按完整 section_path 分组且受 parent_tokens 上限约束）。
# 此前默认是 "redhat" / "redhat_sparse" —— 那来自已废弃的 LlamaIndex
# fixed-semantic 链路（rebuild_llamaindex_corpus.py，2026-09-21 删除）。
COLLECTION = os.getenv("RAG_DENSE_COLLECTION", "phase0_capped_dense")
SPARSE_COLLECTION = os.getenv("RAG_SPARSE_COLLECTION", "phase0_capped_sparse")
SPARSE_ARTIFACT_DIR = Path(os.getenv(
    "RAG_SPARSE_ARTIFACT_DIR",
    str(Path(__file__).parent / "index_artifacts" / "phase0_capped"),
))
STATIC_DIR = Path(__file__).parent / "static"

# ---- 拒答 ----
# 判定逻辑抽在 abstention.py（纯函数，不依赖 fastapi）——
# 放在这里会让任何想用它的地方（脚本 / 评估 / 测试）都被迫拖上
# 一整套 Web 依赖。CI 里撞过：测试只 import 了一下这个模块就整体失败。
# 阈值是「语料 + 嵌入模型」绑定的，换任一个都要用 calibrate_abstain.py 重算。
ABSTAIN_THRESHOLD = load_threshold()

_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建一次客户端和嵌入函数, 全程复用。"""
    print("[启动] 连接 Qdrant:", QDRANT_PATH)
    _state["client"] = QdrantClient(path=QDRANT_PATH)

    if not _state["client"].collection_exists(COLLECTION):
        raise RuntimeError(
            "集合 %s 不存在。请先跑 rebuild_phase0_index.py 建库，"
            "或用 RAG_DENSE_COLLECTION 指定其他集合" % COLLECTION
        )
    info = _state["client"].get_collection(COLLECTION)
    print("[启动] 集合 %s, %d 个点" % (COLLECTION, info.points_count))

    if not _state["client"].collection_exists(SPARSE_COLLECTION):
        raise RuntimeError(
            "集合 %s 不存在。请先跑 rebuild_phase0_index.py 建库（dense/sparse 一次写入），"
            "或用 RAG_SPARSE_COLLECTION 指定其他集合" % SPARSE_COLLECTION
        )
    sparse_info = _state["client"].get_collection(SPARSE_COLLECTION)
    print("[启动] 集合 %s, %d 个点" % (SPARSE_COLLECTION, sparse_info.points_count))

    print("[启动] 初始化嵌入模型 ...")
    _state["embeddings"] = build_embeddings("qwen")
    _state["retriever"] = HybridRetriever(
        client=_state["client"],
        embeddings=_state["embeddings"],
        dense_collection=COLLECTION,
        sparse_collection=SPARSE_COLLECTION,
        artifact_dir=SPARSE_ARTIFACT_DIR,
    )
    print("[启动] 初始化生成模型 ...")
    _state["generator"] = QwenChatGenerator.from_env()
    print("[启动] 生成模型:", _state["generator"].model)
    print("[启动] 初始化 Reranker ...")
    # RAG_RERANKER: voyage (rerank-2.5-lite, 默认) / dashscope / none
    _state["reranker"] = build_reranker(os.getenv("RAG_RERANKER", "voyage"))
    if _state["reranker"] is None:
        print("[启动] Reranker: 已关闭（RAG_RERANKER=none）")
    else:
        print("[启动] Reranker:", _state["reranker"].model)
    print("[启动] 就绪")

    yield

    _state["client"].close()
    print("[关闭] 已断开")


app = FastAPI(title="RAG 检索", lifespan=lifespan)


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int = Field(5, ge=1, le=50, description="返回条数")
    section: Optional[str] = Field(None, description="按 section 过滤(子串匹配)")
    mode: Literal["hybrid", "dense", "sparse"] = Field(
        "hybrid", description="检索模式"
    )
    candidate_k: int = Field(20, ge=1, le=200, description="每路候选数")
    rrf_k: int = Field(60, ge=1, le=1000, description="RRF 平滑常数")
    rerank: bool = Field(True, description="是否启用 rerank")
    rerank_candidate_k: int = Field(
        20, ge=1, le=200, description="送入 reranker 的候选数"
    )


class Hit(BaseModel):
    point_id: int | str
    chunk_id: Optional[str] = None
    artifact_id: Optional[str] = None
    rank: int
    score: float
    rrf_score: Optional[float] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    pre_rerank_rank: Optional[int] = None
    matched_terms: List[str] = Field(default_factory=list)
    text: str
    file_name: Optional[str] = None
    page: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    chunk_index: Optional[int] = None
    char_count: Optional[int] = None
    n_sentences: Optional[int] = None
    # 单句检索单元: unit_kind="sentence" 时, window_text 是 metadata 里的
    # 前后文窗口(不进 embedding), bbox 用于原文高亮
    unit_kind: Optional[str] = None
    window_text: Optional[str] = None
    bbox: Optional[List[float]] = None
    # 块级来源 URL（HTML 路径填入），用于"这段来自哪个页面"的引用
    url: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    elapsed_ms: int
    margin: Optional[float] = Field(None, description="top1 与 top2 的分差")
    hits: List[Hit]
    filtered: bool
    candidate_k: int
    rrf_k: int
    rerank: bool
    rerank_model: Optional[str] = None
    rerank_applied: bool = False
    rerank_ms: int = 0
    rerank_error: Optional[str] = None
    sparse_known_terms: List[str] = Field(default_factory=list)
    sparse_unknown_terms: List[str] = Field(default_factory=list)
    # ---- 拒答 ----
    # dense_top_score 是 **dense 排序的 top-1 相似度**（不是最终排序 top-1 的分数）
    # —— 拒答阈值就是在这个量上标定的。
    abstained: bool = Field(
        False, description="是否因相关度过低而拒答（阈值 0 时恒为 false）"
    )
    abstain_threshold: float = Field(0.0, description="当前生效的拒答阈值，0=关闭")
    dense_top_score: Optional[float] = Field(
        None, description="dense 排序 top-1 的余弦相似度；mode=sparse 时为 null"
    )


class AnswerRequest(SearchRequest):
    context_k: int = Field(5, ge=1, le=20, description="送入生成模型的上下文数")
    max_context_chars: int = Field(
        6000, ge=1000, le=50000, description="上下文最大字符数"
    )
    chat_model: Optional[str] = Field(None, description="覆盖默认 chat 模型")
    max_tokens: int = Field(800, ge=100, le=4000, description="最大生成 token 数")


class CitationOut(BaseModel):
    index: int
    point_id: int | str
    chunk_id: Optional[str] = None
    artifact_id: Optional[str] = None
    file_name: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    snippet: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    mode: str
    model: str
    citations: List[CitationOut]
    hits: List[Hit]
    retrieval_ms: int
    generation_ms: int
    elapsed_ms: int
    context_chars: int
    filtered: bool
    rerank_applied: bool = False
    rerank_ms: int = 0
    # ---- 拒答 ----
    abstained: bool = Field(
        False, description="是否拒答；为 true 时 answer 是固定话术，未调用生成模型"
    )
    abstain_threshold: float = Field(0.0, description="当前生效的拒答阈值，0=关闭")
    dense_top_score: Optional[float] = Field(
        None, description="dense 排序 top-1 的余弦相似度"
    )


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------
def _to_hit(p, rank: int, mode: str) -> Hit:
    pl = p.payload or {}
    if p.rerank_score is not None:
        display_score = p.rerank_score
    elif mode == "hybrid":
        display_score = p.rrf_score or 0.0
    elif mode == "dense":
        display_score = p.dense_score or 0.0
    else:
        display_score = p.sparse_score or 0.0

    return Hit(
        point_id=p.point_id,
        chunk_id=pl.get("chunk_id"),
        artifact_id=pl.get("artifact_id"),
        rank=rank,
        score=round(float(display_score), 6),
        rrf_score=round(p.rrf_score, 6) if p.rrf_score is not None else None,
        dense_score=round(p.dense_score, 6) if p.dense_score is not None else None,
        sparse_score=round(p.sparse_score, 6) if p.sparse_score is not None else None,
        dense_rank=p.dense_rank,
        sparse_rank=p.sparse_rank,
        rerank_score=round(p.rerank_score, 6) if p.rerank_score is not None else None,
        pre_rerank_rank=p.pre_rerank_rank,
        matched_terms=p.matched_terms or [],
        text=pl.get("text", ""),
        file_name=pl.get("file_name"),
        page=pl.get("page"),
        page_start=pl.get("page_start"),
        page_end=pl.get("page_end"),
        section=pl.get("section"),
        section_path=pl.get("section_path") or [],
        chunk_index=pl.get("chunk_index"),
        # 单句策略: 检索命中单句后取这些字段拼上下文与做引用定位
        unit_kind=pl.get("unit_kind"),
        window_text=pl.get("window_text"),
        bbox=pl.get("bbox"),
        url=pl.get("url"),
        char_count=pl.get("char_count"),
        n_sentences=pl.get("n_sentences"),
    )


def _retrieve_with_rerank(req: SearchRequest, limit: int) -> tuple:
    retriever: HybridRetriever = _state["retriever"]
    # 具体实现由 RAG_RERANKER 决定（voyage / dashscope / none）。
    # 注意: 原注解写的是 DashScopeReranker, 但那个名字从未导入过 —— 见 F821。
    reranker: VoyageReranker | DashScopeReranker | None = _state["reranker"]
    candidate_limit = req.rerank_candidate_k if req.rerank else limit

    t0 = time.perf_counter()
    result = retriever.search(
        query=req.query,
        mode=req.mode,
        limit=candidate_limit,
        candidate_k=req.candidate_k,
        rrf_k=req.rrf_k,
        section=req.section,
    )
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    hits = result["hits"]
    rerank_applied = False
    rerank_ms = 0
    rerank_error = None
    if req.rerank and hits:
        t1 = time.perf_counter()
        try:
            hits = reranker.rerank(req.query, hits, top_k=limit)
            rerank_applied = True
        except Exception as exc:
            rerank_error = str(exc)
            # ⚠️ 失败时 hits 还是**候选列表**（`rerank_candidate_k` 条，默认 20），
            # 必须截回 limit —— 否则调用方要 3 条却拿到 20 条。
            # 这个坑恰好在 API 降级（限速 / 超时）时触发，最不该出意外的时候。
            hits = hits[:limit]
        rerank_ms = int((time.perf_counter() - t1) * 1000)

    return result, hits, retrieval_ms, rerank_applied, rerank_ms, rerank_error


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    result, result_hits, retrieval_ms, rerank_applied, rerank_ms, rerank_error = (
        _retrieve_with_rerank(req, limit=req.top_k)
    )
    elapsed = retrieval_ms + rerank_ms

    hits = [
        _to_hit(p, rank=i, mode=req.mode)
        for i, p in enumerate(result_hits, 1)
    ]

    margin = None
    if len(hits) >= 2:
        margin = round(hits[0].score - hits[1].score, 4)

    dense_top = result.get("dense_top_score")
    return SearchResponse(
        query=req.query,
        mode=req.mode,
        elapsed_ms=elapsed,
        margin=margin,
        hits=hits,
        filtered=bool(req.section),
        candidate_k=req.candidate_k,
        rrf_k=req.rrf_k,
        rerank=req.rerank,
        rerank_model=_state["reranker"].model if req.rerank else None,
        rerank_applied=rerank_applied,
        rerank_ms=rerank_ms,
        rerank_error=rerank_error,
        sparse_known_terms=result.get("known_terms", []),
        sparse_unknown_terms=result.get("unknown_terms", []),
        abstained=should_abstain(dense_top, ABSTAIN_THRESHOLD),
        abstain_threshold=ABSTAIN_THRESHOLD,
        dense_top_score=dense_top,
    )


@app.post("/api/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    generator: QwenChatGenerator = _state["generator"]

    t0 = time.perf_counter()
    result, retrieval_hits, retrieval_ms, rerank_applied, rerank_ms, rerank_error = (
        _retrieve_with_rerank(req, limit=req.context_k)
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    hits = [
        _to_hit(hit, rank=i, mode=req.mode)
        for i, hit in enumerate(retrieval_hits, 1)
    ]

    dense_top = result.get("dense_top_score")
    abstained = should_abstain(dense_top, ABSTAIN_THRESHOLD)

    # 拒答时**不调用生成模型** —— 这正是拒答的意义：省一次 LLM 调用，
    # 也避免它对着不相关的上下文硬编一个答案出来。
    if abstained:
        return AnswerResponse(
            query=req.query,
            answer=ABSTAIN_ANSWER,
            mode=req.mode,
            model="(abstained)",
            citations=[],
            hits=hits,
            retrieval_ms=retrieval_ms,
            generation_ms=0,
            elapsed_ms=elapsed_ms,
            context_chars=0,
            filtered=bool(req.section),
            rerank_applied=rerank_applied,
            rerank_ms=rerank_ms,
            abstained=True,
            abstain_threshold=ABSTAIN_THRESHOLD,
            dense_top_score=dense_top,
        )

    generation = generator.generate(
        query=req.query,
        hits=retrieval_hits,
        max_context_chars=req.max_context_chars,
        max_tokens=req.max_tokens,
        model=req.chat_model,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    citations = [
        CitationOut(
            index=c.index,
            point_id=c.point_id,
            chunk_id=c.chunk_id,
            artifact_id=c.artifact_id,
            file_name=c.file_name,
            page=c.page,
            section=c.section,
            chunk_index=c.chunk_index,
            snippet=c.text[:300],
        )
        for c in generation.citations
    ]

    return AnswerResponse(
        query=req.query,
        answer=generation.answer,
        mode=req.mode,
        model=generation.model,
        citations=citations,
        hits=hits,
        retrieval_ms=retrieval_ms,
        generation_ms=max(0, elapsed_ms - retrieval_ms),
        elapsed_ms=elapsed_ms,
        context_chars=generation.context_chars,
        filtered=bool(req.section),
        rerank_applied=rerank_applied,
        rerank_ms=rerank_ms,
        abstained=False,
        abstain_threshold=ABSTAIN_THRESHOLD,
        dense_top_score=dense_top,
    )


@app.get("/api/info")
def info() -> Dict[str, Any]:
    client: QdrantClient = _state["client"]
    c = client.get_collection(COLLECTION)
    s = client.get_collection(SPARSE_COLLECTION)
    return {
        "collection": COLLECTION,
        "points_count": c.points_count,
        "sparse_collection": SPARSE_COLLECTION,
        "sparse_points_count": s.points_count,
        "vector_size": c.config.params.vectors.size,
        "distance": str(c.config.params.vectors.distance),
        "qdrant_path": QDRANT_PATH,
        "model": getattr(_state.get("embeddings"), "model", None) or "unknown",
        "chat_model": _state.get("generator").model if _state.get("generator") else None,
        "rerank_model": _state.get("reranker").model if _state.get("reranker") else None,
        "mode": "hybrid + RRF",
        "rrf_k": 60,
        # 拒答阈值：0 = 关闭。见文件顶部的说明与标定方法。
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "abstain_enabled": ABSTAIN_THRESHOLD > 0,
    }


@app.get("/api/sections")
def sections() -> Dict[str, Any]:
    """列出所有 section(供前端做过滤下拉)。"""
    client: QdrantClient = _state["client"]
    seen: List[str] = []
    offset = None
    while True:
        pts, offset = client.scroll(
            COLLECTION, limit=256, offset=offset,
            with_payload=["section"], with_vectors=False,
        )
        for p in pts:
            s = (p.payload or {}).get("section")
            if s and s not in seen:
                seen.append(s)
        if offset is None:
            break
    return {"sections": sorted(seen)}


@app.get("/")
def index() -> FileResponse:
    f = STATIC_DIR / "index.html"
    if not f.exists():
        raise HTTPException(500, "static/index.html 不存在")
    return FileResponse(f)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """空 favicon —— 否则浏览器每次访问都会 404，
    调试界面时 console 里会一直挂一条红色的 Failed to load resource。"""
    return Response(status_code=204)

