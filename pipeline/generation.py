r"""基于检索结果生成答案。

流程:
    RetrievalHit -> context blocks -> Qwen chat -> answer + citations

默认使用阿里云百炼的 OpenAI 兼容接口。模型和端点可以通过环境变量覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import requests

from hybrid_retriever import RetrievalHit


DEFAULT_CHAT_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)


@dataclass
class Citation:
    index: int
    point_id: int | str
    text: str
    file_name: str | None
    page: int | None
    section: str | None
    chunk_index: int | None
    chunk_id: str | None
    artifact_id: str | None


@dataclass
class GenerationResult:
    answer: str
    citations: List[Citation]
    context_chars: int
    model: str


def build_context(
    hits: Sequence[RetrievalHit],
    max_context_chars: int = 6000,
) -> tuple[str, List[Citation]]:
    """把 hits 组装成带引用编号的上下文。"""
    blocks: List[str] = []
    citations: List[Citation] = []
    used = 0

    for citation_index, hit in enumerate(hits, 1):
        payload: Dict[str, Any] = hit.payload or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            continue

        remaining = max_context_chars - used
        if remaining <= 0:
            break

        block = text[:remaining]
        source = payload.get("file_name") or "unknown"
        section = payload.get("section") or "unknown"
        page = payload.get("page")
        header = (
            f"[C{citation_index}] 来源: {source} | "
            f"章节: {section} | 页码: {page if page is not None else '?'}"
        )
        blocks.append(f"{header}\n{block}")
        citations.append(Citation(
            index=citation_index,
            point_id=hit.point_id,
            text=block,
            file_name=payload.get("file_name"),
            page=payload.get("page"),
            section=payload.get("section"),
            chunk_index=payload.get("chunk_index"),
            chunk_id=payload.get("chunk_id"),
            artifact_id=payload.get("artifact_id"),
        ))
        used += len(block)

    return "\n\n".join(blocks), citations


class QwenChatGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-turbo",
        api_url: str = DEFAULT_CHAT_URL,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "QwenChatGenerator":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY, 无法初始化生成模型")
        return cls(
            api_key=api_key,
            model=os.getenv("QWEN_CHAT_MODEL", "qwen-turbo"),
            api_url=os.getenv("QWEN_CHAT_API_URL", DEFAULT_CHAT_URL),
            timeout=float(os.getenv("QWEN_CHAT_TIMEOUT", "120")),
        )

    def generate(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        max_context_chars: int = 6000,
        max_tokens: int = 800,
        model: str | None = None,
    ) -> GenerationResult:
        selected_model = model or self.model
        context, citations = build_context(hits, max_context_chars=max_context_chars)
        if not context:
            return GenerationResult(
                answer="根据当前检索到的上下文，无法确定。",
                citations=[],
                context_chars=0,
                model=selected_model,
            )

        system_prompt = (
            "你是一个严谨的文档问答助手。只能根据用户提供的上下文回答。"
            "如果上下文不足以回答，明确说“根据当前检索到的上下文，无法确定”。"
            "不得使用上下文之外的事实。"
            "回答中的关键结论必须标注引用来源，例如 [C1]、[C2]。"
            "先给直接答案，再给必要的要点。"
        )
        user_prompt = (
            f"问题:\n{query}\n\n"
            f"上下文:\n{context}\n\n"
            "请基于上述上下文回答，并使用 [C1]、[C2] 标注引用。"
        )

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "生成接口返回 %s: %s"
                % (response.status_code, response.text[:500])
            )

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return GenerationResult(
            answer=answer,
            citations=citations,
            context_chars=len(context),
            model=selected_model,
        )
