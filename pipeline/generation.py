r"""基于检索结果生成答案。

流程:
    RetrievalHit -> context blocks -> Qwen chat -> answer + citations

默认使用阿里云百炼的 OpenAI 兼容接口。模型和端点可以通过环境变量覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    # ---- 引用展示所需的文档信息 ----
    # 这些字段来自 chunk 的 metadata / payload, 不参与 embedding,
    # 只在生成阶段用于把答案里的 [Cn] 还原成"哪篇文档的哪一页 / 哪个页面"。
    title: str | None = None
    source_uri: str | None = None
    url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: List[str] = field(default_factory=list)
    bbox: List[float] | None = None
    unit_kind: str | None = None
    # 本 chunk 覆盖的标题清单（heading 已从正文剥离后, 这里保留,
    # 供生成阶段给 LLM 补"这一段在讲哪几个小节"的背景）。
    headings: List[str] = field(default_factory=list)

    def locator(self) -> str:
        """给出一条人类可读的出处串。"""
        parts: List[str] = []
        if self.title:
            parts.append(str(self.title))
        elif self.file_name:
            parts.append(str(self.file_name))
        if self.section_path:
            parts.append(" > ".join(str(x) for x in self.section_path))
        elif self.section:
            parts.append(str(self.section))
        page = self._page_text()
        if page:
            parts.append(page)
        # 有块级 URL 时补上, 让人能直接跳回原页面
        if self.url:
            parts.append(str(self.url))
        return " | ".join(parts) if parts else "unknown"

    def _page_text(self) -> str:
        if self.page_start is not None and self.page_end is not None:
            if self.page_start == self.page_end:
                return f"第 {self.page_start} 页"
            return f"第 {self.page_start}-{self.page_end} 页"
        if self.page is not None:
            return f"第 {self.page} 页"
        return ""


@dataclass
class GenerationResult:
    answer: str
    citations: List[Citation]
    context_chars: int
    model: str


def _chunk_background(payload: Dict[str, Any]) -> str:
    """从 metadata 拼出一段给 LLM 的"背景", 补偿正文被剥离标题后的上下文缺失。

    正文现在不含标题(heading 只作 metadata), 所以一段正文可能以"这个参数..."
    开头, LLM 不知道它属于哪一章。这里用 section_path / headings / page / url
    把结构信息显式补回来, 让 LLM 能正确锚定"上文提到的 X"。
    """
    parts: List[str] = []
    title = payload.get("title") or payload.get("file_name")
    if title:
        parts.append(f"文档《{title}》")
    section_path = payload.get("section_path") or []
    if section_path:
        parts.append("章节路径: " + " > ".join(str(x) for x in section_path))
    headings = payload.get("headings") or []
    if headings:
        parts.append("本段涵盖小节: " + "、".join(str(h) for h in headings))
    url = payload.get("url")
    if url:
        parts.append(f"出处: {url}")
    return "；".join(parts)


def build_context(
    hits: Sequence[RetrievalHit],
    max_context_chars: int = 6000,
    context_source: str = "auto",
    include_background: bool = True,
) -> tuple[str, List[Citation]]:
    """把 hits 组装成带引用编号的上下文。

    context_source 决定喂给 LLM 的正文取哪一段:
        text   —— 只用 chunk 正文
        window —— 只用 metadata 窗口(单句策略下是前后 N 句)
        auto   —— 有窗口就用窗口, 否则用正文(默认)

    auto 的理由: 单句策略下正文只有一句话, 遇到"它降低了 49%"这类
    指代句, 单句无法支撑生成。窗口保证上下文完整, 而引用信息仍从
    metadata 取, 两者职责分离。

    include_background: 是否给每段正文前加一行"背景"(来自 metadata 的
    section_path / headings / url)。正文已剥离标题, 这一行把结构信息补回,
    让 LLM 知道这段属于哪一章、讲了哪几个小节。可关闭做 A/B。
    """
    blocks: List[str] = []
    citations: List[Citation] = []
    used = 0

    for citation_index, hit in enumerate(hits, 1):
        payload: Dict[str, Any] = hit.payload or {}
        text = str(payload.get("text", "")).strip()
        window = str(payload.get("window_text") or "").strip()

        if context_source == "window":
            body = window or text
        elif context_source == "text":
            body = text
        else:
            body = window or text

        if not body:
            continue

        remaining = max_context_chars - used
        if remaining <= 0:
            break

        block = body[:remaining]
        section_path = payload.get("section_path") or []
        section = payload.get("section")
        if not section and section_path:
            section = " > ".join(str(x) for x in section_path)
        source = payload.get("title") or payload.get("file_name") or "unknown"

        page_start = payload.get("page_start")
        page_end = payload.get("page_end")
        page = payload.get("page")
        if page_start is not None and page_end is not None:
            page_text = (f"第 {page_start} 页" if page_start == page_end
                         else f"第 {page_start}-{page_end} 页")
        elif page is not None:
            page_text = f"第 {page} 页"
        else:
            page_text = "页码未知"

        header = f"[C{citation_index}] 文档: {source} | 章节: {section or '未标注'} | {page_text}"
        if include_background:
            background = _chunk_background(payload)
            block_text = f"{header}\n背景: {background}\n\n{block}" if background else f"{header}\n{block}"
        else:
            block_text = f"{header}\n{block}"
        blocks.append(block_text)
        citations.append(Citation(
            index=citation_index,
            point_id=hit.point_id,
            text=block,
            file_name=payload.get("file_name"),
            page=page,
            section=section,
            chunk_index=payload.get("chunk_index"),
            chunk_id=payload.get("chunk_id"),
            artifact_id=payload.get("artifact_id"),
            title=payload.get("title"),
            source_uri=payload.get("source_uri"),
            url=payload.get("url"),
            page_start=page_start,
            page_end=page_end,
            section_path=list(section_path),
            bbox=payload.get("bbox"),
            unit_kind=payload.get("unit_kind"),
            headings=list(payload.get("headings") or []),
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
        context_source: str = "auto",
    ) -> GenerationResult:
        selected_model = model or self.model
        context, citations = build_context(
            hits,
            max_context_chars=max_context_chars,
            context_source=context_source,
        )
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
