r"""Raw Parse Quality Gate 和 Index Decision。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict

from .models import DocumentArtifact, IndexDecision, QualityReport


GARBLED_RE = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]")


def _structural_fidelity(artifact: DocumentArtifact, blocks):
    """结构保真度检查 —— 返回 (metrics, flags, reasons)。

    **为什么需要**：质量门此前只看 token 数、长度分布、重复率、乱码率 ——
    对「结构丢了」完全无感。issues 11/12 的两次回归当时都没被发现：

    - heading 层级被压平（只剩 lv1/lv2，`2.3.1.` 被判成 lv2）
    - `section_path` 大面积为空（同级标题互相嵌套导致顶层前缀塌缩）
    - PDF 页码完全丢失（经 Markdown 中转的解析器**原理上**保不住页码）

    一本讲配置的手册丢了全部配置示例、质量分仍是 **0.99**。这一段就是让
    那类问题**可见**，而不是被高分掩盖。

    判据都取「有前提才检查」，避免对本来就没结构的文档误报：
    没有 heading 的文档不该因为「层级单一」被扣分。
    """
    metrics: Dict[str, Any] = {}
    flags = []
    reasons = []

    headings = [b for b in blocks if b.type == "heading"]
    non_heading = [b for b in blocks if b.type != "heading"]

    # 1) PDF 页码覆盖 —— 页码是引用（"第 N 页"）的刚需，丢了无法追溯
    mime = (artifact.mime_type or "").lower()
    if "pdf" in mime and blocks:
        with_page = sum(1 for b in blocks if b.page)
        coverage = with_page / len(blocks)
        metrics["page_coverage"] = round(coverage, 4)
        if with_page == 0:
            flags.append("pdf_without_page_numbers")
            reasons.append("PDF 解析结果没有任何页码")

    # 2) heading 层级多样性 —— 只有 1 个层级说明层级被压平了
    if len(headings) >= 5:
        levels = {b.metadata.get("level") for b in headings}
        levels.discard(None)
        metrics["distinct_heading_levels"] = len(levels)
        if len(levels) <= 1:
            flags.append("flat_heading_levels")
            reasons.append("有 %d 个标题但只有 %d 个层级，层级疑似被压平"
                           % (len(headings), len(levels)))

    # 3) section_path 覆盖 —— 有标题却没人挂在标题下，说明层级链断了
    if len(headings) >= 3 and non_heading:
        with_path = sum(1 for b in non_heading if b.section_path)
        coverage = with_path / len(non_heading)
        metrics["section_path_coverage"] = round(coverage, 4)
        if with_path == 0:
            flags.append("section_path_all_empty")
            reasons.append("有标题但所有正文块的 section_path 都为空")

    return metrics, flags, reasons


def assess_raw_quality(artifact: DocumentArtifact) -> QualityReport:
    blocks = artifact.blocks
    text_blocks = [b for b in blocks if b.type in {"text", "list", "heading"}]
    code_blocks = [b for b in blocks if b.type == "code"]
    formula_blocks = [b for b in blocks if b.type == "formula"]
    image_blocks = [b for b in blocks if b.type == "image"]
    all_text = "\n".join(b.text for b in blocks if b.text)
    char_count = len(all_text)
    garbled = len(GARBLED_RE.findall(all_text))
    garbled_ratio = garbled / max(len(all_text), 1)

    nonempty = [b.text.strip() for b in blocks if b.text.strip()]
    duplicate_ratio = 0.0
    if nonempty:
        duplicate_ratio = 1.0 - len(set(nonempty)) / len(nonempty)

    flags = []
    reasons = []
    if not blocks:
        flags.append("no_blocks")
        reasons.append("解析结果没有 block")
    if char_count < 100:
        flags.append("very_short_text")
        reasons.append("正文过短")
    if artifact.source_type in {"url", "local_file"} and char_count < 300:
        flags.append("low_content")
    if duplicate_ratio > 0.35:
        flags.append("high_duplicate_ratio")
        reasons.append("重复 block 比例过高")
    if garbled_ratio > 0.01:
        flags.append("garbled_text")
        reasons.append("乱码字符比例过高")
    if not artifact.title:
        flags.append("missing_title")

    # 结构保真度：此前完全没有这一维度（见 _structural_fidelity 的说明）
    struct_metrics, struct_flags, struct_reasons = _structural_fidelity(
        artifact, blocks)
    flags.extend(struct_flags)
    reasons.extend(struct_reasons)

    score = 1.0
    score -= min(char_count < 100, 1) * 0.45
    score -= min(char_count < 300, 1) * 0.15
    score -= duplicate_ratio * 0.30
    score -= min(garbled_ratio * 5, 0.40)
    score -= 0.05 if not artifact.title else 0.0
    # 结构问题只做**小扣分**：目的是让它可见、把状态压到 medium 提示人工看一眼，
    # 而不是把文档一棒打死 —— 结构坏了内容往往还在，仍然值得进索引。
    score -= 0.06 * len(struct_flags)
    score = max(0.0, min(1.0, score))

    if "no_blocks" in flags or char_count == 0:
        status = "reject"
    elif score >= 0.85:
        status = "high"
    elif score >= 0.60:
        status = "medium"
    elif score >= 0.35:
        status = "low"
    else:
        status = "reject"

    metrics: Dict[str, Any] = {
        "char_count": char_count,
        "block_count": len(blocks),
        "text_block_count": len(text_blocks),
        "code_block_count": len(code_blocks),
        "formula_block_count": len(formula_blocks),
        "image_block_count": len(image_blocks),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "garbled_ratio": round(garbled_ratio, 6),
        "language": artifact.language,
    }
    metrics.update(struct_metrics)
    return QualityReport(
        score=round(score, 4),
        status=status,
        flags=flags,
        metrics=metrics,
        reasons=reasons,
    )


def decide_index(artifact: DocumentArtifact) -> IndexDecision:
    quality = artifact.quality
    if quality is None:
        raise ValueError("artifact.quality must be assessed first")
    if quality.status == "high":
        return IndexDecision(
            tier="high", dense_index=True, sparse_index=True,
            sparse_weight=1.0, reason="high quality",
        )
    if quality.status == "medium":
        return IndexDecision(
            tier="medium", dense_index=True, sparse_index=True,
            sparse_weight=0.6, reason="medium quality",
        )
    if quality.status == "low":
        return IndexDecision(
            tier="low", dense_index=True, sparse_index=False,
            sparse_weight=0.0, rerank_penalty=0.2,
            review_required=True, reason="low quality",
        )
    return IndexDecision(
        tier="reject", dense_index=False, sparse_index=False,
        sparse_weight=0.0, rerank_penalty=1.0,
        review_required=True, reason="rejected quality",
    )
