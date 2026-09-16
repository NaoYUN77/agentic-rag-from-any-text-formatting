r"""Raw Parse Quality Gate 和 Index Decision。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict

from .models import DocumentArtifact, IndexDecision, QualityReport


GARBLED_RE = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]")


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

    score = 1.0
    score -= min(char_count < 100, 1) * 0.45
    score -= min(char_count < 300, 1) * 0.15
    score -= duplicate_ratio * 0.30
    score -= min(garbled_ratio * 5, 0.40)
    score -= 0.05 if not artifact.title else 0.0
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
