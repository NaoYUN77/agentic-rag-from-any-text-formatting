r"""qrels 读写与 schema 校验。

qrels = query relevance judgments, 即"这条 query 的正确答案在原文的哪些位置"。

为什么 gold 锚在文本上而不是 ID 上
----------------------------------
项目里有四层 ID, 没有一层稳定:

    artifact_id  doc_5c3e74c436ef      uuid4 生成, 重摄取即变
    block_id     b0001                 文档内位置序号, 换 parser 即错位
    chunk_id     doc_XXX_c0001         含位置序号, 改分块粒度即失义
    point_id     1850931163052542526   chunk_id 的哈希, 随之而变

所以 gold 存**原文片段**。评估时用 text 反查当前索引, 这样重建索引、
换 parser、改分块策略之后标注依然有效 —— 这是能跨切块策略对比的前提。

block_id / span 是辅助定位, 只用于人工排查, 不参与自动映射。
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# category 判据: 用于分维度看指标
CATEGORIES = {
    "semantic",      # 语义相近但字面不同 (考 embedding)
    "exact",         # 含标识符/参数名/专有名词 (考 BM25)
    "mixed",         # 两者都有 (考 RRF 融合)
    "multi_hop",     # 需跨段/跨文档 (考 parent expansion)
    "unanswerable",  # 语料里没有答案 (考系统会不会硬答)
}

SPLITS = {"dev", "test", "smoke"}


class QrelsError(ValueError):
    """qrels 格式或内容不合法。"""


@dataclass
class GoldSpan:
    """一个 gold 片段。text 是主锚点, 其余是辅助定位。"""
    text: str
    source_uri: Optional[str] = None
    block_id: Optional[str] = None
    span: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class QrelsItem:
    query_id: str
    query: str
    gold: List[GoldSpan] = field(default_factory=list)
    query_lang: str = "zh"
    category: str = "semantic"
    split: str = "dev"
    notes: str = ""
    annotated_by: str = ""
    annotated_at: str = ""

    # ---- 校验 ----
    def validate(self) -> None:
        if not self.query_id.strip():
            raise QrelsError("query_id 不能为空")
        if not self.query.strip():
            raise QrelsError(f"{self.query_id}: query 不能为空")
        if self.category not in CATEGORIES:
            raise QrelsError(
                f"{self.query_id}: category={self.category!r} 不合法, "
                f"可选 {sorted(CATEGORIES)}"
            )
        if self.split not in SPLITS:
            raise QrelsError(
                f"{self.query_id}: split={self.split!r} 不合法, 可选 {sorted(SPLITS)}"
            )
        # unanswerable 必须没有 gold; 其余必须有
        if self.category == "unanswerable":
            if self.gold:
                raise QrelsError(
                    f"{self.query_id}: unanswerable 不应有 gold "
                    f"(它有 {len(self.gold)} 条)"
                )
        elif not self.gold:
            raise QrelsError(f"{self.query_id}: 非 unanswerable 必须有 gold")
        for i, g in enumerate(self.gold):
            if not g.text or len(g.text.strip()) < 8:
                raise QrelsError(
                    f"{self.query_id}: gold[{i}].text 太短或为空 "
                    f"(至少 8 字符, 否则反查会多命中)"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_lang": self.query_lang,
            "gold": [g.to_dict() for g in self.gold],
            "category": self.category,
            "split": self.split,
            "notes": self.notes,
            "annotated_by": self.annotated_by,
            "annotated_at": self.annotated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QrelsItem":
        gold = []
        for g in d.get("gold") or []:
            if isinstance(g, str):
                gold.append(GoldSpan(text=g))
            else:
                gold.append(GoldSpan(
                    text=str(g.get("text") or ""),
                    source_uri=g.get("source_uri"),
                    block_id=g.get("block_id"),
                    span=list(g["span"]) if g.get("span") else None,
                ))
        return cls(
            query_id=str(d.get("query_id") or ""),
            query=str(d.get("query") or ""),
            gold=gold,
            query_lang=str(d.get("query_lang") or "zh"),
            category=str(d.get("category") or "semantic"),
            split=str(d.get("split") or "dev"),
            notes=str(d.get("notes") or ""),
            annotated_by=str(d.get("annotated_by") or ""),
            annotated_at=str(d.get("annotated_at") or ""),
        )


def load_qrels(path: str | Path) -> List[QrelsItem]:
    """读 JSONL, 逐条校验。任何一条不合法就抛错, 不静默跳过。"""
    p = Path(path)
    if not p.exists():
        raise QrelsError(f"qrels 文件不存在: {p}")
    items: List[QrelsItem] = []
    seen: Dict[str, int] = {}
    for lineno, line in enumerate(
        io.open(p, encoding="utf-8").read().splitlines(), 1
    ):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QrelsError(f"{p}:{lineno} JSON 解析失败: {exc}") from exc
        item = QrelsItem.from_dict(raw)
        item.validate()
        if item.query_id in seen:
            raise QrelsError(
                f"{p}:{lineno} query_id 重复: {item.query_id} "
                f"(首次出现在第 {seen[item.query_id]} 行)"
            )
        seen[item.query_id] = lineno
        items.append(item)
    if not items:
        raise QrelsError(f"{p}: 没有任何有效条目")
    return items


def save_qrels(items: Sequence[QrelsItem], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            it.validate()
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def summarize(items: Iterable[QrelsItem]) -> Dict[str, Any]:
    items = list(items)
    from collections import Counter
    by_cat = Counter(i.category for i in items)
    by_split = Counter(i.split for i in items)
    by_lang = Counter(i.query_lang for i in items)
    n_gold = sum(len(i.gold) for i in items)
    return {
        "total": len(items),
        "by_category": dict(by_cat),
        "by_split": dict(by_split),
        "by_lang": dict(by_lang),
        "gold_spans": n_gold,
        "avg_gold_per_query": round(n_gold / max(len(items), 1), 2),
    }


def today() -> str:
    return date.today().isoformat()
