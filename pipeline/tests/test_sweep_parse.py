r"""扫描脚本的指标解析测试。

回归背景（真实踩过）：`_parse_metrics` 原先**全文扫行**取 `| dense | ... |` 表格行，
但报告里有两张表 —— 「分阶段指标」和「分阶段 × 类别」。
后者含 `unanswerable` 行（全 0），**后出现的会把真实值覆盖掉**。

结果：`sweep_chunk_params.json` 里 600/800/1200 三个变体的五个阶段全是 `0.0`。
任何人读那份 JSON 都会得出「chunk 大小没影响」的错误结论。

修复是「先切出『分阶段指标』这一节再解析」。这里把这个不变量钉住。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from sweep_chunk_params import _parse_metrics  # noqa: E402


REPORT = """# 检索评估报告

## 分阶段指标

| 阶段 | hit@5 | recall@5 | recall@20 | MRR@20 | nDCG@5 | n |
|---|---|---|---|---|---|---|
| dense | 0.765 | 0.735 | 0.912 | 0.654 | 0.644 | 34 |
| sparse | 0.765 | 0.765 | 0.824 | 0.615 | 0.645 | 34 |
| union | 0.824 | 0.794 | 0.971 | 0.651 | 0.659 | 34 |
| rrf | 0.882 | 0.853 | 0.941 | 0.718 | 0.738 | 34 |
| rerank | 0.941 | 0.912 | 0.912 | 0.820 | 0.838 | 34 |

## 分阶段 × 类别

| 阶段 | 类别 | hit@5 | recall@5 | recall@20 | MRR@20 | n |
|---|---|---|---|---|---|---|
| dense | exact | 1.000 | 1.000 | 1.000 | 0.894 | 11 |
| dense | unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| sparse | unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| union | unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| rrf | unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| rerank | unanswerable | 0.000 | 0.000 | 0.000 | 0.000 | 0 |

## 每条 query 的排名变化

```text
cr_01 gold=1 dense=1.000
```
"""


class ParseMetricsTests(unittest.TestCase):
    def test_does_not_get_overwritten_by_unanswerable_rows(self) -> None:
        """回归：`unanswerable` 行的全 0 不能覆盖真实值。

        这是 `sweep_chunk_params.json` 里五个阶段全 0 的根因。
        """
        rows = _parse_metrics(REPORT)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows["dense"]["hit@5"], 0.765)
        self.assertEqual(rows["sparse"]["hit@5"], 0.765)
        self.assertEqual(rows["union"]["hit@5"], 0.824)
        self.assertEqual(rows["rrf"]["hit@5"], 0.882)
        self.assertEqual(rows["rerank"]["hit@5"], 0.941)
        for stage in ("dense", "sparse", "union", "rrf", "rerank"):
            self.assertNotEqual(rows[stage]["hit@5"], 0.0,
                                "%s 被 unanswerable 行覆盖成 0 了" % stage)

    def test_all_five_metrics_are_captured(self) -> None:
        rows = _parse_metrics(REPORT)
        self.assertEqual(
            sorted(rows["dense"]),
            ["MRR@20", "hit@5", "nDCG@5", "recall@20", "recall@5"],
        )
        self.assertEqual(rows["dense"]["recall@20"], 0.912)
        self.assertEqual(rows["dense"]["MRR@20"], 0.654)
        self.assertEqual(rows["dense"]["nDCG@5"], 0.644)

    def test_missing_section_returns_empty(self) -> None:
        self.assertEqual(_parse_metrics("# 别的报告\n\n没有指标表\n"), {})

    def test_short_rows_are_ignored(self) -> None:
        md = "## 分阶段指标\n\n| dense | 0.1 |\n"
        self.assertEqual(_parse_metrics(md), {})


if __name__ == "__main__":
    unittest.main()
