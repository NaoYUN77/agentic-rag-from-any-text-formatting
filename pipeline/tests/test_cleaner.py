r"""Block 清洗测试。

重点覆盖目录（TOC）识别 —— 这是 issues/12 的最后一处损失。
"""

from __future__ import annotations

import unittest

from ingest.cleaner import clean_artifact, is_toc_block
from ingest.models import DocumentArtifact, DocumentBlock, IndexDecision, QualityReport


def block(bid: str, text: str, btype: str = "text") -> DocumentBlock:
    return DocumentBlock(
        block_id=bid,
        type=btype,
        text=text,
        markdown=text,
        section_path=[],
        order=0,
        quality_score=1.0,
    )


def artifact(blocks) -> DocumentArtifact:
    return DocumentArtifact(
        schema_version="1.0",
        artifact_id="doc_cleaner",
        source_uri="test",
        source_type="local_file",
        mime_type="text/plain",
        parser="plain_text",
        blocks=list(blocks),
        quality=QualityReport(score=1.0, status="high"),
        index_decision=IndexDecision("high", True, True),
    )


TOC_TEXT = "\n".join([
    "1.1. KEEPALIVED 3",
    "1.2. HAPROXY 3",
    "1.3. KEEPALIVED 和 HAPROXY 3",
    "2.1. 基本 KEEPALIVED 负载平衡器配置 4",
    "2.2. 三层 KEEPALIVED LOAD BALANCER 配置 6",
    "2.3. KEEPALIVED 计划概述 7",
    "2.4. 路由方法 8",
    "2.5. KEEPALIVED 的持久性和防火墙标记 11",
    "3.1. NAT 负载平衡器网络 12",
    "A.1. 先决条件 38",
    "A.2. 准备 HAPROXY 节点 38",
])


class TocDetectionTests(unittest.TestCase):
    def test_toc_block_is_detected(self) -> None:
        self.assertTrue(is_toc_block(block("b1", TOC_TEXT)))

    def test_numbered_list_is_not_toc(self) -> None:
        """回归: 有序列表不能误判成目录。

        单级编号（「1. 第一步」）不带点分层级，不该匹配 —— 否则会把
        正常正文当目录丢掉。检测器要求 `(\\.[0-9]+)+` 至少一层就是为了这个。
        """
        text = "\n".join(
            f"{i}. 第 {i} 步：配置负载平衡器并重启服务。" for i in range(1, 12)
        )
        self.assertFalse(is_toc_block(block("b1", text)))

    def test_plain_prose_is_not_toc(self) -> None:
        text = "\n".join([
            "负载平衡器把请求分发到多个后端服务器。",
            "它通过健康检查剔除故障节点。",
            "KEEPALIVED 使用 VRRP 协议实现高可用。",
            "HAPROXY 则专注于七层代理。",
            "两者可以配合使用。",
            "配置写在 /etc/keepalived/keepalived.conf。",
        ])
        self.assertFalse(is_toc_block(block("b1", text)))

    def test_short_block_is_not_toc(self) -> None:
        """行数太少不判定 —— 避免把「1.1 概述 3」这种孤立行误删。"""
        text = "\n".join(["1.1. KEEPALIVED 3", "1.2. HAPROXY 3"])
        self.assertFalse(is_toc_block(block("b1", text)))

    def test_heading_and_code_never_toc(self) -> None:
        self.assertFalse(is_toc_block(block("b1", TOC_TEXT, btype="heading")))
        self.assertFalse(is_toc_block(block("b1", TOC_TEXT, btype="code")))


class CleanArtifactTocTests(unittest.TestCase):
    def test_toc_dropped_and_warned(self) -> None:
        art = artifact([
            block("b1", "正文第一段，讲负载平衡器的基本概念。"),
            block("b2", TOC_TEXT),
            block("b3", "正文第二段，讲 KEEPALIVED 的配置方法。"),
        ])
        out = clean_artifact(art)
        ids = [b.block_id for b in out.blocks]
        self.assertNotIn("b2", ids)
        self.assertIn("b1", ids)
        self.assertIn("b3", ids)
        # 丢内容必须留痕
        self.assertTrue(
            any("dropped_toc_blocks" in w and "b2" in w for w in out.warnings),
            out.warnings,
        )

    def test_no_warning_when_no_toc(self) -> None:
        art = artifact([block("b1", "一段普通正文，没有任何目录特征。")])
        out = clean_artifact(art)
        self.assertEqual(out.warnings, [])
        self.assertEqual(len(out.blocks), 1)


if __name__ == "__main__":
    unittest.main()
