r"""抓取守卫测试：识别「拦截/挑战页」。

回归背景（真实踩过）：`fetch_phase0_sources.py` 只用 `len(html) < 5000` 判断
抓取是否成功。实测 openai.com 返回的 **Cloudflare 挑战页有 11,437 字节**，
轻松过了那道检查 —— 于是挑战页会被当成正文写进 `experiments/_sources/`，
后续解析出一篇**没有标题、没有正文**的"文档"，却毫无告警。

这正是 3 篇 openai 文档结构丢失的**同类**问题（那次是回退用了旧的
`document.md`，结果是 0 标题）。**光看大小拦不住这类问题，必须识别内容特征。**
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from fetch_phase0_sources import _looks_like_challenge  # noqa: E402


CHALLENGE = (
    '<html><head><title>请稍候…</title></head>'
    '<body><div id="challenge-platform">'
    "Enable JavaScript and cookies to continue"
    "</div></body></html>"
)

ENGLISH_CHALLENGE = (
    "<html><head><title>Just a moment...</title></head>"
    '<body><div class="cf-challenge">Checking your browser</div></body></html>'
)

REAL_ARTICLE = (
    "<html><head><title>Building Effective AI Agents | Anthropic</title></head>"
    "<body><h1>Building Effective AI Agents</h1>"
    "<p>We've worked with dozens of teams building LLM agents across industries."
    " The most successful implementations use simple, composable patterns.</p>"
    "<h2>Building blocks</h2><p>In this section, we'll explore the common patterns"
    " for agentic systems in production.</p></body></html>"
)


class ChallengeDetectionTests(unittest.TestCase):
    def test_chinese_challenge_page_is_detected(self) -> None:
        """Cloudflare 中文挑战页（实测 openai.com 返回的就是这个）。"""
        self.assertTrue(_looks_like_challenge(CHALLENGE))

    def test_english_challenge_page_is_detected(self) -> None:
        self.assertTrue(_looks_like_challenge(ENGLISH_CHALLENGE))

    def test_real_article_is_not_flagged(self) -> None:
        self.assertFalse(_looks_like_challenge(REAL_ARTICLE))

    def test_size_alone_is_not_enough(self) -> None:
        """核心回归：挑战页可以**很大**，大小检查拦不住它。

        把挑战页 padding 到 20KB —— 仍然必须被识别出来。
        """
        padded = CHALLENGE.replace(
            "</body>", "<div>" + "x" * 20000 + "</div></body>")
        self.assertGreater(len(padded), 5000)          # 过得了大小检查
        self.assertTrue(_looks_like_challenge(padded))  # 但仍被内容特征抓住

    def test_real_anthropic_html_is_not_flagged(self) -> None:
        """用真实抓取的正文回归 —— 确认不会误杀。

        文件不存在时跳过（语料不入公开仓库）。
        """
        p = _PIPELINE / "experiments" / "_sources" / "anthropic_agent_skills.html"
        if not p.exists():
            self.skipTest("语料文件不存在（第三方语料不入库）")
        html = p.read_text(encoding="utf-8", errors="replace")
        self.assertGreater(len(html), 100_000)
        self.assertFalse(_looks_like_challenge(html))

    def test_markers_only_checked_in_head(self) -> None:
        """只在开头 30KB 里找特征 —— 正文里偶尔出现 "captcha" 不该误判。

        一篇讲反爬的正当文章可能在正文里提到这些词，不该因此被丢掉。
        """
        long_article = REAL_ARTICLE.replace(
            "</body>", "<div>" + "y" * 40000 + "</div>"
            "<p>Our captcha handling is documented elsewhere.</p></body>")
        self.assertFalse(_looks_like_challenge(long_article))


if __name__ == "__main__":
    unittest.main()
