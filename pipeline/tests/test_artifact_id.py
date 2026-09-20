r"""artifact_id 必须**确定性** —— 回归测试。

背景
----
4 个 parser 原本都写 `artifact_id=f"doc_{uuid.uuid4().hex[:12]}"`, 每次重建
同一份语料都得到不同的 id。实测两次重建: `doc_049627f36e74` vs `doc_0e5e3c88e501`。

为什么这是个真问题
------------------
`chunk_id = f"{artifact_id}_c{index:04d}"`, 而 `point_id` 又由 chunk_id 派生。
id 不稳定意味着:
  - 增量更新无法对齐 (无法按 artifact 删除旧 chunk)
  - **跨 run 无法按 chunk_id 做逐 chunk 对比** —— 这一点直接卡住了
    "改切块参数/换 parser 后重建再 diff" 的实验循环
    (实测踩到: 按 chunk_id 求交集得到 0, 误判为"没有变化")
  - 依赖 chunk_id 的历史产物 (eval/runs/*/per_query.jsonl) 会失效

修法见 `ingest/models.stable_artifact_id`。
"""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

from ingest.models import SourcePayload, stable_artifact_id

_PIPELINE = Path(__file__).resolve().parent.parent
_PARSERS = ["html", "markdown", "plain", "pdf_common"]


def _src(uri: str = "https://example.com/a", final_uri: str = "",
         sha256: str = "deadbeef") -> SourcePayload:
    data = b"x"
    return SourcePayload(
        source_type="url", uri=uri, final_uri=final_uri or uri,
        mime_type="text/html", content_type="text/html",
        data=data, encoding="utf-8",
        size_bytes=len(data), sha256=sha256,
    )


class StableArtifactIdTests(unittest.TestCase):
    """派生规则本身。"""

    def test_same_source_gives_same_id(self) -> None:
        self.assertEqual(stable_artifact_id(_src()), stable_artifact_id(_src()))

    def test_different_uri_gives_different_id(self) -> None:
        self.assertNotEqual(
            stable_artifact_id(_src(uri="https://example.com/a")),
            stable_artifact_id(_src(uri="https://example.com/b")),
        )

    def test_format_matches_legacy(self) -> None:
        """保持 `doc_<12hex>` 形态, 避免下游按前缀/长度解析的地方失效。"""
        aid = stable_artifact_id(_src())
        self.assertTrue(aid.startswith("doc_"))
        self.assertEqual(len(aid), 4 + 12)
        int(aid[4:], 16)  # 必须是合法十六进制

    def test_prefers_final_uri_over_uri(self) -> None:
        """final_uri 是权威来源 (重定向后的真实地址)。"""
        a = stable_artifact_id(_src(uri="https://a.example/x",
                                    final_uri="https://real.example/y"))
        b = stable_artifact_id(_src(uri="https://other.example/z",
                                    final_uri="https://real.example/y"))
        self.assertEqual(a, b, "final_uri 相同就应得到同一个 id")

    def test_falls_back_to_sha256_without_uri(self) -> None:
        """没有 URI 时用内容哈希兜底, 不能抛异常。"""
        a = stable_artifact_id(_src(uri="", final_uri="", sha256="aa" * 32))
        b = stable_artifact_id(_src(uri="", final_uri="", sha256="bb" * 32))
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("doc_"))

    def test_raises_without_any_basis(self) -> None:
        """既无 URI 也无 sha256 时必须显式失败, 不能静默生成随机 id。"""
        with self.assertRaises(ValueError):
            stable_artifact_id(_src(uri="", final_uri="", sha256=""))

    def test_ignores_whitespace_in_uri(self) -> None:
        a = stable_artifact_id(_src(uri="  https://example.com/a  "))
        b = stable_artifact_id(_src(uri="https://example.com/a"))
        self.assertEqual(a, b)


class NoRandomArtifactIdTests(unittest.TestCase):
    """架构不变量: parser 不得再用随机数生成 artifact_id。"""

    def test_no_uuid4_in_parsers(self) -> None:
        offenders = []
        for name in _PARSERS:
            path = _PIPELINE / "ingest" / "parsers" / f"{name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # uuid.uuid4() 或 uuid.uuid1()
                if (isinstance(node, ast.Attribute)
                        and node.attr in ("uuid1", "uuid4")
                        and getattr(node.value, "id", "") == "uuid"):
                    offenders.append(f"{name}.py:{node.lineno}")
        self.assertEqual(
            offenders, [],
            f"以下位置仍用随机 uuid 生成 id, 会破坏 chunk_id 稳定性: {offenders}",
        )

    def test_all_parsers_call_stable_artifact_id(self) -> None:
        for name in _PARSERS:
            path = _PIPELINE / "ingest" / "parsers" / f"{name}.py"
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                "stable_artifact_id(source)", text,
                f"{name}.py 未使用 stable_artifact_id",
            )


class CrossProcessDeterminismTests(unittest.TestCase):
    """真正要紧的是**跨进程**一致 (uuid4 正是在这里失败)。"""

    def test_reparse_twice_in_fresh_interpreter(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys, json; sys.path.insert(0, r'%s');"
            "from ingest.parsers.markdown import parse_markdown_text;"
            "from ingest.models import SourcePayload;"
            "d=b'## A\\n\\nx\\n\\n### B\\n\\ny\\n\\n## C\\n\\nz\\n';"
            "s=SourcePayload(source_type='url',uri='https://e.com/p',"
            "final_uri='https://e.com/p',mime_type='text/markdown',"
            "content_type='text/markdown',data=d,encoding='utf-8',"
            "size_bytes=len(d),sha256='%s');"
            "print(parse_markdown_text(d.decode(), s).artifact_id)"
            % (_PIPELINE, hashlib.sha256(b"x").hexdigest())
        )
        outs = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True,
                encoding="utf-8", errors="replace", cwd=str(_PIPELINE),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
            outs.append(proc.stdout.strip())
        self.assertEqual(outs[0], outs[1], "两次独立进程解析得到不同 artifact_id")
        self.assertTrue(outs[0].startswith("doc_"))


if __name__ == "__main__":
    unittest.main()
