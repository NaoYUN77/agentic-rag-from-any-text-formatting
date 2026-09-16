"""用 MinerU 把 PDF 解析成 Markdown, 并做基础清洗。

MinerU 由上海人工智能实验室开源, 本项目通过官方 CLI `mineru-open-api` 调用。
注意: flash-extract 模式会把文档上传到 mineru.net 处理, 属于外部服务调用。
     敏感文档请改用本地部署的 MinerU, 或 `extract` + 自有 token 的模式。

CLI 安装:
    uv tool install mineru-open-api
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

CLI_NAME = "mineru-open-api"


def find_cli() -> str:
    """定位 mineru-open-api 可执行文件。"""
    exe = shutil.which(CLI_NAME)
    if exe:
        return exe
    for cand in (
        Path.home() / ".local" / "bin" / (CLI_NAME + ".exe"),
        Path.home() / ".local" / "bin" / CLI_NAME,
        Path.home() / ".cargo" / "bin" / (CLI_NAME + ".exe"),
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "找不到 mineru-open-api, 请先安装:\n  uv tool install mineru-open-api"
    )


def extract(
    pdf_path: str,
    out_path: Optional[str] = None,
    pages: Optional[str] = None,
    language: str = "ch",
    timeout: int = 900,
) -> str:
    """调用 MinerU 把 PDF 转成 Markdown。

    pages 形如 "1-20"; 留空表示全文(受 flash-extract 20 页上限约束)。
    返回 Markdown 文本。
    """
    cli = find_cli()
    cmd: List[str] = [cli, "flash-extract", str(pdf_path)]
    if pages:
        cmd += ["--pages", pages]
    if language:
        cmd += ["--language", language]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", str(out_path)]

    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError("MinerU 解析失败 (exit=%s):\n%s" % (proc.returncode, tail))

    if out_path:
        return Path(out_path).read_text(encoding="utf-8", errors="replace")
    return proc.stdout


# --------------------------------------------------------------------------
# 清洗: MinerU 输出的是"原始 Markdown", 带目录点线、图片占位符等噪声
# --------------------------------------------------------------------------
_TOC_LEADER = re.compile(r"^[\s.·…]{10,}.*$")          # 目录点线  "....第 1 章"
_IMAGE_PLACEHOLDER = re.compile(r"^<!--\s*image\s*-->$", re.I)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")           # 单独成行的页码
_MANY_BLANK = re.compile(r"\n{3,}")
# 连续的目次条目, 形如 "1.1. KEEPALIVED ...... 5"
_TOC_ENTRY = re.compile(r"^.{0,80}\.{3,}\s*\d+\s*$")


def clean(markdown: str, drop_headings: bool = False) -> str:
    """去掉目录点线、图片占位符、孤立页码, 压缩连续空行。

    drop_headings=True 时连 Markdown 标题一起去掉, 只留正文。
    """
    kept: List[str] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if _IMAGE_PLACEHOLDER.match(s):
            continue
        if _TOC_LEADER.match(s) or _TOC_ENTRY.match(s):
            continue
        if _PAGE_NUMBER.match(s):
            continue
        if drop_headings and s.startswith("#"):
            continue
        kept.append(line)
    return _MANY_BLANK.sub("\n\n", "\n".join(kept)).strip()


def load_markdown(path: str, do_clean: bool = True) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return clean(text) if do_clean else text
