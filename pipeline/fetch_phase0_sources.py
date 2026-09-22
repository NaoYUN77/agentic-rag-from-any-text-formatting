r"""用 playwright-cli 抓取 Phase 0 的 10 篇源 HTML 到 experiments/_sources/。

为什么不用 requests
------------------
- OpenAI 三篇对非浏览器 UA 直接 403 (Cloudflare invisible challenge)。
  实测: curl 403, headless Chrome dump-dom 失败, playwright 开真浏览器
  也只拿到 "请稍候…" 页面 —— 说明 Cloudflare 在这个环境下不放行。
- Anthropic 七篇用 requests 可以过, 但为了链路统一, 全部走浏览器。

调用方式
--------
playwright-cli 是**进程级会话**: 每个 Bash 调用是独立进程, 所以
open 与后续命令必须在同一次调用里串起来。但 Python 里用
subprocess 连续调用就能绕过这个限制 —— 这正是本脚本的做法。

用法:
    cd pipeline
    .\.venv_rag\Scripts\python.exe fetch_phase0_sources.py
    .\.venv_rag\Scripts\python.exe fetch_phase0_sources.py --only openai_agents_api
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NODE_BIN = Path(r"C:\Users\l\.workbuddy-ai\binaries\node\versions\22.22.2-2")
OUT_DIR = Path("experiments/_sources")

DOCS = [
    ("anthropic_contextual_retrieval",
     "https://www.anthropic.com/engineering/contextual-retrieval"),
    ("anthropic_building_effective_agents",
     "https://www.anthropic.com/engineering/building-effective-agents"),
    ("anthropic_context_engineering",
     "https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents"),
    ("anthropic_multi_agent_research",
     "https://www.anthropic.com/engineering/multi-agent-research-system"),
    ("anthropic_agent_skills",
     "https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills"),
    ("anthropic_code_execution_mcp",
     "https://www.anthropic.com/engineering/code-execution-with-mcp"),
    ("anthropic_demystifying_evals",
     "https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents"),
    ("openai_scaling_storage",
     "https://openai.com/index/scaling-storage-one-billion-users-part-one/"),
    ("openai_agents_api",
     "https://openai.com/index/introducing-the-agents-api/"),
    ("openai_model_misalignment",
     "https://openai.com/index/model-misalignment-reporting-framework/"),
]


def _env() -> dict:
    import os

    env = dict(os.environ)
    env["PATH"] = str(NODE_BIN) + os.pathsep + env.get("PATH", "")
    return env


def _cli() -> str:
    """Windows 上 npm 的全局 bin 是 .cmd, Popen 不会自动补扩展名。"""
    for name in ("playwright-cli.cmd", "playwright-cli"):
        candidate = NODE_BIN / name
        if candidate.exists():
            return str(candidate)
    return "playwright-cli"


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    args = [_cli()] + args[1:]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=_env(),
        shell=False,
    )


def _parse_result(stdout: str) -> str | None:
    """从 playwright-cli 输出里取 `### Result` 段。"""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "### Result":
            if i + 1 < len(lines):
                raw = lines[i + 1].strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
    return None


#: 拦截/挑战页的特征串（小写比对）。命中任一即判定"这不是正文"。
#:
#: ⚠️ **光看大小拦不住** —— 实测 openai.com 返回的 Cloudflare 挑战页有 **11,437 字节**，
#: 能过 `len(html) < 5000` 那道检查，于是被当成正文存了下来：
#: 后续解析出一篇**没有标题、没有正文**的"文档"，却毫无告警。
#: （这正是 3 篇 openai 文档结构丢失的同类问题 —— 那次是回退用了旧的
#: `document.md`，这次差点把一个挑战页写进语料目录。）
_CHALLENGE_MARKERS = (
    "just a moment",
    "请稍候",
    "enable javascript",
    "cf-challenge",
    "challenge-platform",
    "attention required",
    "access denied",
    "captcha",
)


def _looks_like_challenge(html: str) -> bool:
    """拿到的更像拦截/挑战页而不是正文。"""
    head = html[:30000].lower()
    return any(m in head for m in _CHALLENGE_MARKERS)


def fetch_one(doc: str, url: str, session: str) -> tuple[bool, str]:
    target = OUT_DIR / f"{doc}.html"
    if target.exists() and target.stat().st_size > 5000:
        return True, f"已存在 {target.stat().st_size} bytes"

    # open + 拿 HTML 必须尽量连续; session 是进程级, 但 playwright-cli
    # 会把会话状态写在磁盘上, 同一 session 名跨进程可复用 (实测可行)。
    r = _run(["playwright-cli", f"-s={session}", "open", "--browser=chrome", url])
    if r.returncode != 0:
        return False, f"open 失败: {r.stderr[:200]}"

    time.sleep(4)  # 等 JS 渲染 / challenge

    r = _run([
        "playwright-cli", f"-s={session}", "run-code",
        "async (page) => { const h = await page.content(); return h; }",
    ])
    html = _parse_result(r.stdout)
    _run(["playwright-cli", f"-s={session}", "close"])

    if not html or not isinstance(html, str):
        return False, f"取 HTML 失败: {(r.stdout or r.stderr)[:200]}"
    if len(html) < 5000:
        return False, f"HTML 过小 ({len(html)}), 可能被 challenge 拦住"
    # ⚠️ 大小检查**不够**：挑战页可以有 1 万多字节（实测 openai.com）。
    # 必须在写盘前识别出来，否则会把挑战页当正文存进语料目录，
    # 后面解析出一篇没有标题、没有正文的"文档"却毫无告警。
    if _looks_like_challenge(html):
        title = ""
        m = re.search(r"<title[^>]*>([^<]{0,60})", html, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
        return False, ("拿到的是拦截/挑战页而不是正文（title=%r, %d bytes）"
                       "—— 换抓取方式或换语料源" % (title, len(html)))

    target.write_text(html, encoding="utf-8")
    return True, f"{len(html)} bytes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = [(d, u) for d, u in DOCS if args.only in (None, d)]

    ok = fail = 0
    for doc, url in docs:
        print(f"[fetch] {doc}", flush=True)
        try:
            good, msg = fetch_one(doc, url, session=f"s_{doc}")
        except subprocess.TimeoutExpired:
            good, msg = False, "超时"
        print(f"        {'OK  ' if good else 'FAIL'} {msg}", flush=True)
        ok += good
        fail += not good

    print()
    print(f"done: ok={ok} fail={fail}")
    for f in sorted(OUT_DIR.glob("*.html")):
        print(f"  {f.stat().st_size:>9}  {f.name}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
