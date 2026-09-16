"""语料预处理: 内容分类 + 结构提取。

解决的问题:
    issues/01   目录/版权/正文没有分类处理, Markdown 标记混进正文
    issues/05   section 提取不干净(标题和正文粘在一起)

做法:
    1. 按空行切段落
    2. 识别标题行, 维护【章节栈】, 生成层级 section 路径
    3. 标题行不进正文(只更新 section), 从而剥离 Markdown 标记
    4. 按位置和内容特征给每个块分类

产出: 带类型和章节归属的 Block 列表, 供分块使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Block:
    text: str                      # 正文(已剥离 Markdown 标记)
    kind: str                      # body | toc | copyright | frontmatter | callout
    section: str                   # 层级章节路径, 如 "第 2 章 X > 2.1. Y"
    order: int
    raw: str = ""                  # 原始段落(便于排查)


# --------------------------------------------------------------------------
# 识别规则
# --------------------------------------------------------------------------
# 提示框标记: 它们不是章节标题, 只是"这段是注意事项"
CALLOUT_WORDS = {
    "注意", "重要", "警告", "提示", "说明", "备注", "备注说明",
    "软件", "其他资源", "其它资源", "先决条件",
}

COPYRIGHT_PAT = re.compile(
    r"法律通告|版权|copyright|licensed under|cc-by|creative commons|"
    r"all rights reserved|许可协议",
    re.I,
)

TOC_ENTRY_PAT = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S.*?\s+\d+\s*$")


# shell 命令名(常见于配置示例)。这些行以 "# " 开头是"root 提示符"的意思,
# 不是 Markdown 标题 —— MinerU 从 PDF 抽取时不会加代码围栏, 只能靠内容判别。
SHELL_CMDS = {
    "systemctl", "service", "chkconfig", "firewall-cmd", "firewalld", "iptables",
    "arptables", "ip", "ifconfig", "route", "arp", "echo", "export", "set",
    "cat", "vi", "vim", "nano", "yum", "dnf", "rpm", "apt", "apt-get",
    "systemd", "sysctl", "modprobe", "lsmod", "grep", "sed", "awk", "tar",
    "wget", "curl", "ssh", "scp", "mkdir", "chmod", "chown", "useradd",
}


def looks_like_shell(title: str) -> bool:
    """判断 "# xxx" 里的 xxx 是不是 shell 命令行而不是标题。"""
    t = title.strip()
    if not t:
        return False
    # 含中文 -> 是标题
    if re.search(r"[\u4e00-\u9fff]", t):
        return False
    # 带命令行选项
    if " --" in t or t.startswith("-"):
        return True
    first = t.split()[0].lower().strip(":：")
    if first in SHELL_CMDS:
        return True
    # 配置文件 / 服务名结尾
    if re.search(r"\.(conf|service|sh|py|json|xml)$", t, re.I):
        return True
    return False


def is_heading(para: str) -> Optional[tuple]:
    """段落是不是 Markdown 标题? 返回 (级别, 标题文本)。

    会排除两种情况:
        1. shell 的 root 提示符 "# systemctl start xxx"
        2. 单行里的 "#" 只是代码注释
    """
    m = re.match(r"^(#{1,6})\s+(.+?)\s*$", para, re.S)
    if not m:
        return None
    level, title = len(m.group(1)), m.group(2).strip()
    if looks_like_shell(title):
        return None
    return level, title


def is_callout(title: str) -> bool:
    t = title.strip().strip("：:")
    return t in CALLOUT_WORDS


def toc_ratio(para: str) -> float:
    """段落里符合"目录条目"格式的行占比。"""
    lines = [l for l in para.splitlines() if l.strip()]
    if not lines:
        return 0.0
    hit = sum(1 for l in lines if TOC_ENTRY_PAT.match(l))
    return hit / len(lines)


def classify(para: str, idx: int, in_front: bool) -> str:
    """给段落分类。"""
    if COPYRIGHT_PAT.search(para):
        return "copyright"
    if toc_ratio(para) >= 0.5:
        return "toc"
    if in_front:
        return "frontmatter"
    return "body"


# --------------------------------------------------------------------------
# 章节栈
# --------------------------------------------------------------------------
class SectionStack:
    """维护标题层级, 生成 "章 > 节 > 小节" 这样的路径。"""

    def __init__(self) -> None:
        self._levels: List[str] = []

    def push(self, level: int, title: str) -> None:
        # level 是 Markdown 的 # 数量(1 最粗)
        while len(self._levels) >= level:
            self._levels.pop()
        self._levels.append(title)

    @property
    def path(self) -> str:
        return " > ".join(self._levels) if self._levels else "文档开头"


# --------------------------------------------------------------------------
# 主解析
# --------------------------------------------------------------------------
def split_paragraphs(md: str) -> List[str]:
    """按空行切段落。保留标题行作为独立段落。"""
    parts = re.split(r"\n\s*\n", md)
    return [p.strip() for p in parts if p.strip()]


def find_body_start(paras: List[str]) -> int:
    """正文起点 = 第一个一级标题(#)的位置。

    前 94 行那种"书名/作者/法律通告/目录"通常只有 ## 级标题,
    真正的正文章节用 # 标记。
    """
    for i, p in enumerate(paras):
        h = is_heading(p)
        if h and h[0] == 1:
            return i
    return 0


def parse_markdown(md: str, drop_callouts: bool = True) -> List[Block]:
    """把 Markdown 解析成带类型和章节归属的 Block 列表。"""
    paras = split_paragraphs(md)
    body_start = find_body_start(paras)

    stack = SectionStack()
    blocks: List[Block] = []
    order = 0

    for i, p in enumerate(paras):
        in_front = i < body_start
        h = is_heading(p)

        if h:
            level, title = h
            if is_callout(title):
                # 提示框: 不改章节, 标记行本身丢掉(内容在后面)
                if not drop_callouts:
                    blocks.append(Block(text=title, kind="callout",
                                        section=stack.path, order=order, raw=p))
                    order += 1
                continue
            # 真标题: 更新章节栈, 标题不进正文
            stack.push(level, title)
            continue

        kind = classify(p, i, in_front)
        blocks.append(Block(text=p, kind=kind, section=stack.path,
                            order=order, raw=p))
        order += 1

    return blocks


def summarize(blocks: List[Block]) -> dict:
    """统计各类块的数量和字数, 便于验证。"""
    from collections import Counter
    cnt = Counter(b.kind for b in blocks)
    chars = Counter()
    for b in blocks:
        chars[b.kind] += len(b.text)
    return {
        "blocks": len(blocks),
        "by_kind": dict(cnt),
        "chars_by_kind": dict(chars),
        "sections": len({b.section for b in blocks if b.kind == "body"}),
    }

