r"""拒答（abstention）判定。

**为什么单独一个模块**：这段逻辑是**纯函数**，不依赖 FastAPI / HTTP。
放在 `rag_server.py` 里会让任何想用它的地方（脚本、评估、测试）都被迫拖上
一整套 Web 依赖 —— CI 里就撞过：测试只 import 了一下 rag_server，
就因为没装 `fastapi` 而整个失败。

判据是 **query 的 dense 排序 top-1 余弦相似度**低于阈值 —— 与
`eval/run_eval.py` 的「拒答能力」一节用的是**同一个量**。
⚠️ 是 **dense 自己的 top-1**，不是「最终排序 top-1 的 dense_score」——
后者在 rrf / rerank 之后已经换人了，两边定义不一致阈值就对不上。

阈值的标定见 `calibrate_abstain.py`。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: 拒答时返回给用户的固定话术。**不能**让生成模型去"编一个委婉的说法" ——
#: 那等于换个方式硬答。
ABSTAIN_ANSWER = (
    "语料里没有找到足以回答这个问题的内容。"
    "（当前检索到的最高相关度低于拒答阈值，为避免编造答案，这里不作答。）"
)

#: 阈值文件由 `calibrate_abstain.py --write` 生成。
THRESHOLD_FILENAME = ".abstain_threshold"


def should_abstain(
    dense_top_score: Optional[float], threshold: float
) -> bool:
    """相关度过低则拒答。

    - `threshold <= 0` -> **恒为 False**：不设阈值就完全保持原有行为
    - `dense_top_score is None` -> False：判据缺失时（例如 `mode=sparse`
      没跑 dense）**宁可作答**。「没依据就拒答」比答错更糟 ——
      用户会以为语料里真没有。
    - 边界用 `<` 而非 `<=`：边界上宁可作答，不误拒
    """
    if threshold <= 0 or dense_top_score is None:
        return False
    return dense_top_score < threshold


def load_threshold(
    env_raw: Optional[str] = None,
    threshold_file: Optional[Path] = None,
    default_dir: Optional[Path] = None,
) -> float:
    """拒答阈值：环境变量优先，其次读标定脚本写的 `.abstain_threshold`。

    为什么要支持文件：阈值是「语料 + 嵌入模型」绑定的，换任一个都得重算。
    只靠环境变量的话，重算完还得手工搬一次数字，很容易忘 ——
    然后线上就带着**过期阈值**跑（症状是「明明有答案却被拒答」，极难归因）。
    用 `calibrate_abstain.py --write` 写文件，服务自动跟着走。

    环境变量填错 / 文件损坏都**退化为 0（关闭）而不是崩溃**。

    两个来源都是参数（默认取真实来源），便于单测。
    """
    if env_raw is None:
        env_raw = os.getenv("RAG_ABSTAIN_THRESHOLD")
    if env_raw is not None and env_raw.strip():
        try:
            return float(env_raw)
        except ValueError:
            print("[启动] ⚠️ RAG_ABSTAIN_THRESHOLD=%r 不是数字，按 0（关闭）处理"
                  % env_raw)
            return 0.0
    if threshold_file is None:
        base = default_dir if default_dir is not None else Path(__file__).parent
        threshold_file = base / THRESHOLD_FILENAME
    if threshold_file.exists():
        try:
            return float(threshold_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return 0.0
    return 0.0
