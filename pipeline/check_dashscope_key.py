"""诊断阿里云百炼 API Key 是否可用。

用法:
    $env:DASHSCOPE_API_KEY = "sk-真实Key"
    python check_dashscope_key.py

可选: 如果 Key 绑定了特定地域/业务空间, 设置自定义 Host
    $env:DASHSCOPE_API_HOST = "https://xxxx.cn-beijing.maas.aliyuncs.com/api/v1"
"""

import os
import sys

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_HOST = "https://dashscope.aliyuncs.com/api/v1"
PATH = "/services/embeddings/multimodal-embedding/multimodal-embedding"


def mask(key: str) -> str:
    if len(key) <= 10:
        return key[:2] + "*" * max(len(key) - 2, 0)
    return key[:6] + "*" * (len(key) - 10) + key[-4:]


def main() -> int:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        print("[X] 环境变量 DASHSCOPE_API_KEY 未设置")
        return 1

    print("=" * 70)
    print("第一步: Key 形态检查")
    print("=" * 70)
    print("  长度         : %d" % len(key))
    print("  掩码显示     : %s" % mask(key))
    print("  首尾有空白   : %s" % ("是 <-- 必须去掉!" if key != key.strip() else "无"))
    print("  含换行符     : %s" % ("是 <-- 必须去掉!" if ("\n" in key or "\r" in key) else "无"))
    print("  以 sk- 开头  : %s" % ("是" if key.startswith("sk-") else "否 <-- 百炼 Key 应以 sk- 开头"))
    placeholder = ("你的" in key) or ("xxx" in key.lower()) or ("your" in key.lower())
    print("  疑似占位符   : %s" % ("是 <-- 还是模板文字, 必须换成真实 Key" if placeholder else "否"))
    print()

    host = os.getenv("DASHSCOPE_API_HOST", DEFAULT_HOST).rstrip("/")
    url = host + PATH
    print("=" * 70)
    print("第二步: 调用测试")
    print("=" * 70)
    print("  URL: %s" % url)
    print()

    payload = {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": "测试文本"}]},
        "parameters": {"dimension": 1024},
    }
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        print("[X] 网络请求失败: %s" % exc)
        return 1

    print("  HTTP 状态码: %d" % resp.status_code)
    print("  响应体: %s" % resp.text[:600])
    print()

    if resp.status_code == 200:
        data = resp.json()
        embs = data["output"]["embeddings"]
        print("[OK] 调用成功! 返回向量数 %d, 维度 %d" % (len(embs), len(embs[0]["embedding"])))
        print("     usage: %s" % data.get("usage"))
        return 0

    if resp.status_code == 401:
        print("[X] 401 认证失败。按可能性排查:")
        print("    1. Key 是占位符或复制不完整(最常见)")
        print("    2. 该 Key 已被删除/禁用")
        print("    3. 百炼服务尚未开通, 或 Key 不属于当前账号")
        print("    4. Key 绑定了特定地域/业务空间, 需要用带 WorkspaceId 的 Host")
        print("       (见文件顶部注释里的 DASHSCOPE_API_HOST)")
    elif resp.status_code == 403:
        print("[X] 403 权限不足。确认已购买/开通 qwen3-vl-embedding 的额度。")
    elif resp.status_code == 400:
        print("[X] 400 请求参数问题(不是 Key 的问题), 把完整响应发出来看看。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
