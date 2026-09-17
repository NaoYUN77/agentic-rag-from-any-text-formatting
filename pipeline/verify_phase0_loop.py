"""Phase 0 功能闭环可用性验证。

不改动任何生产代码，只通过 HTTP 调用在线服务，验证：
  1. dense 检索
  2. sparse 检索
  3. hybrid + RRF
  4. rerank
  5. generation + citations
并交叉校验 dense / sparse / hybrid 返回同一 chunk_id。

用法:
    .\\.venv_rag\\Scripts\\python.exe verify_phase0_loop.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
TIMEOUT = 120

# 本机回环地址必须绕过系统代理, 否则会被代理拦截成 502
for _var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
_PROXYLESS = urllib.request.build_opener(urllib.request.ProxyHandler({}))

QUERIES = [
    "What is contextual retrieval and how does it improve retrieval accuracy?",
    "How do agent skills help agents handle real world tasks?",
    "Why does context engineering matter for building agents?",
    "How does BM25 sparse retrieval differ from dense embedding retrieval?",
    "What are common failure modes when building multi-agent research systems?",
]


def post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _PROXYLESS.open(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    with _PROXYLESS.open(BASE + path, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def brief(hits: list, n: int = 3) -> list:
    out = []
    for h in hits[:n]:
        out.append(
            {
                "rank": h.get("rank"),
                "chunk_id": h.get("chunk_id"),
                "score": h.get("score"),
                "dense": h.get("dense_score"),
                "sparse": h.get("sparse_score"),
                "rrf": h.get("rrf_score"),
                "rerank": h.get("rerank_score"),
                "section": (h.get("section") or "")[:46],
            }
        )
    return out


def cid_set(hits: list) -> set:
    return {h.get("chunk_id") for h in hits if h.get("chunk_id")}


def main() -> int:
    failures: list[str] = []

    info = get("/api/info")
    print("=" * 78)
    print("服务信息")
    print("=" * 78)
    for key in (
        "collection",
        "points_count",
        "sparse_collection",
        "sparse_points_count",
        "model",
        "chat_model",
        "rerank_model",
        "mode",
        "rrf_k",
    ):
        print(f"  {key:22}: {info.get(key)}")

    if info.get("collection") != "phase0_dense":
        failures.append("服务未指向 phase0_dense")

    for qi, query in enumerate(QUERIES, 1):
        print()
        print("=" * 78)
        print(f"[Q{qi}] {query}")
        print("=" * 78)

        results: dict[str, dict] = {}
        for mode in ("dense", "sparse", "hybrid"):
            # rerank 默认为 True, 验证纯通道分时必须显式关闭,
            # 否则 score 会被 rerank_score 覆盖。
            try:
                r = post(
                    "/api/search",
                    {
                        "query": query,
                        "mode": mode,
                        "top_k": 5,
                        "candidate_k": 20,
                        "rerank": False,
                    },
                )
            except urllib.error.HTTPError as exc:
                failures.append(f"Q{qi} {mode} HTTP {exc.code}: {exc.read()[:200]!r}")
                print(f"  {mode:7}: HTTP {exc.code}")
                continue

            hits = r.get("hits") or []
            results[mode] = r
            print(f"  {mode:7}: hits={len(hits)} elapsed={r.get('elapsed_ms')}ms")
            for row in brief(hits):
                print(
                    f"      #{row['rank']} {row['chunk_id']} "
                    f"score={row['score']} d={row['dense']} s={row['sparse']} "
                    f"rrf={row['rrf']}"
                )
                print(f"          section={row['section']!r}")

            if not hits:
                failures.append(f"Q{qi} {mode} 返回空结果")

            # score 必须等于该通路的原生分
            top = hits[0]
            expected_key = {"dense": "dense_score", "sparse": "sparse_score",
                            "hybrid": "rrf_score"}[mode]
            if top.get("score") != top.get(expected_key):
                failures.append(
                    f"Q{qi} {mode} score({top.get('score')}) != "
                    f"{expected_key}({top.get(expected_key)})"
                )

            if mode == "sparse":
                known = r.get("sparse_known_terms") or []
                unknown = r.get("sparse_unknown_terms") or []
                print(f"      sparse known={len(known)} unknown={len(unknown)}")

        # dense / sparse / hybrid 的 chunk_id 一致性。
        # 注意: RRF 是在 candidate_k 深度上融合的, 所以 hybrid 结果
        # 可以来自单通道 top-k 之外 (例如 dense rank 11 + sparse rank 9)。
        # 正确断言应是: hybrid 的每个 id 至少出现在某个通道的候选池里。
        d, s, h = (cid_set(results.get(m, {}).get("hits") or []) for m in ("dense", "sparse", "hybrid"))
        print(f"  id 交叉: dense={len(d)} sparse={len(s)} hybrid={len(h)} dense∩sparse={len(d & s)}")

        pool: set = set()
        for mode in ("dense", "sparse"):
            try:
                wide = post(
                    "/api/search",
                    {"query": query, "mode": mode, "top_k": 20,
                     "candidate_k": 20, "rerank": False},
                )
                pool |= cid_set(wide.get("hits") or [])
            except urllib.error.HTTPError:
                pass
        orphan = h - pool
        if h and orphan:
            failures.append(f"Q{qi} hybrid 出现不在任何通道候选池内的 chunk_id: {sorted(orphan)}")
        elif h:
            print(f"  hybrid ⊆ 单通道候选池: True (池大小 {len(pool)})")

        # rerank
        try:
            rr = post(
                "/api/search",
                {
                    "query": query,
                    "mode": "hybrid",
                    "top_k": 5,
                    "candidate_k": 20,
                    "rerank": True,
                    "rerank_candidate_k": 20,
                },
            )
            hits = rr.get("hits") or []
            print(f"  rerank : applied={rr.get('rerank_applied')} model={rr.get('rerank_model')} "
                  f"err={rr.get('rerank_error')} elapsed={rr.get('elapsed_ms')}ms")
            for row in brief(hits):
                print(
                    f"      #{row['rank']} {row['chunk_id']} score={row['score']} "
                    f"rerank={row['rerank']} (pre={row['rrf']})"
                )
            if not rr.get("rerank_applied"):
                failures.append(f"Q{qi} rerank 未生效: {rr.get('rerank_error')}")
            if not hits:
                failures.append(f"Q{qi} rerank 返回空结果")
            else:
                # score 必须等于 rerank_score
                if hits[0].get("score") != hits[0].get("rerank_score"):
                    failures.append(
                        f"Q{qi} rerank 后 score({hits[0].get('score')}) != "
                        f"rerank_score({hits[0].get('rerank_score')})"
                    )
                # rerank 候选集必须包含未 rerank 的 top1, 否则召回被破坏
                h0 = (results.get("hybrid", {}).get("hits") or [])
                if h0 and cid_set([h0[0]]) - cid_set(hits):
                    failures.append(
                        f"Q{qi} rerank 后丢失了 hybrid top1 ({h0[0].get('chunk_id')})"
                    )
        except urllib.error.HTTPError as exc:
            failures.append(f"Q{qi} rerank HTTP {exc.code}: {exc.read()[:200]!r}")
            print(f"  rerank : HTTP {exc.code}")

    # generation 单独跑一条，避免过多 LLM 调用
    print()
    print("=" * 78)
    print("[Generation] 上下文构造 + Qwen 回答 + citations")
    print("=" * 78)
    q = QUERIES[0]
    try:
        ans = post(
            "/api/answer",
            {"query": q, "mode": "hybrid", "top_k": 5, "candidate_k": 20, "rerank": True},
        )
        text = ans.get("answer") or ""
        cits = ans.get("citations") or []
        print(f"  query      : {q}")
        print(f"  elapsed    : {ans.get('elapsed_ms')}ms")
        print(f"  answer len : {len(text)}")
        print(f"  citations  : {len(cits)}")
        print("  --- answer (前 600 字) ---")
        print("  " + text[:600].replace("\n", "\n  "))
        print("  --- citations ---")
        for c in cits[:6]:
            print(f"    [{c.get('index') or c.get('marker')}] chunk={c.get('chunk_id')} "
                  f"artifact={c.get('artifact_id')} section={(c.get('section') or '')[:44]!r}")
        if not text.strip():
            failures.append("generation 返回空 answer")
        if not cits:
            failures.append("generation 未返回 citations")
    except urllib.error.HTTPError as exc:
        failures.append(f"generation HTTP {exc.code}: {exc.read()[:300]!r}")
        print(f"  HTTP {exc.code}")

    print()
    print("=" * 78)
    if failures:
        print(f"结果: FAIL ({len(failures)} 项)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("结果: PASS — dense / sparse / hybrid / rerank / generation 全部可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
