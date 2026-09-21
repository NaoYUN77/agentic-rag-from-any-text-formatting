r"""展示一篇文章经过完整链路后的真实情况。

输出一份 HTML 报告, 逐层展示:
    1. 源与路由    (URL / MIME / parser / 决策理由)
    2. Block 层    (类型分布 + 每个块的真实字段)
    3. 章节树      (heading 栈推出来的 section_path)
    4. Chunk 层    (chunk 正文 + 引用定位字段)
    5. 索引 Payload (真正写进 Qdrant 的那一份)
    6. 引用展示    (generation 侧拼出来的来源串)

用法:
    cd pipeline
    .\.venv_rag\Scripts\python.exe show_url_doc.py --doc anthropic_contextual_retrieval
    .\.venv_rag\Scripts\python.exe show_url_doc.py --list
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html as html_mod
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.models import SourcePayload  # noqa: E402
from ingest.parsers.html import parse_html  # noqa: E402
from ingest.router import FormatRouter  # noqa: E402
from ingest.chunker import BlockAwareHierarchicalChunkBuilder  # noqa: E402
from ingest.qdrant_indexer import _chunk_payload  # noqa: E402

SIDECAR = Path("experiments/_sources")


def esc(s: Any) -> str:
    return html_mod.escape(str(s)) if s is not None else ""


def load(doc: str):
    p = SIDECAR / f"{doc}.html"
    raw = p.read_text(encoding="utf-8", errors="replace")
    data = raw.encode("utf-8")
    uri = (json.loads((Path("experiments") / doc / "artifact.json")
                      .read_text(encoding="utf-8")).get("source_uri") or "")
    src = SourcePayload(
        source_type="url", uri=uri, final_uri=uri,
        mime_type="text/html", content_type="text/html; charset=utf-8",
        data=data, encoding="utf-8",
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )
    return src, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="anthropic_contextual_retrieval")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.list:
        for f in sorted(SIDECAR.glob("*.html")):
            print(f"{f.stat().st_size:>9}  {f.stem}")
        return 0

    doc = args.doc
    src, raw = load(doc)

    # --- 1. 路由 ---
    route = FormatRouter().route(src)
    payload = src
    artifact = parse_html(payload)

    # --- 2. 切块 ---
    builder = BlockAwareHierarchicalChunkBuilder(chunk_tokens=800)
    parents, chunks = builder.build(artifact)

    art_dict = artifact.to_dict()
    payloads = [_chunk_payload(c.to_dict(), art_dict, i) for i, c in enumerate(chunks)]

    types = collections.Counter(b.type for b in artifact.blocks)

    out = Path(args.out) if args.out else Path(f"docs/url_doc_{doc}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    P: List[str] = []
    P.append("""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>URL 文章处理实况</title>
<style>
  :root{
    --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3;
    --card:#fafafa; --accent:#0b6bcb; --ok:#1a7f37; --warn:#9a6700;
    --code:#f4f4f5;
  }
  *{box-sizing:border-box}
  body{margin:0;padding:32px 20px;background:var(--bg);color:var(--fg);
       font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
       "Microsoft YaHei",sans-serif;}
  .wrap{max-width:1080px;margin:0 auto}
  h1{font-size:24px;margin:0 0 6px}
  h2{font-size:17px;margin:34px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--line)}
  h3{font-size:14px;margin:20px 0 8px;color:var(--muted);
     text-transform:uppercase;letter-spacing:.06em}
  .sub{color:var(--muted);margin:0 0 22px;font-size:13px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:8px;
        padding:14px 16px;margin:12px 0}
  table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
  th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
  th{background:#f0f0f1;font-weight:600;white-space:nowrap}
  code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
  pre{background:var(--code);border:1px solid var(--line);border-radius:6px;
      padding:10px 12px;overflow-x:auto;white-space:pre-wrap;word-break:break-word}
  .kv{display:grid;grid-template-columns:190px 1fr;gap:4px 14px;font-size:13px}
  .kv dt{color:var(--muted)}
  .kv dd{margin:0;word-break:break-word}
  .pill{display:inline-block;padding:1px 8px;border-radius:11px;font-size:11px;
        border:1px solid var(--line);background:#fff;margin-right:5px}
  .ok{color:var(--ok);font-weight:600}
  .zero{color:var(--warn);font-weight:600}
  .blk{border:1px solid var(--line);border-radius:6px;padding:9px 11px;margin:7px 0;
       background:#fff}
  .blk .h{font-size:12px;color:var(--muted);margin-bottom:5px;
          display:flex;gap:9px;flex-wrap:wrap;align-items:center}
  .blk .t{font-size:13px;white-space:pre-wrap;word-break:break-word}
  .chunk{border-left:3px solid var(--accent);padding-left:12px;margin:16px 0}
  .tag{font-family:ui-monospace,monospace;font-size:11px;background:#eef4fb;
       color:var(--accent);padding:1px 7px;border-radius:4px}
  .note{background:#fffbe8;border:1px solid #f0e0a0;border-radius:6px;
        padding:11px 14px;margin:12px 0;font-size:13px}
  .flow{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted);
        margin:8px 0;padding:10px;background:var(--code);border-radius:6px}
</style></head><body><div class="wrap">""")

    P.append(f"<h1>一篇 URL 文章经过完整链路后的真实情况</h1>")
    P.append(f'<p class="sub">{esc(artifact.source_uri)}</p>')

    # --- 1. 路由与源 ---
    P.append("<h2>① 源与路由</h2>")
    P.append('<div class="card"><dl class="kv">')
    for k, v in [
        ("source_uri", artifact.source_uri),
        ("source_type", artifact.source_type),
        ("mime_type", artifact.mime_type),
        ("HTTP 原始字节", f"{src.size_bytes:,} B"),
        ("sha256", (src.sha256 or "")[:32] + "…"),
        ("路由 parser", f"<b>{esc(route.parser)}</b>"),
        ("路由理由", route.reason),
        ("路由 confidence", route.confidence),
        ("路由 mode", route.mode or "-"),
        ("实际 parser", f"<b>{esc(artifact.parser)}</b> / {esc(artifact.parser_version)}"),
        ("title", artifact.title),
        ("language", artifact.language),
        ("published_at", artifact.published_at or "-"),
    ]:
        P.append(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>")
    P.append("</dl></div>")

    # --- 2. Block ---
    P.append("<h2>② Block 层（DOM 直产，不经 Markdown）</h2>")
    P.append('<div class="card">')
    P.append(f"<p>共 <b>{len(artifact.blocks)}</b> 个 block：</p><p>")
    for t, c in types.most_common():
        P.append(f'<span class="pill">{esc(t)} × {c}</span>')
    P.append("</p>")
    n_url = sum(1 for b in artifact.blocks if b.url)
    n_lang = sum(1 for b in artifact.blocks if b.type == "code" and b.code_language)
    n_code = types.get("code", 0)
    n_cap = sum(1 for b in artifact.blocks if b.caption)
    n_lnk = sum(1 for b in artifact.blocks if (b.metadata or {}).get("links"))
    P.append("<h3>新增字段覆盖</h3>")
    P.append('<table><tr><th>字段</th><th>覆盖</th><th>说明</th></tr>')
    P.append(f'<tr><td><code>url</code></td><td class="{"ok" if n_url else "zero"}">'
             f'{n_url} / {len(artifact.blocks)}</td><td>块级来源 URL，引用时告知读者"这段来自哪一页"</td></tr>')
    P.append(f'<tr><td><code>code_language</code></td><td class="{"ok" if n_lang else "zero"}">'
             f'{n_lang} / {n_code}</td><td>代码块语言，从 class 识别</td></tr>')
    P.append(f'<tr><td><code>caption</code></td><td class="{"ok" if n_cap else "zero"}">{n_cap}</td>'
             f'<td>图片 caption（figure/figcaption）</td></tr>')
    P.append(f'<tr><td><code>metadata.links</code></td><td class="{"ok" if n_lnk else "zero"}">{n_lnk}</td>'
             f'<td>块内锚文本 + href</td></tr>')
    P.append("</table></div>")

    P.append("<h3>全部 block（真实字段）</h3>")
    for b in artifact.blocks:
        m = b.metadata or {}
        bits = [f'<span class="tag">{esc(b.block_id)}</span>', f'<b>{esc(b.type)}</b>']
        if b.type == "heading":
            bits.append(f'lv{m.get("level")}')
        if b.code_language:
            bits.append(f'lang={esc(b.code_language)}')
        if m.get("row_count"):
            bits.append(f'rows={m["row_count"]}')
        if b.caption:
            bits.append(f'caption={esc(b.caption[:44])}')
        if m.get("links"):
            bits.append(f'links={len(m["links"])}')
        sp = " › ".join(b.section_path) if b.section_path else "（根）"
        bits.append(f'<span style="color:#888">section: {esc(sp)}</span>')
        txt = b.text if len(b.text) <= 260 else b.text[:260] + " …"
        P.append(f'<div class="blk"><div class="h">{" ".join(bits)}</div>'
                 f'<div class="t">{esc(txt)}</div></div>')

    # --- 3. 章节树 ---
    P.append("<h2>③ 章节树（由 heading 栈推出）</h2>")
    tree: Dict[str, Any] = {}
    for b in artifact.blocks:
        if b.type != "heading":
            continue
        node = tree
        for part in (b.section_path or []):
            node = node.setdefault(part, {})
        node.setdefault(b.text, {})
    lines: List[str] = []

    def walk(node: Dict[str, Any], d: int) -> None:
        for k, v in node.items():
            lines.append("  " * d + "├─ " + k)
            walk(v, d + 1)

    walk(tree, 0)
    P.append('<div class="card"><pre>' + esc("\n".join(lines)) + "</pre></div>")
    P.append('<div class="note"><b>注意</b>：这篇文章正文的第一个 heading 是 '
             '<code>lv3</code>，因为 readability 把网站导航里的 <code>h1/h2</code> 剥掉了。'
             '所以整棵树的根是那个 lv3 小节，而不是文章主标题 —— 这就是 '
             '<code>issues/11</code> 记录的问题。<b>它在旧的 Markdown 链路上完全一样</b>，'
             '本次改动没有引入、也没有修复它。</div>')

    # --- 4. Chunk ---
    P.append("<h2>④ Chunk 层</h2>")
    P.append(f'<div class="card"><dl class="kv">'
             f'<dt>parents</dt><dd>{len(parents)}</dd>'
             f'<dt>chunks</dt><dd>{len(chunks)}</dd>'
             f'<dt>chunk tokens 合计</dt><dd>{sum(c.token_count for c in chunks):,}</dd>'
             f'<dt>参数</dt><dd>chunk_tokens=800（固定大小, 无 overlap）</dd>'
             f"</dl></div>")
    P.append('<div class="flow">scope 分组 → '
             + " ｜ ".join(esc(p.scope_title[:38]) for p in parents) + "</div>")
    for i, c in enumerate(chunks):
        pl = payloads[i]
        fr = pl.get("fragments") or []
        bid = ", ".join(str(f.get("block_id")) for f in fr)
        P.append('<div class="chunk">')
        P.append(f'<div class="h"><span class="tag">{esc(c.chunk_id)}</span> '
                 f'tokens={c.token_count} · chars={c.char_count} · '
                 f'parent={esc(c.parent_id)} · section_block={esc(pl.get("section_block_id"))}</div>')
        P.append(f'<div style="font-size:12px;color:#666;margin:4px 0">'
                 f'section_path: {esc(" › ".join(pl.get("section_path") or []) or "（空）")}<br>'
                 f'url: {esc(pl.get("url"))}<br>'
                 f'fragments({len(fr)}): {esc(bid[:180])}</div>')
        t = c.full_text if len(c.full_text) <= 400 else c.full_text[:400] + " …"
        P.append(f'<pre>{esc(t)}</pre>')
        P.append("</div>")

    # --- 5. Payload ---
    P.append("<h2>⑤ 索引 Payload（真正写进 Qdrant 的那一份）</h2>")
    P.append('<div class="note">payload 只放"检索与引用需要"的字段。'
             '刻意不放 <code>dense_text</code>/<code>sparse_text</code>（只用于建向量）、'
             '<code>fragments[].text</code>（正文的再一份拷贝）、'
             '<code>quality</code>/<code>index_decision</code>（建索引的决策痕迹）。</div>')
    total_payload = sum(len(json.dumps(p, ensure_ascii=False).encode("utf-8")) for p in payloads)
    P.append(f'<div class="card"><dl class="kv">'
             f'<dt>payload 总字节</dt><dd>{total_payload:,} B</dd>'
             f'<dt>平均每 chunk</dt><dd>{total_payload // len(payloads):,} B</dd>'
             f'<dt>字段数</dt><dd>{len(payloads[0])}</dd>'
             f"</dl></div>")
    P.append("<h3>字段清单</h3><div class='card'><p>")
    for k in payloads[0]:
        P.append(f'<span class="pill">{esc(k)}</span>')
    P.append("</p></div>")
    P.append("<h3>第 2 个 chunk 的完整 payload</h3>")
    P.append("<pre>" + esc(json.dumps(payloads[1], ensure_ascii=False, indent=2)[:2600]) + "</pre>")

    # --- 6. 引用 ---
    P.append("<h2>⑥ 引用展示（generation 侧）</h2>")
    P.append("<p>检索命中后，<code>build_context()</code> 用 payload 里的字段拼出给 LLM 的来源串：</p>")
    sample = payloads[0]
    loc = " · ".join(x for x in [
        f'第 {sample["page_start"]} 页' if sample.get("page_start") else "",
        f'section: {" › ".join(sample.get("section_path") or [])}' if sample.get("section_path") else "",
    ] if x)
    P.append("<pre>" + esc(
        f'[C1] {sample.get("title")}\n'
        f'     来源: {sample.get("url")}\n'
        f'     {loc}\n'
        f'     chunk: {sample.get("chunk_id")}'
    ) + "</pre>")
    P.append('<div class="note"><code>url</code> 是本次新增的。'
             'HTML 没有"页码"概念，所以引用定位靠的是 <b>URL + 章节路径</b>；'
             'PDF 则靠 <b>页码 + bbox</b>。两者都从 block 一路带到 payload。</div>')

    P.append("<h2>⑦ 数据规模汇总</h2>")
    P.append('<table><tr><th>层</th><th>数量</th></tr>')
    for k, v in [("HTML 原始字节", f"{src.size_bytes:,}"),
                 ("Block", len(artifact.blocks)),
                 ("heading", types.get("heading", 0)),
                 ("ParentNode", len(parents)),
                 ("IndexReadyChunk", len(chunks)),
                 ("payload 总字节", f"{total_payload:,}")]:
        P.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")
    P.append("</table>")

    P.append("</div></body></html>")

    out.write_text("".join(P), encoding="utf-8")
    print(f"written: {out}")

    # 同时把机读数据落盘
    dump = out.with_suffix(".json")
    dump.write_text(json.dumps({
        "doc": doc,
        "route": route.__dict__,
        "artifact": art_dict,
        "chunks": [c.to_dict() for c in chunks],
        "payloads": payloads,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
