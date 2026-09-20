r"""格式路由 CLI demo。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ingest.pipeline import IngestPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="格式路由与统一 Block 输出")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--url")
    group.add_argument("--file")
    ap.add_argument("--out-dir")
    ap.add_argument("--allow-private", action="store_true")
    ap.add_argument("--pdf-pages", default=None)
    ap.add_argument("--pdf-language", default="ch")
    ap.add_argument("--no-chunk", action="store_true")
    ap.add_argument("--chunker", choices=["block", "sentence"], default="block",
                    help="切块策略: block=800token粗粒度; sentence=单句+metadata窗口")
    ap.add_argument("--window-sentences", type=int, default=3,
                    help="sentence 策略下, metadata 窗口的前后句数")
    ap.add_argument("--chunk-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=400)
    ap.add_argument("--window-tokens", type=int, default=400,
                    help="预留参数, 当前不参与计算 (见 issues/09)")
    args = ap.parse_args()

    value = args.url or args.file
    pipeline = IngestPipeline()
    result = pipeline.ingest(
        value,
        allow_private=args.allow_private,
        parser_kwargs={
            "pages": args.pdf_pages,
            "language": args.pdf_language,
        },
        build_chunks=not args.no_chunk,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        window_tokens=args.window_tokens,
        chunker=args.chunker,
        window_sentences=args.window_sentences,
    )
    artifact = result.artifact

    print("=" * 80)
    print("Format Router")
    print("=" * 80)
    print(f"source      : {artifact.source_uri}")
    print(f"mime        : {artifact.mime_type}")
    print(f"parser      : {artifact.parser}")
    print(f"route       : {result.route.parser} ({result.route.reason})")
    print(f"title       : {artifact.title}")
    print(f"language    : {artifact.language}")
    print(f"blocks      : {len(artifact.blocks)}")
    print(f"quality     : {artifact.quality.status} {artifact.quality.score}")
    print(f"index       : {artifact.index_decision.to_dict()}")
    print(f"attempts    : {result.attempts}")
    print(f"parents     : {len(result.parents)}")
    print(f"chunks      : {len(result.chunks)}")
    print()

    for block in artifact.blocks[:10]:
        preview = block.text[:120].replace("\n", " ")
        print(f"[{block.block_id}] {block.type}: {preview}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "artifact.json").write_text(
            json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "document.md").write_text(
            artifact.to_markdown(),
            encoding="utf-8",
        )
        if result.parents:
            (out_dir / "parents.json").write_text(
                json.dumps([p.to_dict() for p in result.parents], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if result.chunks:
            with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
                for chunk in result.chunks:
                    fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        print(f"\nwritten: {out_dir}")


if __name__ == "__main__":
    main()
