r"""预览 LlamaIndex 固定长度 + overlap + 语义边界切分。"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from semantic_chunker_demo import build_embeddings
from llamaindex_fixed_semantic_splitter import (
    FixedSemanticNodeParser,
    build_section_documents,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=400)
    parser.add_argument("--window-size", type=int, default=400)
    args = parser.parse_args()

    path = Path(args.md)
    md = path.read_text(encoding="utf-8", errors="replace")
    source_name = args.source_name or path.name
    docs = build_section_documents(md, source_name)
    print(f"documents={len(docs)}")

    parser_obj = FixedSemanticNodeParser(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        window_size=args.window_size,
        embed_model=build_embeddings("qwen"),
    )
    nodes = parser_obj.get_nodes_from_documents(docs)
    token_counts = [int(n.metadata["token_count"]) for n in nodes]
    char_counts = [int(n.metadata["char_count"]) for n in nodes]

    print(f"nodes={len(nodes)}")
    print(
        "tokens min/avg/max=%d / %.1f / %d"
        % (min(token_counts), statistics.mean(token_counts), max(token_counts))
    )
    print(
        "chars min/avg/max=%d / %.1f / %d"
        % (min(char_counts), statistics.mean(char_counts), max(char_counts))
    )

    for i, node in enumerate(nodes[:5], 1):
        print("=" * 80)
        print(f"node {i} id={node.node_id}")
        print(f"metadata={node.metadata}")
        print(node.text[:500])


if __name__ == "__main__":
    main()
