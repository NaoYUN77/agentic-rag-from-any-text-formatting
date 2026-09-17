r"""把 Phase 0 JSON 导出结果写入 Qdrant dense + sparse 索引。

输入:
    export-dir/
      artifact.json
      chunks.jsonl
      parents.json          # 可选, 只记录数量

输出:
    Qdrant dense collection
    Qdrant sparse collection
    index_artifacts/phase0/
      vocab.json
      bm25_stats.json
      chunks.jsonl
      summary.json
      index_manifest.json

用法:
    cd pipeline

    .\.venv_rag\Scripts\python.exe -m ingest.qdrant_indexer `
        --export-dir experiments/router_demo `
        --qdrant-path qdrant_data `
        --dense-collection phase0_dense `
        --sparse-collection phase0_sparse `
        --artifact-dir index_artifacts/phase0 `
        --rebuild
"""

from __future__ import annotations

import argparse
import re
import gc
import time
import shutil
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sparse_pipeline_demo import (
    Chunk,
    SparseBM25Index,
    save_artifacts,
    store_sparse_collection,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_QDRANT_PATH = "qdrant_data"
DEFAULT_DENSE_COLLECTION = "phase0_dense"
DEFAULT_SPARSE_COLLECTION = "phase0_sparse"
DEFAULT_ARTIFACT_DIR = "index_artifacts/phase0"
DEFAULT_VECTOR_SIZE = 1024


@dataclass
class ExportBundle:
    export_dir: Path
    artifact: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    parent_count: int = 0


@dataclass
class IndexPlan:
    dense: List[Tuple[int, str, Dict[str, Any]]] = field(default_factory=list)
    sparse: List[Tuple[int, str, Dict[str, Any]]] = field(default_factory=list)
    chunks_loaded: int = 0
    parent_count: int = 0
    source_dirs: List[str] = field(default_factory=list)

    @property
    def dense_count(self) -> int:
        return len(self.dense)

    @property
    def sparse_count(self) -> int:
        return len(self.sparse)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_chunks(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 不是 JSON object")
            rows.append(value)
        return rows

    value = _read_json(path)
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, dict) and isinstance(value.get("chunks"), list):
        return [dict(item) for item in value["chunks"]]
    raise ValueError(f"无法从 {path} 读取 chunk 列表")


def load_export_bundle(export_dir: str | Path) -> ExportBundle:
    root = Path(export_dir).resolve()
    artifact_path = root / "artifact.json"
    chunks_jsonl = root / "chunks.jsonl"
    chunks_json = root / "chunks.json"

    if not artifact_path.exists():
        raise FileNotFoundError(f"缺少 artifact.json: {artifact_path}")
    if chunks_jsonl.exists():
        chunks_path = chunks_jsonl
    elif chunks_json.exists():
        chunks_path = chunks_json
    else:
        raise FileNotFoundError(f"缺少 chunks.jsonl 或 chunks.json: {root}")

    artifact = _read_json(artifact_path)
    chunks = _read_chunks(chunks_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"artifact.json 不是 JSON object: {artifact_path}")
    if not chunks:
        raise ValueError(f"导出目录没有 chunk: {root}")

    parent_count = 0
    parents_path = root / "parents.json"
    if parents_path.exists():
        parents = _read_json(parents_path)
        if isinstance(parents, list):
            parent_count = len(parents)

    return ExportBundle(
        export_dir=root,
        artifact=artifact,
        chunks=chunks,
        parent_count=parent_count,
    )


def point_id_for(chunk_id: str) -> int:
    """把业务 chunk_id 映射成稳定、合法的 Qdrant point id。"""
    digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _source_file_name(artifact: Mapping[str, Any]) -> str:
    metadata = artifact.get("metadata") or {}
    if isinstance(metadata, Mapping):
        for key in ("file_name", "source_name"):
            value = metadata.get(key)
            if value:
                return str(value)

    title = artifact.get("title")
    if title:
        return str(title)

    source_uri = str(artifact.get("source_uri") or "")
    if source_uri:
        parsed = urlparse(source_uri)
        name = Path(parsed.path).name
        if name:
            return name
        return source_uri
    return "unknown"


def _section_path(chunk: Mapping[str, Any]) -> List[str]:
    value = chunk.get("section_path") or []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _chunk_payload(
    chunk: Mapping[str, Any],
    artifact: Mapping[str, Any],
    chunk_index: int,
) -> Dict[str, Any]:
    section_path = _section_path(chunk)
    section = " > ".join(section_path)
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    full_text = str(chunk.get("full_text") or "").strip()
    decision = chunk.get("index_decision") or {}

    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "artifact_id": str(chunk.get("artifact_id") or artifact.get("artifact_id") or ""),
        "parent_id": chunk.get("parent_id"),
        "section_block_id": chunk.get("section_block_id"),
        "section": section,
        "section_path": section_path,
        "file_name": _source_file_name(artifact),
        "source_uri": artifact.get("source_uri"),
        "source_type": artifact.get("source_type"),
        "mime_type": artifact.get("mime_type"),
        "parser": artifact.get("parser"),
        "title": artifact.get("title"),
        "page": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_index": chunk_index,
        "token_count": int(chunk.get("token_count") or 0),
        "char_count": int(chunk.get("char_count") or len(full_text)),
        "text": full_text,
        "dense_text": str(chunk.get("dense_text") or full_text).strip(),
        "sparse_text": str(chunk.get("sparse_text") or "").strip(),
        "fragments": chunk.get("fragments") or [],
        "quality": chunk.get("quality"),
        "index_decision": decision,
        "overlap_from_previous": int(chunk.get("overlap_from_previous") or 0),
    }


def build_index_plan(bundles: Iterable[ExportBundle]) -> IndexPlan:
    plan = IndexPlan()
    seen_dense: Dict[str, int] = {}
    seen_sparse: Dict[str, int] = {}

    for bundle in bundles:
        plan.source_dirs.append(str(bundle.export_dir))
        plan.parent_count += bundle.parent_count
        for chunk_index, chunk in enumerate(bundle.chunks):
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"{bundle.export_dir}: 存在缺少 chunk_id 的记录")

            point_id = point_id_for(chunk_id)
            payload = _chunk_payload(chunk, bundle.artifact, chunk_index)
            decision = payload.get("index_decision") or {}
            dense_enabled = bool(decision.get("dense_index", True))
            sparse_enabled = bool(decision.get("sparse_index", True))
            dense_text = str(payload.get("dense_text") or payload["text"]).strip()
            sparse_text = str(payload.get("sparse_text") or "").strip()

            if dense_enabled and dense_text:
                previous = seen_dense.get(chunk_id)
                if previous is not None and previous != point_id:
                    raise ValueError(f"chunk_id 对应多个 point_id: {chunk_id}")
                seen_dense[chunk_id] = point_id
                plan.dense.append((point_id, dense_text, payload))

            if sparse_enabled and sparse_text:
                previous = seen_sparse.get(chunk_id)
                if previous is not None and previous != point_id:
                    raise ValueError(f"chunk_id 对应多个 point_id: {chunk_id}")
                seen_sparse[chunk_id] = point_id
                plan.sparse.append((point_id, sparse_text, payload))

            plan.chunks_loaded += 1

    if not plan.dense:
        raise ValueError("没有可写入 dense collection 的 chunk")
    if not plan.sparse:
        raise ValueError("没有可写入 sparse collection 的 chunk")
    return plan


def _ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    rebuild: bool,
) -> None:
    exists = client.collection_exists(collection)
    if exists and rebuild:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _write_dense(
    plan: IndexPlan,
    embeddings: Any,
    qdrant_path: Path,
    collection: str,
    vector_size: int,
    rebuild: bool,
) -> int:
    client = QdrantClient(path=str(qdrant_path))
    try:
        _ensure_collection(client, collection, vector_size, rebuild)
        texts = [text for _, text, _ in plan.dense]
        vectors = embeddings.embed_documents(texts)
        if len(vectors) != len(plan.dense):
            raise RuntimeError(
                f"embedding 数量不匹配: got {len(vectors)}, expected {len(plan.dense)}"
            )

        points: List[PointStruct] = []
        for (point_id, _, payload), vector in zip(plan.dense, vectors):
            if len(vector) != vector_size:
                raise RuntimeError(
                    f"向量维度不匹配: chunk_id={payload.get('chunk_id')} "
                    f"got {len(vector)}, expected {vector_size}"
                )
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        client.upsert(collection_name=collection, points=points, wait=True)
        return int(client.count(collection_name=collection, exact=True).count)
    finally:
        client.close()


def _sparse_chunks(plan: IndexPlan) -> List[Chunk]:
    chunks: List[Chunk] = []
    for order, (point_id, index_text, payload) in enumerate(plan.sparse):
        metadata = dict(payload)
        section = str(metadata.get("section") or "文档开头")
        source = str(metadata.get("file_name") or "unknown")
        chunks.append(Chunk(
            id=str(metadata.get("chunk_id") or point_id),
            source=source,
            section=section,
            order=order,
            text=str(metadata.get("text") or ""),
            index_text=index_text,
            metadata=metadata,
            point_id=point_id,
        ))
    return chunks


def _collection_storage_dir(qdrant_path: Path, collection: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", collection):
        raise ValueError(f"unsafe collection name: {collection!r}")
    return qdrant_path / "collection" / collection


def _delete_collections(qdrant_path: Path, collections: Sequence[str]) -> None:
    """Delete target collections before a rebuild.

    Qdrant local on Windows can leave the collection SQLite directory behind
    after delete_collection(). Removing the validated collection directory is
    required so a rebuild cannot mix points from two artifact generations.
    """
    client = QdrantClient(path=str(qdrant_path))
    try:
        for collection in collections:
            if client.collection_exists(collection):
                client.delete_collection(collection)
    finally:
        client.close()
    del client
    gc.collect()

    for collection in collections:
        storage_dir = _collection_storage_dir(qdrant_path, collection)
        if not storage_dir.exists():
            continue
        for attempt in range(5):
            try:
                shutil.rmtree(storage_dir)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                gc.collect()
                time.sleep(0.1)


def write_index(
    plan: IndexPlan,
    embeddings: Any,
    qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
    dense_collection: str = DEFAULT_DENSE_COLLECTION,
    sparse_collection: str = DEFAULT_SPARSE_COLLECTION,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    rebuild: bool = False,
    source: str = "phase0_json",
) -> Dict[str, Any]:
    qpath = Path(qdrant_path).resolve()
    adir = Path(artifact_dir).resolve()
    qpath.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)

    if rebuild:
        _delete_collections(qpath, (dense_collection, sparse_collection))

    dense_points = _write_dense(
        plan=plan,
        embeddings=embeddings,
        qdrant_path=qpath,
        collection=dense_collection,
        vector_size=vector_size,
        rebuild=False,
    )

    sparse_chunks = _sparse_chunks(plan)
    sparse_index = SparseBM25Index(sparse_chunks)
    save_artifacts(
        out_dir=adir,
        source=source,
        chunks=sparse_chunks,
        index=sparse_index,
        query_count=0,
    )
    sparse_points = store_sparse_collection(
        qdrant_path=str(qpath),
        collection=sparse_collection,
        chunks=sparse_chunks,
        index=sparse_index,
        rebuild=False,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "source_dirs": plan.source_dirs,
        "qdrant_path": str(qpath),
        "dense_collection": dense_collection,
        "sparse_collection": sparse_collection,
        "artifact_dir": str(adir),
        "vector_size": vector_size,
        "chunks_loaded": plan.chunks_loaded,
        "parent_count": plan.parent_count,
        "dense_points": dense_points,
        "sparse_points": sparse_points,
        "sparse_vocabulary_size": sparse_index.vocab_size,
        "bm25": {"k1": sparse_index.k1, "b": sparse_index.b},
    }
    (adir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 Phase 0 JSON 导出写入 Qdrant")
    parser.add_argument("--export-dir", action="append", required=True,
                        help="format_router_demo.py 的输出目录, 可重复")
    parser.add_argument("--qdrant-path", default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--dense-collection", default=DEFAULT_DENSE_COLLECTION)
    parser.add_argument("--sparse-collection", default=DEFAULT_SPARSE_COLLECTION)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--embedding-backend", default="qwen",
                        choices=["qwen", "openai", "local"])
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE)
    parser.add_argument("--rebuild", action="store_true",
                        help="写入前删除目标 dense/sparse collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="只生成索引计划, 不调用 embedding 或 Qdrant")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    bundles = [load_export_bundle(path) for path in args.export_dir]
    plan = build_index_plan(bundles)

    print(f"export dirs : {len(bundles)}")
    print(f"chunks      : {plan.chunks_loaded}")
    print(f"dense       : {plan.dense_count}")
    print(f"sparse      : {plan.sparse_count}")
    print(f"parents     : {plan.parent_count}")

    if args.dry_run:
        return

    from semantic_chunker_demo import build_embeddings

    embeddings = build_embeddings(args.embedding_backend)
    manifest = write_index(
        plan=plan,
        embeddings=embeddings,
        qdrant_path=args.qdrant_path,
        dense_collection=args.dense_collection,
        sparse_collection=args.sparse_collection,
        artifact_dir=args.artifact_dir,
        vector_size=args.vector_size,
        rebuild=args.rebuild,
        source="phase0_json",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
