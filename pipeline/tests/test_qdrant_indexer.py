from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingest.qdrant_indexer import (
    build_index_plan,
    load_export_bundle,
    write_index,
)

from sparse_pipeline_demo import (
    Chunk,
    SparseBM25Index,
    store_sparse_collection,
)


class FixedEmbeddings:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for index, text in enumerate(texts):
            vector = [0.0] * self.dim
            vector[index % self.dim] = 1.0
            vector[-1] = float(len(text) % 7 or 1)
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _chunk(
    chunk_id: str,
    *,
    dense: bool,
    sparse: bool,
    text: str,
    sparse_text: str,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "artifact_id": "doc_test",
        "parent_id": "doc_test_p0001",
        "section_block_id": "b0002",
        "section_path": ["Proxy", "Keepalived"],
        "fragments": [
            {
                "block_id": "b0002",
                "start_char": 0,
                "end_char": len(text),
                "block_type": "text",
                "text": text,
            }
        ],
        "full_text": text,
        "sparse_text": sparse_text,
        "token_count": len(text.split()),
        "char_count": len(text),
        "page_start": 1,
        "page_end": 1,
        "overlap_from_previous": 0,
        "quality": {"score": 0.95, "status": "high", "flags": [], "metrics": {}},
        "index_decision": {
            "tier": "high" if sparse else "low",
            "dense_index": dense,
            "sparse_index": sparse,
            "sparse_weight": 1.0 if sparse else 0.0,
            "rerank_penalty": 0.0,
            "review_required": not sparse,
            "reason": "test",
        },
    }


def _write_export(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "1.0",
        "artifact_id": "doc_test",
        "source_uri": "https://example.com/keepalived",
        "source_type": "url",
        "mime_type": "text/html",
        "parser": "html_trafilatura",
        "title": "Keepalived Guide",
        "metadata": {"file_name": "keepalived.html"},
    }
    chunks = [
        _chunk(
            "doc_test_c0001",
            dense=True,
            sparse=True,
            text="Keepalived uses VRRP to move a virtual IP between routers.",
            sparse_text="keepalived VRRP virtual IP routers",
        ),
        _chunk(
            "doc_test_c0002",
            dense=True,
            sparse=False,
            text="Low quality OCR text that should only enter the dense collection.",
            sparse_text="noise ocr",
        ),
    ]
    (root / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (root / "parents.json").write_text(
        json.dumps([{"parent_id": "doc_test_p0001"}], ensure_ascii=False),
        encoding="utf-8",
    )


class QdrantIndexerTests(unittest.TestCase):
    def test_build_plan_respects_index_decision_and_shared_point_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "export"
            _write_export(export_dir)
            bundle = load_export_bundle(export_dir)
            plan = build_index_plan([bundle])

            self.assertEqual(bundle.parent_count, 1)
            self.assertEqual(plan.chunks_loaded, 2)
            self.assertEqual(plan.dense_count, 2)
            self.assertEqual(plan.sparse_count, 1)

            dense_ids = {
                payload["chunk_id"]: point_id
                for point_id, _, payload in plan.dense
            }
            sparse_ids = {
                payload["chunk_id"]: point_id
                for point_id, _, payload in plan.sparse
            }
            self.assertEqual(
                dense_ids["doc_test_c0001"],
                sparse_ids["doc_test_c0001"],
            )

    def test_write_index_creates_dense_sparse_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_dir = root / "export"
            qdrant_path = root / "qdrant_data"
            artifact_dir = root / "index_artifacts"
            _write_export(export_dir)

            bundle = load_export_bundle(export_dir)
            plan = build_index_plan([bundle])
            manifest = write_index(
                plan=plan,
                embeddings=FixedEmbeddings(dim=4),
                qdrant_path=qdrant_path,
                dense_collection="phase0_dense_test",
                sparse_collection="phase0_sparse_test",
                artifact_dir=artifact_dir,
                vector_size=4,
                rebuild=True,
            )

            self.assertEqual(manifest["dense_points"], 2)
            self.assertEqual(manifest["sparse_points"], 1)
            self.assertTrue((artifact_dir / "vocab.json").exists())
            self.assertTrue((artifact_dir / "bm25_stats.json").exists())
            self.assertTrue((artifact_dir / "index_manifest.json").exists())

            point_id = next(
                point_id
                for point_id, _, payload in plan.dense
                if payload["chunk_id"] == "doc_test_c0001"
            )
            client = QdrantClient(path=str(qdrant_path))
            try:
                self.assertTrue(client.collection_exists("phase0_dense_test"))
                self.assertTrue(client.collection_exists("phase0_sparse_test"))
                dense = client.retrieve(
                    collection_name="phase0_dense_test",
                    ids=[point_id],
                    with_payload=True,
                )
                sparse = client.retrieve(
                    collection_name="phase0_sparse_test",
                    ids=[point_id],
                    with_payload=True,
                )
                self.assertEqual(dense[0].payload["chunk_id"], "doc_test_c0001")
                self.assertEqual(sparse[0].payload["chunk_id"], "doc_test_c0001")
                self.assertEqual(sparse[0].payload["section"], "Proxy > Keepalived")
            finally:
                client.close()

    def test_rebuild_removes_stale_collection_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_dir = root / "export"
            qdrant_path = root / "qdrant_data"
            artifact_dir = root / "index_artifacts"
            _write_export(export_dir)
            plan = build_index_plan([load_export_bundle(export_dir)])

            write_index(
                plan=plan,
                embeddings=FixedEmbeddings(dim=4),
                qdrant_path=qdrant_path,
                dense_collection="phase0_dense_test",
                sparse_collection="phase0_sparse_test",
                artifact_dir=artifact_dir,
                vector_size=4,
                rebuild=True,
            )

            client = QdrantClient(path=str(qdrant_path))
            try:
                client.upsert(
                    collection_name="phase0_dense_test",
                    points=[PointStruct(
                        id=999,
                        vector=[1.0, 0.0, 0.0, 0.0],
                        payload={"chunk_id": "stale"},
                    )],
                    wait=True,
                )
                self.assertEqual(
                    client.get_collection("phase0_dense_test").points_count,
                    plan.dense_count + 1,
                )
            finally:
                client.close()

            manifest = write_index(
                plan=plan,
                embeddings=FixedEmbeddings(dim=4),
                qdrant_path=qdrant_path,
                dense_collection="phase0_dense_test",
                sparse_collection="phase0_sparse_test",
                artifact_dir=artifact_dir,
                vector_size=4,
                rebuild=True,
            )
            self.assertEqual(manifest["dense_points"], plan.dense_count)

            client = QdrantClient(path=str(qdrant_path))
            try:
                self.assertEqual(
                    client.retrieve(
                        collection_name="phase0_dense_test",
                        ids=[999],
                        with_payload=True,
                    ),
                    [],
                )
            finally:
                client.close()


    def test_sparse_weight_is_applied_to_document_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            qdrant_path = Path(temp_dir) / "qdrant"
            qdrant_path.mkdir(parents=True, exist_ok=True)
            chunk = Chunk(
                id="doc_c0001",
                source="test",
                section="Proxy",
                order=0,
                text="keepalived proxy",
                index_text="keepalived proxy",
                metadata={"index_decision": {"sparse_weight": 0.6}},
                point_id=1,
            )
            index = SparseBM25Index([chunk])
            raw_indices, raw_values = index.describe_sparse(0)
            store_sparse_collection(
                qdrant_path=str(qdrant_path),
                collection="sparse_weight_test",
                chunks=[chunk],
                index=index,
                rebuild=True,
            )

            client = QdrantClient(path=str(qdrant_path))
            try:
                point = client.retrieve(
                    collection_name="sparse_weight_test",
                    ids=[1],
                    with_vectors=True,
                )[0]
                sparse_vector = point.vector["sparse"]
                self.assertEqual(sparse_vector.indices, raw_indices)
                for raw_value, stored_value in zip(raw_values, sparse_vector.values):
                    self.assertAlmostEqual(stored_value, raw_value * 0.6)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
