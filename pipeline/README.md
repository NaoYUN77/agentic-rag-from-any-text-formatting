# Pipeline

`pipeline/` 保存可运行的 RAG 工程实现、测试、实验和本地数据。

## 1. 活跃实现

```text
ingest/                      多格式摄取 Phase 0
  models.py                  DocumentArtifact / DocumentBlock / QualityReport
  source.py                  本地文件与 URL 加载
  router.py                  扩展名 / MIME / magic bytes 路由
  parsers/                   HTML / Markdown / PDF / optional adapters
  cleaner.py                 Block 清理
  quality.py                 Raw Parse Quality Gate + Index Decision
  chunker.py                 Block-aware Hierarchical ChunkBuilder
  pipeline.py                IngestPipeline 编排

format_router_demo.py        Phase 0 摄取 CLI
tests/test_format_router.py  路由、Parser、质量门控、ChunkBuilder 测试
```

## 2. 当前在线链路

现有检索服务仍使用旧的 LlamaIndex chunks：

```text
固定长度与语义边界 NodeParser
-> Dense / Sparse
-> Qdrant
-> RRF
-> Rerank
-> Generation
```

Phase 0 已经能生成 `ParentNode + IndexReadyChunk`，但还没有接入 Qdrant 入库链路。两条链路不要混淆。

## 3. 运行环境

```powershell
cd pipeline
python -m venv .venv_rag
.\.venv_rag\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. 运行测试

```powershell
cd pipeline
.\.venv_rag\Scripts\python.exe -m unittest discover -s tests -v
```

## 5. 运行摄取 CLI

```powershell
.\.venv_rag\Scripts\python.exe format_router_demo.py `
    --file corpus/document.md `
    --out-dir experiments/router_demo
```

URL 示例：

```powershell
.\.venv_rag\Scripts\python.exe format_router_demo.py `
    --url https://example.com/article `
    --out-dir experiments/router_demo
```

CLI 会写出：

```text
artifact.json
document.md
parents.json
chunks.jsonl
```

## 6. 其他目录

```text
corpus/          本地语料，第三方版权内容不提交
eval/            评估集与 qrels
experiments/     原始实验输出和历史结论
backups/         本地备份，不提交
index_artifacts/ 本地索引产物，不提交
qdrant_data/     本地向量库，不提交
static/          FastAPI Web 页面
```

## 7. 数据与密钥

- API Key 只通过环境变量或本地未跟踪文件提供。
- 不提交第三方 PDF、完整 Markdown 语料、Qdrant 数据、索引产物和日志。
- 本地环境使用 `pipeline/.venv_rag/`，不应提交。
