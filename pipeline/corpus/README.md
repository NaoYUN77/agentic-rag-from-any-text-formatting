# 语料目录

公开仓库不包含第三方 PDF、手册和书籍转换出的 Markdown 全文。

本地运行时，请把你的合法语料放在本目录，例如:

```text
pipeline/corpus/
├── my_chinese_guide.pdf
├── my_english_paper.pdf
├── my_corpus.md
└── README.md
```

推荐流程:

```text
PDF
  -> MinerU / Docling
  -> Markdown
  -> LlamaIndex FixedSemanticNodeParser

HTML
  -> Trafilatura
  -> Markdown
  -> LlamaIndex FixedSemanticNodeParser
```

运行示例:

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."

.\.venv_rag\Scripts\python.exe rebuild_llamaindex_corpus.py `
    --chunk-tokens 800 `
    --overlap-tokens 400 `
    --window-tokens 400
```

注意:

```text
1. 不要把有版权风险的 PDF 或完整 Markdown 提交到 GitHub。
2. 不要把 API Key 写入仓库。
3. Qdrant 和 index_artifacts 是本地产物，不提交。
```
