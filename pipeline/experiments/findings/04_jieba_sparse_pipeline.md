# 实验 04: 独立 jieba + BM25 稀疏检索链路

## 1. 实验条件

```text
日期        2026-09-13
语料        pipeline/corpus/redhat_p1-20.md
解析入口    Markdown (PDF 入口复用 MinerU, 本实验未用真实 PDF)
清洗        pdf_loader.clean
实验分块    标题感知的段落打包, max_chars=800
分析器      jieba + 技术 token 保护 + 停用词过滤
索引        倒排索引 + BM25 稀疏权重
BM25        k1=1.2, b=0.75
脚本        pipeline/sparse_pipeline_demo.py
原始输出    pipeline/experiments/14_sparse_pipeline.txt
结构化产物  pipeline/experiments/14_sparse_pipeline_artifacts/
```

## 2. 实测数据

```text
清洗后 Markdown    17365 字
chunk 数           62
词表大小           1025
平均 token/chunk   69.081
非零 sparse 权重   2852
稀疏密度           2852 / (62 * 1025) = 0.044878
```

这里的“稀疏密度”定义为:

```text
doc_sparse 中非零权重的总数量 / (chunk 数 * 词表大小)
```

约 `4.49%`。它说明即使词表有 1025 个 term id，每个 chunk 也只保存其中少数非零词项。

## 3. 工程结论

### 3.1 词表不是高频词榜单

当前词表来自:

```text
所有 chunk
    -> jieba/技术 token 分析
    -> 唯一词项并集
    -> 排序
    -> term -> id
```

BM25 后续用 IDF 给高频词降权，但词表阶段并没有“先按高频筛出前 N 个词”。

### 3.2 sparse 并不调用 embedding 模型

当前链路完成一次索引只需要:

```text
CPU 分词
词项统计
BM25 权重计算
倒排表构建
```

没有调用百炼 embedding，也没有 Qdrant，适合先单独积累分析器和 BM25 的经验。

### 3.3 查询先经过 posting list 缩小候选

查询分析后:

```text
query terms -> posting lists -> 候选 chunk 并集 -> BM25 打分 -> top-k
```

不是对全部 chunk 逐一向量化比较。

### 3.4 技术 token 需要保护

分词器把英文配置项、路径、版本号和 IP 地址单独保护，例如:

```text
virtual_ipaddress
keepalived.conf
10.11.12.0/24
```

否则它们被拆成碎片后，中文文档里的精确匹配能力会明显下降。

## 4. 当前发现的问题

```text
1. 封面、作者、版权等 frontmatter 仍然进入了 chunk。
2. 纯数字和章节编号进入了词表，后续可能需要区分“有用编号”和“噪声编号”。
3. 停用词表还是手写的小表，没有根据语料统计生成。
4. 还没有把 term -> id 映射持久化到可跨进程查询的存储。
5. 还没有建立 Recall@k / MRR 评估集，只能确认链路可运行，不能宣布检索质量好。
6. PDF 入口已实现，但本实验没有真实 PDF 文件验证 MinerU 全页解析。
```

## 5. 适用边界

```text
可以确认:
    - jieba + BM25 的索引链路能在本地完整跑通。
    - 词表可以由语料构建，不需要 embedding。
    - 稀疏表示确实只保存少量非零词项。

不能从本实验确认:
    - “这个 chunk 策略”一定比其它策略好。
    - BM25 在这个语料上的召回率一定高。
    - PDF 全页解析一定没有遗漏。
```

## 6. 下一步

```text
1. 用真实 PDF 跑一次 --pdf 入口，核对页码范围、图片和表格内容。
2. 将 term -> id 与 IDF 统计持久化。
3. 给 chunk 增加 content_type，区分 body/frontmatter/toc/copyright。
4. 把分析器和 BM25 从实验脚本抽成可复用模块。
5. 接入 Qdrant sparse vector 或独立倒排索引，并建立评估集。
```
