# 稀疏召回 vs 稠密召回: 词表、向量、索引与工程边界

> 本文是 `retrieval_scoring_and_rrf.md` 的深入拆解版, 重点回答:
>
> ```text
> 稀疏召回和稠密召回到底指什么?
> 稀疏召回里的词表从哪来?
> embedding 的词表和稀疏检索的词表是不是一回事?
> 为什么一个擅长精确匹配, 另一个擅长语义匹配?
> 两路召回在实际工程中怎样协作?
> ```
>
> 关联文档:
> - 打分与 RRF: `retrieval_scoring_and_rrf.md`
> - 当前通道问题: `../issues/04_channel_layer.md`
> - BM25 实验脚本: `../pipeline/retrieval_channel_compare.py`
> - 当前 dense 链路: `../pipeline/rag_pipeline.py`

---

## 一、一句话区分

```text
稀疏召回:
    比较"哪些词出现了"
    文本映射到词表维度
    代表算法: TF-IDF / BM25 / SPLADE

稠密召回:
    比较"整体语义像不像"
    文本映射到模型维度
    代表模型: BGE / Qwen embedding / E5 / GTE
```

“稀疏”和“稠密”首先描述的是**向量的密度**:

```text
稀疏向量:
    维度很大, 大部分是 0
    非零位置通常对应具体词项

稠密向量:
    维度较小, 大部分维度都有值
    每个维度通常不对应一个具体词
```

在 RAG 工程里, 这两种表示各自形成一条召回路径, 所以也叫:

```text
稀疏通道
稠密通道
多路召回
```

“通道”不是数据库术语, 而是工程术语。它表示一条能够独立产生候选文档和排名的检索分支。

---

## 二、概念从哪里来

### 2.1 稀疏召回来自向量空间模型

早期信息检索把文档看成词语的集合。

假设整个语料有词表:

```text
V = [w1, w2, w3, ..., wV]
```

文档 `d` 可以表示成:

```text
v_d = [weight(w1, d), weight(w2, d), ..., weight(wV, d)]
```

如果 `w` 没有出现在文档中, 对应位置就是 `0`。

现实中的词表可能有几万到几十万维, 但一篇文档只包含其中一小部分词, 所以向量天然稀疏。

这条路线的发展大致是:

```text
词袋模型
    |
    v
TF-IDF
    |
    v
BM25
    |
    v
SPLADE / 学习型稀疏
```

### 2.2 稠密召回来自分布式表示

神经网络 embedding 不再要求“一个维度对应一个词”, 而是把整段文本压成一个固定长度向量:

```text
文本 -> embedding model -> [0.12, -0.07, 0.31, ..., 0.09]
```

这个向量里的每一个数通常没有单独可解释的含义。语义信息分布式地编码在多个维度里。

这条路线通常依赖:

```text
预训练语言模型
对比学习
文本对训练
向量相似度
近似最近邻索引
```

因此, 稀疏召回主要来自统计检索传统, 稠密召回主要来自神经表示学习传统。

---

## 三、稀疏召回: 核心是词项匹配

### 3.1 什么是词项

进入检索词表之前, 文本要先经过分析器:

```text
原始文本
    |
    v
大小写归一化
    |
    v
分词
    |
    v
词形归一化 / 停用词过滤
    |
    v
词项序列
```

例如:

```text
原文:
    Keepalived 使用 VRRP 实现故障转移

可能得到:
    [keepalived, 使用, vrrp, 实现, 故障, 转移]
```

这里的每一个输出都叫一个 token 或 term。

### 3.2 词表是什么

词表是索引中所有可识别词项的集合:

```text
V = {v1, v2, v3, ..., vV}
```

它不是“每篇文档自己的词表”, 而是整个索引共享的词项空间。

假设语料只有两篇文档:

```text
D1 = Keepalived 使用 VRRP 实现故障转移
D2 = HAProxy 提供负载均衡
```

分词后:

```text
D1 tokens =
    [keepalived, 使用, vrrp, 实现, 故障, 转移]

D2 tokens =
    [haproxy, 提供, 负载, 均衡]
```

词表是两者的并集:

```text
V = {
    keepalived,
    haproxy,
    使用,
    vrrp,
    实现,
    故障,
    转移,
    提供,
    负载,
    均衡
}
```

如果词表有 `10` 个词, 稀疏向量就是 `10` 维。

### 3.3 稀疏向量长什么样

给词表编号:

```text
0  keepalived
1  haproxy
2  使用
3  vrrp
4  实现
5  故障
6  转移
7  提供
8  负载
9  均衡
```

那么:

```text
D1 = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
D2 = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
```

这里的 `0` 和 `1` 只是最简单的“是否出现”。

真实系统可以把 `1` 替换成更复杂的权重:

```text
词频
TF-IDF 权重
BM25 贡献
学习出来的稀疏权重
```

所以广义上可以说:

```text
稀疏向量 = 词表大小维度的加权词项向量
```

但经典 BM25 实现通常不真的构造这个完整数组, 而是直接查倒排索引。

### 3.4 倒排索引是什么

倒排索引保存:

```text
词项 -> 哪些文档包含它
```

例如:

```text
keepalived -> [D1]
haproxy    -> [D2]
vrrp       -> [D1]
负载       -> [D2]
```

复杂一些的倒排索引还保存:

```text
词频
出现位置
字段信息
```

查询时:

```text
1. 对 query 做同样的分析
2. 查找 query 词对应的 posting list
3. 对候选文档计算 BM25 等分数
4. 排序返回 top-k
```

因为大部分词不会出现在大部分文档中, 稀疏向量在存储和计算上都可以利用稀疏性。

### 3.5 经典 BM25 公式

BM25 对查询 `q` 和文档 `d` 的打分:

```text
score_BM25(q, d)
= Σ_{t ∈ q}
    IDF(t)
    *
    f(t,d) * (k1 + 1)
    -----------------------------------------
    f(t,d) + k1 * (1 - b + b * |d| / avgdl)
```

其中:

```text
t        查询中的一个词项
f(t,d)   词项 t 在文档 d 中出现的次数
|d|      文档 d 的长度
avgdl    平均文档长度
k1       词频饱和参数
b        文档长度归一化强度
```

BM25 只对查询中出现的词项累加分数。没有出现的词仍然可以理解为权重为零。

这就是它属于稀疏检索的原因:

```text
词表很大
每次查询只命中极少数词项
大部分维度不参与计算
```

### 3.6 稀疏召回里的词表从哪里来

不同系统有三种主要来源。

#### 方式 A: 从当前语料构建

```text
遍历所有文档
    |
    v
对每篇文档分词
    |
    v
收集所有唯一 token
    |
    v
形成词表和倒排索引
```

当前 BM25 实现属于这一种。

代码位置:

```text
../pipeline/retrieval_channel_compare.py
```

代码中的:

```python
self.df
```

保存“词项 -> 包含该词项的文档数”。它的 key 集合实际上就是当前 BM25 索引的词表。

查询时:

```python
if w not in self.df:
    continue
```

如果查询词没有出现在语料词表中, 它就无法贡献 BM25 分数。

#### 方式 B: 使用固定 tokenizer 词表

某些稀疏模型预先确定了 tokenizer 词表:

```text
V = 模型固定的 token 集
```

文本被切成 token 后, 通过模型预测每个 token 的权重。

SPLADE 这类学习型稀疏检索就属于这一路线。

它和经典 BM25 的区别是:

```text
BM25:
    权重主要由 TF、IDF、文档长度统计得到

SPLADE:
    权重由神经网络学习得到
    可以为查询扩展出原查询中没有出现的相关词项
```

#### 方式 C: 用预训练模型同时输出 dense 和 sparse

BGE-M3 可以输出:

```text
dense vector
sparse weights
multi-vector
```

它的 sparse 输出本质上是模型学习的稀疏词项权重, 可用于稀疏召回；dense 输出用于语义召回。

这不代表两种召回结果自动融合。融合仍然需要 RRF、加权分数融合或 rerank。

### 3.7 词表如何影响召回

假设查询是:

```text
virtual_ipaddress 怎么配置
```

如果索引词表中存在完整词项:

```text
virtual_ipaddress
```

BM25 可以精确命中。

如果分词器把它拆成:

```text
virtual
_
ip
address
```

但文档索引时用了另一种切法, 例如:

```text
virtual_ipaddress
```

查询词和文档词没有对齐, 就可能出现假阴性:

```text
文档真的包含这个词
但 BM25 没有匹配到
```

因此, 稀疏召回最重要的工程约束不是“词表大不大”, 而是:

```text
索引和查询必须使用同一套分析规则
```

包括:

```text
同一个分词器
同一个版本
同一套大小写规则
同一套停用词表
同一个领域词典
同一种数字和符号处理方式
```

### 3.8 中文词表的特殊问题

中文没有天然空格边界, 分词质量直接决定词表质量。

例如:

```text
keepalived 配置高可用集群
```

可能被切成:

```text
[keepalived, 配置, 高可用, 集群]
```

也可能被错误切成:

```text
[keep, alive, d, 配置, 高, 可用, 集群]
```

技术文档中尤其需要保护:

```text
产品名         keepalived
配置项名       virtual_ipaddress
错误码         TS-999
版本号         v2.4.1
文件路径       /etc/keepalived/keepalived.conf
协议名         VRRP
```

常见做法:

```text
1. 增加领域词典
2. 对配置项、错误码、路径使用专门规则
3. 在关键词通道中保留大小写和符号
4. 对标识符同时保留整词和子词特征
```

### 3.9 未登录词问题

未登录词通常叫 OOV, Out-of-Vocabulary。

经典 BM25 的 OOV 处理可能是:

```text
直接忽略
映射到 UNK
拆成字符或子词
```

这三种处理的效果不同:

```text
忽略:
    完全不匹配, 可能漏召回

UNK:
    不同未知词可能都映射到同一个词项, 区分度下降

子词:
    更容易部分匹配, 但精确短语可能被拆散
```

这就是为什么技术文档的词表需要领域定制。

---

## 四、稠密召回: 核心是语义向量

### 4.1 embedding 做什么

embedding 模型是一个函数:

```text
z = f_theta(text)
```

输入文本, 输出固定长度的稠密向量:

```text
输入:
    "keepalived 负责虚拟 IP 故障转移"

输出:
    [0.12, -0.08, 0.31, ..., 0.05]
```

如果输出维度是 `1024`, 那么向量有 `1024` 个数。

### 4.2 维度是什么

例如:

```text
768 维向量
```

不是:

```text
768 个汉字
768 个词
768 个 token
```

而是:

```text
向量数组里有 768 个浮点数
```

维度由模型架构和训练决定:

```text
bge-small-zh-v1.5    常见 512 维
bge-base-zh-v1.5     常见 768 维
bge-large-zh-v1.5    常见 1024 维
qwen3-vl-embedding   当前项目使用 1024 维
```

不同模型、不同维度、不同归一化方式的向量不能直接互相比较。更换 embedding 模型通常需要重建整个向量索引。

### 4.3 embedding 内部也有 tokenizer 词表吗

有, 但它和稀疏召回的词表不是同一个概念。

embedding 模型内部通常会:

```text
文本
    |
    v
tokenizer: 文本 -> token ids
    |
    v
Transformer: token ids -> 上下文向量序列
    |
    v
pooling: 多个 token 向量 -> 一个文本向量
```

例如:

```text
"keepalived"
    |
    v
tokenizer 词表中的子词序列
```

但最终输出:

```text
[0.12, -0.08, 0.31, ..., 0.05]
```

这里的 `1024` 个输出维度不对应 `1024` 个词。

所以必须区分:

```text
稀疏召回的词表
    = 检索空间的维度
    = 一个维度通常对应一个词项

embedding 模型的 tokenizer 词表
    = 模型内部编码文本时使用的 token 集合
    = 不是最终向量维度的含义
```

### 4.4 相似度怎么算

最常见的检索相似度是余弦相似度:

```text
cos(q, d) = (q · d) / (||q|| * ||d||)
```

其中:

```text
q · d = 向量点积
||q|| = q 的 L2 范数
||d|| = d 的 L2 范数
```

取值范围:

```text
[-1, 1]
```

排序含义:

```text
相似度越大 -> 越相似 -> 排名越靠前
```

有些模型使用点积:

```text
score = q · d
```

如果向量已经归一化:

```text
q · d = cos(q, d)
```

不同向量库对距离的定义不同:

```text
COSINE     余弦相似度, 越大越相似
DOT        点积, 越大越相似
EUCLIDEAN  L2 距离, 越小越相似
```

查询、索引和阈值必须使用同一种距离语义。

### 4.5 稠密向量如何存储和检索

如果只有几百个 chunk, 可以直接:

```text
query 和每个 chunk 向量算 cosine
```

这就是精确暴力检索。

当 chunk 达到百万级时, 通常会使用 ANN, 即 Approximate Nearest Neighbor:

```text
HNSW
IVF
PQ
```

HNSW 常见思路:

```text
把向量建成多层图
从上层粗略定位到附近区域
再逐层向下搜索邻居
```

ANN 的取舍:

```text
速度快
召回率高但通常不是 100%
需要调节构建参数和查询参数
```

### 4.6 稠密召回的优势

稠密召回擅长:

```text
同义表达
    "汽车" <-> "automobile"

自然语言改写
    "主节点挂掉后怎么自动切换"
    <-> "故障转移机制"

跨词表达
    "避免请求集中到一台机器"
    <-> "负载均衡"

语义摘要匹配
    问题是对文档内容的概括
```

### 4.7 稠密召回的失效模式

稠密召回容易在以下场景失效:

```text
数值和编号
    TS-999 和 TS-990 可能很接近

配置项名
    virtual_ipaddress 和 virtual_router_id

文件名和函数名
    keepalived.conf 和 keepalived.service

罕见专有名词
    模型训练中很少见的术语

精确短语
    用户直接粘贴文档中的原始句子

长文档中的局部细节
    一个向量不足以表达所有局部信息
```

根本原因:

```text
embedding 是有损压缩。
它优先保留整体语义, 不保证保留每个字符和局部事实。
```

---

## 五、两者的核心差异

| 维度 | 稀疏召回 | 稠密召回 |
|---|---|---|
| 核心表示 | 词项权重 | embedding 向量 |
| 向量维度 | 词表大小, 可能几十万 | 模型维度, 常见 384 ~ 1536 |
| 非零维度 | 很少 | 通常很多 |
| 强项 | 精确词、编号、配置项、术语 | 同义、改写、语义相近 |
| 弱项 | 同义词、语义泛化 | 精确字符串、罕见术语 |
| 索引 | 倒排索引 | HNSW / IVF 等向量索引 |
| 打分 | TF-IDF / BM25 | cosine / dot / L2 |
| 查询处理 | 分词、查 posting list | embedding、ANN 搜索 |
| 可解释性 | 较高, 可看命中词项 | 较低, 难解释单个维度 |
| 词表关系 | 词表就是检索维度 | 内部 tokenizer 词表不等于输出维度 |
| 典型实现 | Lucene / Elasticsearch / BM25 | Qdrant / FAISS / Milvus |

要特别注意:

```text
稀疏不等于低质量
稠密不等于一定更好
```

它们是两种互补的信号。

---

## 六、同一查询在两条通道里会怎么走

假设查询:

```text
keepalived.conf 里 virtual_ipaddress 怎么配置
```

### 6.1 稀疏通道

```text
query
    |
    v
分词
    |
    v
[keepalived, conf, virtual_ipaddress, 配置]
    |
    v
查倒排索引
    |
    v
精确命中包含 virtual_ipaddress 的 chunk
    |
    v
BM25 打分
    |
    v
稀疏 top-N
```

它关心:

```text
virtual_ipaddress 这个字符串有没有出现
keepalived.conf 这个字符串有没有出现
```

### 6.2 稠密通道

```text
query
    |
    v
embedding
    |
    v
[0.11, -0.04, 0.63, ..., 0.27]
    |
    v
向量索引 ANN 搜索
    |
    v
语义相近的 chunk
    |
    v
cosine 排序
    |
    v
稠密 top-N
```

它关心:

```text
整个 query 的语义方向
```

它不一定能看到 `virtual_ipaddress` 某个字符是否精确出现。

### 6.3 两路融合

```text
稀疏 top-N1 ----+
                 |
                 v
               RRF
                 |
                 ^
                 |
稠密 top-N2 ----+
                 |
                 v
             融合 top-K
                 |
                 v
             rerank
                 |
                 v
            最终 top-k
```

RRF 使用排名而不是原始分数:

```text
RRF(d) = Σ_i 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档 `d` 在第 `i` 条通道中的名次。

---

## 七、映射到当前工程

当前工程已经有两部分基础:

### 7.1 dense 部分

当前完整链路:

```text
清洗
    |
    v
分块
    |
    v
Qwen embedding
    |
    v
Qdrant dense vector
    |
    v
cosine 检索
```

代码:

```text
../pipeline/rag_pipeline.py
```

它负责跨词和语义相似的召回。

### 7.2 sparse 部分

独立完整链路脚本:

```text
../pipeline/sparse_pipeline_demo.py
```

它负责从 PDF/Markdown 到 BM25 top-k 的完整验证:

```text
PDF/Markdown -> 清洗 -> 段落级实验分块
-> jieba -> term id -> 倒排索引 -> BM25 top-k
```

当前实测产物:

```text
../pipeline/experiments/14_sparse_pipeline.txt
../pipeline/experiments/14_sparse_pipeline_artifacts/
../pipeline/experiments/15_sparse_qdrant_pipeline.txt
../pipeline/experiments/15_sparse_qdrant_verify.txt
```

它还可以直接读取现有 dense collection 中的 chunk, 复用相同
`point_id/payload`, 再把 BM25 sparse vector 写入独立的
`redhat_sparse` collection。2026-09-15 增量加入 NGINX PDF 后,
当前共有 354 个 LlamaIndex chunk(Red Hat 42 + NGINX 312), dense/sparse 各 354 个 point。

对照实验脚本:

```text
../pipeline/retrieval_channel_compare.py
```

它使用:

```text
jieba 分词
BM25 k1=1.2
BM25 b=0.75
```

负责精确词项、配置项、错误码和术语的召回。

### 7.3 当前融合架构

```text
query
  |
  +-- Qdrant dense -> top-20
  |
  +-- BM25 sparse -> top-20
              |
              v
              RRF
              |
              v
           top-5
```

实现位于:

```text
../pipeline/hybrid_retriever.py
../pipeline/rag_server.py
../pipeline/static/index.html
```

实测输出见 `../pipeline/experiments/16_hybrid_rrf.txt`。rerank 第一版已接入,
使用 gte-rerank-v2, 对比结果见 `../pipeline/experiments/19_rerank_compare.txt`。

工程上要确保:

```text
1. dense 和 sparse 使用同一个 chunk id
2. 分词器版本固定
3. 查询和索引使用同一个分析器
4. 两路候选深度足够
5. 融合参数 k 和权重通过评估集验证
```

---

## 八、最容易混淆的五个问题

### 8.1 稀疏通道就是 BM25 吗

不是。

BM25 是稀疏召回的代表算法之一。稀疏召回还包括:

```text
TF-IDF
BM25+
SPLADE
BGE-M3 sparse
```

### 8.2 稠密通道就是向量数据库吗

不是。

```text
稠密通道 = embedding 模型 + 向量相似度 + 候选召回
向量数据库 = 存储和检索向量的基础设施
```

Qdrant、FAISS、Milvus 负责后者, 不负责生成语义向量。

### 8.3 稀疏向量和稠密向量是两种数据库吗

不是。它们是两种表示和检索方式。一个向量数据库也可能同时支持 dense 和 sparse 向量。

### 8.4 BM25 会生成 embedding 吗

通常不会。

BM25 主要依赖:

```text
词频
文档频率
文档长度
```

经典 BM25 不需要神经网络 embedding。

### 8.5 embedding 模型内部的 tokenizer 词表就是稀疏词表吗

不是。

```text
稀疏词表:
    直接决定检索向量有哪些维度

embedding tokenizer 词表:
    只决定模型怎样把文本切成 token
    最终还要经过 Transformer 和 pooling
```

这也是为什么 embedding 的输出维度通常和 tokenizer 词表大小无关。

---

## 九、工程检查清单

### 稀疏召回

```text
[ ] 索引和查询使用同一分词器
[ ] 分词器版本固定
[ ] 产品名、配置项、错误码进入领域词典
[ ] 明确大小写和符号是否保留
[ ] 检查停用词是否误删重要术语
[ ] 记录词表规模、平均文档长度、OOV 情况
[ ] 观察查询词是否经常不在词表中
```

### 稠密召回

```text
[ ] 索引和查询使用同一 embedding 模型
[ ] 模型版本固定
[ ] 向量维度一致
[ ] 距离类型一致: cosine / dot / L2
[ ] 向量归一化策略一致
[ ] 更换模型时重建索引
[ ] 记录 ANN 参数和候选深度
```

### 混合召回

```text
[ ] 两路使用相同 chunk id
[ ] 两路候选分别保留排名
[ ] RRF 的 k 和权重通过评估集验证
[ ] 记录每路单独召回结果
[ ] 不要只看最终融合结果而忽略单路失效
[ ] 用 Recall@k、MRR、nDCG 等真实指标评估
```

---

## 十、最终速记

```text
稀疏召回
    看词项
    维度 = 词表大小
    非零维度很少
    代表: BM25
    强项: 精确匹配

稠密召回
    看语义向量
    维度 = embedding 输出维度
    大部分维度有值
    代表: BGE / Qwen
    强项: 语义匹配

混合召回
    BM25 提供字面信号
    embedding 提供语义信号
    RRF 合并两路排名
    rerank 对少量候选精排
```

一句话:

```text
稀疏召回解决“它有没有说过这个词”,
稠密召回解决“它说的是不是同一个意思”,
混合召回同时利用这两种证据。
```
