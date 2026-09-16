# 稀疏检索通用概念: 词项、词表、倒排索引与 BM25

> 本文只建立稀疏检索的通用概念, 不讨论具体代码、数据库或项目实现。
>
> 核心问题:
>
> ```text
> 稀疏检索到底在做什么?
> 为什么文本要先变成词项?
> 词表是什么?
> 倒排索引和稀疏向量有什么区别?
> BM25 为什么要用 tf、df 和文档长度?
> ```

---

## 一、稀疏检索的核心思想

稀疏检索的基本判断标准是:

```text
查询和文档是否包含相同或相关的词项?
```

它不直接比较整段文本的语义。

它比较的是:

```text
query 中有哪些词项
document 中有哪些词项
这些词项有多重要
```

例如:

```text
query:
    keepalived 故障转移

document A:
    keepalived 使用 VRRP 实现故障转移

document B:
    HAProxy 负责 HTTP 负载均衡
```

稀疏检索更容易发现 A，因为 A 和 query 共享:

```text
keepalived
故障转移
```

B 没有这些词项，因此通常得分较低。

所以稀疏检索的核心是:

```text
词项匹配 + 词项权重 + 排序
```

---

## 二、为什么叫“稀疏”

假设整个语料只有这些词:

```text
[keepalived, haproxy, 故障转移, 负载, 均衡, vrrp]
```

那么词表大小就是 6。

一篇文档一般不会同时包含词表中的所有词。

例如:

```text
文档 A:
    keepalived 故障转移 vrrp

词表向量:
    [1, 0, 1, 0, 0, 1]
```

文档 B:

```text
    haproxy 负载 均衡

词表向量:
    [0, 1, 0, 1, 1, 0]
```

可以看到:

```text
向量维度 = 词表大小
大部分位置是 0
只有少数位置非零
```

这就是“稀疏”。

```text
稀疏向量
    维度很大
    非零值很少

每一维通常对应一个词项
```

它和稠密向量的区别是:

```text
稀疏向量
    维度来源 = 词表
    非零位置 = 出现的词项

稠密向量
    维度来源 = 模型结构
    每个维度通常不对应具体词项
```

---

## 三、文本到词项

计算机不能直接比较“词”的字符串，通常要先把文本转成词项。

这个过程叫:

```text
analysis
analyzer
tokenization
```

完整流程可以抽象成:

```text
原始文本
    |
    v
归一化
    |
    v
分词
    |
    v
过滤或规范化
    |
    v
词项序列
```

### 归一化

把不同写法变成统一形式:

```text
Keepalived -> keepalived
ＶＲＲＰ    -> vrrp
```

### 分词

把文本切分成可比较的单位:

```text
"Keepalived 使用 VRRP"
    -> [keepalived, 使用, vrrp]
```

不同语言的切分方式不同:

```text
英文
    通常按空格和标点切分

中文
    通常需要专门的分词方法

代码和技术文本
    可能需要保护文件名、路径、配置项和错误码
```

### 停用词过滤与词形归一化

可选步骤:

```text
停用词过滤
    the, of, 的, 了

词形归一化
    running -> run
```

这些步骤会影响最终词项空间。

因此在同一个检索系统中:

```text
文档和查询必须使用一致的分析规则
```

---

## 四、词表: 把词项变成坐标

词表是语料中所有可检索词项的集合:

```text
V = {term1, term2, ..., termV}
```

为了让计算机更高效地处理，通常建立映射:

```text
term -> term_id
```

例如:

```text
keepalived -> 0
haproxy    -> 1
故障转移    -> 2
负载        -> 3
均衡        -> 4
vrrp        -> 5
```

也可以保留反向映射:

```text
0 -> keepalived
1 -> haproxy
2 -> 故障转移
```

词表的作用是:

```text
给每个词项分配一个固定坐标
```

没有词表，就无法把词项序列稳定地表示成数值向量。

### 词表不是什么

词表不是:

```text
高频词排名
检索结果
BM25 分数
文档摘要
```

词表只是一个坐标系统。

---

## 五、文档和查询的稀疏表示

有了词表后，一篇文档可以表示成一个稀疏向量:

```text
doc vector:
    [value(word1), value(word2), ..., value(wordV)]
```

查询也可以表示成:

```text
query vector:
    [value(word1), value(word2), ..., value(wordV)]
```

其中:

```text
value >= 0
没有出现的词项 value = 0
出现过的词项 value 由权重决定
```

最简单的权重是:

```text
出现 = 1
不出现 = 0
```

但这种方法太粗糙，因为:

```text
出现一次和出现十次没有区别
重要词和常见词没有区别
长文档和短文档没有区别
```

所以需要更合理的统计权重。

---

## 六、四个基本统计量

### 6.1 TF: 词频

TF 表示一个词项在一篇文档中出现的次数。

```text
tf(t, d) = term t 在文档 d 中出现的次数
```

例如:

```text
doc A:
    keepalived keepalived vrrp

tf(keepalived, A) = 2
tf(vrrp, A) = 1
```

直觉:

```text
一个词在文档中出现越多, 通常说明它越重要
```

但词频不能无限增加权重。

### 6.2 DF: 文档频率

DF 表示一个词项出现在多少篇文档中。

```text
df(t) = 包含 term t 的文档数量
```

例如:

```text
doc A: keepalived fault
doc B: keepalived config
doc C: haproxy balance

df(keepalived) = 2
df(haproxy) = 1
```

直觉:

```text
df 越大 -> 词越常见 -> 区分度越低
df 越小 -> 词越稀有 -> 区分度越高
```

注意:

```text
df 统计的是包含该词的文档数量
不是该词在所有文档中的总出现次数
```

### 6.3 N: 文档总数

```text
N = 语料中的文档总数
```

如果语料有 100 篇文档:

```text
N = 100
```

### 6.4 IDF: 逆文档频率

IDF 用 df 衡量词项的稀有程度:

```text
IDF(t) = 与 df(t) 成反比的权重
```

一个常见形式是:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

直觉:

```text
词出现在越多文档中
    -> df 越大
    -> IDF 越小

词只出现在少数文档中
    -> df 越小
    -> IDF 越大
```

因此:

```text
“的” 这类常见词权重低
“TS-999” 这类稀有术语权重高
```

---

## 七、文档长度和平均文档长度

BM25 还考虑文档长度。

```text
|d|     = 文档 d 的 token 数
avgdl   = 所有文档的平均 token 数
```

例如:

```text
doc A = 100 tokens
doc B = 20 tokens

同一个词在 A 和 B 中各出现一次
```

长文档天然更容易包含某个词。

如果完全不考虑长度，长文档会占优势。

所以 BM25 加入长度归一化:

```text
文档比平均长度越长
通常越受到惩罚
```

---

## 八、倒排索引

### 8.1 先把“正向”和“倒排”分开

最自然的文档表示方式叫正向索引:

```text
document -> terms
```

例如:

```text
D1 -> [keepalived, 使用, vrrp, 故障转移]
D2 -> [keepalived, 配置, 故障转移]
D3 -> [haproxy, 负载, 均衡]
```

它的方向是:

```text
先知道文档, 再查文档里有哪些词项
```

但如果用户 query 是一个词:

```text
keepalived
```

我们想快速知道哪些文档包含它，正向索引就不方便。

于是把映射方向反过来:

```text
term -> documents
```

例如:

```text
keepalived -> [D1, D2]
故障转移    -> [D1, D2]
vrrp       -> [D1]
haproxy    -> [D3]
```

这就是倒排索引:

```text
term -> 包含该 term 的文档集合
```

“倒排”不是指倒着排序，而是指:

```text
把 document -> terms
反过来建立 term -> documents
```

### 8.2 term dictionary 和 posting list

倒排索引通常包含两部分:

```text
term dictionary
    term -> posting list 的位置

posting list
    所有包含该 term 的文档记录
```

最简单:

```text
keepalived -> [D1, D2, D7]
haproxy    -> [D3]
vrrp       -> [D1]
```

这里的:

```text
[D1, D2, D7]
```

就是 `keepalived` 的 posting list。

### 8.3 posting list 里有什么

最简单的 posting 只保存:

```text
doc_id
```

例如:

```text
keepalived -> [D1, D2, D7]
```

但 BM25 需要知道词频，所以通常至少保存:

```text
doc_id
tf
```

例如:

```text
keepalived -> [
    (D1, tf=2),
    (D2, tf=1),
    (D7, tf=4)
]
```

如果还需要支持短语查询和邻近查询，还可以保存位置:

```text
keepalived -> [
    (D1, tf=2, positions=[0, 8]),
    (D2, tf=1, positions=[3]),
]
```

三种信息的用途不同:

```text
doc_id      哪些文档包含该 term
tf          该 term 在文档中出现多少次
positions   该 term 出现在哪些位置
```

对应到打分:

```text
doc_id
    决定候选文档

tf
    给 BM25 提供词频

positions
    支持短语和邻近查询
```

### 8.4 查询时如何使用倒排索引

假设 query 分析后是:

```text
query terms = [keepalived, 故障转移]
```

倒排索引:

```text
keepalived -> [D1, D2]
故障转移    -> [D1, D2, D3]
```

查询过程:

```text
Step 1
    查 keepalived 的 posting list
    -> D1, D2

Step 2
    查 故障转移 的 posting list
    -> D1, D2, D3

Step 3
    合并或求交
    -> 候选文档 D1, D2, D3

Step 4
    对候选文档计算 BM25

Step 5
    按分数排序

Step 6
    返回 top-k
```

为什么高效:

```text
不需要扫描所有文档
只访问 query 中实际出现的 term 对应的 posting list
```

如果 query 中有 3 个词:

```text
只为这 3 个词查找对应的 posting list
```

如果某个 query term 不在词表中:

```text
没有对应的 posting list
该 term 不产生候选
```

### 8.5 倒排索引不负责打分

倒排索引解决的是:

```text
query term -> 可能相关的文档
```

它本身不等于 BM25。

它还需要和其它统计量结合:

```text
tf
df
N
|d|
avgdl
```

然后由 BM25 计算:

```text
score(query, document)
```

所以:

```text
倒排索引负责缩小候选范围
BM25 负责给候选排序
```

### 8.6 倒排索引和稀疏向量的关系

倒排索引:

```text
term -> postings
```

稀疏向量:

```text
indices + values
```

两者都可以表达:

```text
某篇文档包含哪些非零词项
```

区别是它们的物理组织方式和查询方式:

```text
倒排索引
    通过 term 查 posting list
    更适合词项查找

稀疏向量
    通过点积计算 query 和文档的相关度
    更适合统一的向量检索接口
```

所以:

```text
倒排索引是一种索引结构
稀疏向量是一种数值表示
BM25 是一种打分函数
```

三者解决的是不同层面的问题。

---

## 九、稀疏向量

稀疏向量是另一种表示:

```text
indices = [非零词项 ID]
values  = [对应权重]
```

例如:

```text
doc vector:
    indices = [0, 3, 7]
    values  = [1.2, 0.8, 2.1]
```

它不显式保存:

```text
[0, 0, 1.2, 0, 0, 0, 0, 0.8, ..., 2.1]
```

只保存非零位置。

因此:

```text
稀疏向量适合用点积计算相关度
倒排索引适合通过词项快速找候选文档
```

两者描述的是同一类稀疏词项证据，只是组织方式不同。

---

## 十、TF-IDF 的基本思想

TF-IDF 是最经典的稀疏权重之一:

```text
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

意思是:

```text
一个词在文档中越常出现, 权重越高
一个词在整个语料中越稀有, 权重越高
```

但它有两个问题:

```text
1. 词频线性增长
2. 文档长度影响没有充分建模
```

BM25 是对这些问题的进一步改进。

---

## 十一、BM25 的核心公式

BM25 对查询 `q` 和文档 `d` 的评分可以写成:

```text
score(q, d)
= Σ IDF(t)
    * [ f(t,d) * (k1 + 1) ]
    / [ f(t,d) + k1 * (1 - b + b * |d| / avgdl) ]
```

其中:

```text
t        query 中的词项
f(t,d)   词项 t 在文档 d 中的词频
|d|      文档 d 的长度
avgdl    平均文档长度
k1       词频饱和参数
b        文档长度归一化参数
```

BM25 的直觉:

```text
1. 词项越稀有, IDF 越高
2. 词项在文档中出现越多, 分数越高
3. 但词频增长会饱和
4. 文档越长, 匹配难度越大, 会受到长度归一化
```

### 11.1 词频饱和

如果词频线性增长:

```text
出现 1 次  -> 权重 1
出现 10 次 -> 权重 10
出现 100 次 -> 权重 100
```

这通常不合理。

BM25 让词频增长逐渐变慢:

```text
出现 1 次  -> 增加明显
出现 10 次 -> 增加变缓
出现 100 次 -> 边际收益很小
```

### 11.2 文档长度归一化

参数 `b` 控制长度归一化强度:

```text
b = 0
    不做长度归一化

b = 1
    完全使用长度归一化

常见 b = 0.75
```

---

## 十二、一个可手算的小例子

假设只有三篇文档:

```text
D1 = keepalived, vrrp, 故障转移
D2 = haproxy, 负载, 均衡
D3 = keepalived, 故障转移
```

### 统计 df

```text
df(keepalived) = 2
df(vrrp) = 1
df(故障转移) = 2
df(haproxy) = 1
df(负载) = 1
df(均衡) = 1
```

### 文档长度

```text
|D1| = 3
|D2| = 3
|D3| = 2

avgdl = (3 + 3 + 2) / 3 = 2.6667
```

### 假设 query

```text
q = keepalived, 故障转移
```

对于 D1 和 D3，两个查询词都出现。

计算 BM25 时:

```text
keepalived 和 故障转移的 df = 2
    -> IDF 较低

文档长度越接近或低于 avgdl
    长度归一化影响较小

同一个词在文档中出现次数越多
    tf 部分越高, 但受 k1 饱和
```

最终:

```text
D1 和 D3 比 D2 得分更高
```

D2 不含 query 词项，因此主要得分为 0。

---

## 十三、稀疏检索的索引阶段

通用索引流程:

```text
文档集合
    |
    v
分析器
    |
    v
词项序列
    |
    v
建立词表
    |
    v
统计 tf / df / 文档长度
    |
    v
计算词项权重
    |
    v
建立倒排索引或稀疏向量索引
```

索引一旦建立，就意味着:

```text
词表已经确定
term_id 已经确定
df / IDF 已经基于当前语料计算
文档长度统计已经保存
```

这解释了为什么:

```text
语料或分词规则改变后
通常需要重新建立索引
```

---

## 十四、稀疏检索的查询阶段

```text
query
    |
    v
同一个分析器
    |
    v
query 词项
    |
    v
映射到词表
    |
    v
找到候选文档
    |
    v
计算 BM25 或其它稀疏分数
    |
    v
按分数排序
    |
    v
返回 top-k
```

查询中最重要的是:

```text
1. query 和文档必须使用同一词项空间
2. 只有出现过的词项参与匹配
3. top-k 是文档的排名, 不是词项排名
```

---

## 十五、倒排索引和 BM25 的区别

这是两个层次:

```text
倒排索引
    索引结构
    负责找到候选文档

BM25
    打分函数
    负责给候选文档排序
```

例如:

```text
query = keepalived, 故障转移

1. 查 posting list:
    keepalived -> D1, D3
    故障转移   -> D1, D2, D3

2. 合并候选:
    D1, D2, D3

3. 计算 BM25:
    D1 -> 高分
    D3 -> 高分
    D2 -> 低分或 0

4. 排序:
    D1, D3, D2
```

所以:

```text
倒排索引不是 BM25
稀疏向量也不是 BM25
BM25 是打分算法
```

---

## 十六、稀疏检索的优势

```text
1. 精确词项匹配强
    错误码、配置项、版本号、产品名

2. 可解释
    能看出哪些词项命中

3. 通常不需要神经网络
    经典 TF-IDF 和 BM25 主要依赖统计量

4. 成本相对可控
    索引和查询都可以利用稀疏性

5. 适合作为混合检索的字面通道
    和 dense 语义检索互补
```

---

## 十七、稀疏检索的局限

```text
1. 同义词弱
    “汽车” 和 “automobile”

2. 改写能力弱
    “如何避免请求集中到一台机器”
    vs
    “负载均衡”

3. 词表不匹配
    OOV、错误分词、领域词被拆碎

4. 多义词难处理
    同一个词在不同语境下含义不同

5. 依赖分析器
    分词和归一化规则会直接改变结果
```

这也是为什么工程上经常使用:

```text
sparse retrieval + dense retrieval + rank fusion + rerank
```

但这些都是后续概念。

本文只需要先建立:

```text
稀疏检索 = 词项空间中的匹配和加权排序
```

---

## 十八、一页总结

```text
文档集合
    |
    v
分析器
    |
    v
词项序列
    |
    v
词表
    term -> term_id
    |
    v
统计量
    tf / df / N / |d| / avgdl
    |
    v
权重
    TF-IDF / BM25 / 其它
    |
    +----------------------+
    |                      |
    v                      v
倒排索引               稀疏向量
term -> documents      indices + values
    |                      |
    +----------+-----------+
               |
               v
        query -> 同一词项空间
               |
               v
          匹配候选
               |
               v
           BM25 排序
               |
               v
             top-k
```

最终记住三句话:

```text
词表是稀疏检索的坐标系。
倒排索引和稀疏向量是两种表示方式。
BM25 是给词项匹配结果排序的打分函数。
```
