# 实验 08: 生成模型延迟对比

## 1. 实验条件

```text
日期            2026-09-13
检索            Hybrid + RRF
上下文数        5
上下文上限      6000 字
max_tokens      700
对比模型        qwen-plus / qwen-turbo
接口            POST /api/answer
输出            experiments/17_generation.txt
                experiments/18_generation_turbo.txt
```

## 2. 实测结果

```text
模型          retrieval_ms  generation_ms  total_ms  answer_chars  citations
qwen-plus     2             9868           9871      705           5
qwen-plus     2             15884          15886     -             -
qwen-turbo    2             3093           3095      553           5
```

其中第二次 `qwen-plus` 数据来自页面上的实际交互。

## 3. 结论

### 3.1 延迟主要来自生成, 不来自检索

检索稳定在毫秒级:

```text
retrieval_ms = 2 ~ 847
```

同一个进程缓存查询向量后, 检索约 `2 ms`。生成模型耗时则达到秒级。

### 3.2 qwen-turbo 明显更快

本次实测:

```text
qwen-plus   ~9.9s ~ 15.9s
qwen-turbo  ~3.1s
```

相对于约 `9.9s` 的 plus 结果, turbo 约为 `3.2x` 加速；
相对于页面约 `15.9s` 的结果, turbo 约为 `5.1x` 加速。

### 3.3 当前默认策略

```text
qwen-turbo  页面默认, 适合交互
qwen-plus   页面可选, 适合需要更高质量的长回答
```

## 4. 适用边界

```text
1. 这不是严格的模型质量评估, 只比较延迟。
2. 两次回答长度和内容不同, 生成耗时不能完全等价。
3. 非流式请求的耗时包含排队、网络、模型推理和完整答案生成。
4. qwen-turbo 更短的回答也会降低耗时, 不能只归因于模型本身。
```

## 5. 下一步

```text
1. 增加流式输出, 优先降低首 token 等待时间。
2. 对同一问题和同一上下文做 turbo/plus 的固定输入对比。
3. 增加答案质量、引用准确率评估。
4. 测试 context_k=3 / 4 对延迟和质量的影响。
```
