# 问题 07: PDF 代码块被误判为标题, section 噪声严重

**优先级: P1**　**状态: 待解决**

## 现象

NGINX PDF 增量入库后, 部分 payload.section 变成:

```text
comment
name1:password1
name2:password2:comment
name3:password3 > 解决方案
```

原本这些内容属于代码块或配置示例, 不是章节标题。

## 根因

MinerU 输出 Markdown 时会对部分代码行加 `#` 或形成伪标题结构。
当前 `preprocess.is_heading()` 只能排除少量 shell 提示符,
无法覆盖 NGINX 配置、SAML 配置、变量示例等代码块。

## 影响

```text
section 路径不可信
按 section 过滤失效
引用展示出现配置代码
metadata 进入 index_text 时会污染 BM25 词项
```

## 解决方向

```text
1. 解析阶段识别代码围栏和配置块
2. 标题合法性增加长度、符号、配置语法规则
3. 代码块内容不进入 section 栈
4. 对 PDF 解析结果做独立结构 QA
5. 为不同 PDF 编写 parser-specific 清理测试
```

## 待验证

```text
[ ] 修复后 Nginx chunk 的 section 是否恢复为真实章节
[ ] section 过滤是否比当前更稳定
[ ] 修复 metadata 后 BM25 排序是否改善
[ ] 引用展示是否不再出现配置代码
```
