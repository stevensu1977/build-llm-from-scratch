# 第 1 章：先看完整路线

第 1 章介绍语言模型是什么、预训练与后训练如何衔接，以及本书将怎样逐步搭出 GPT。它主要是概念说明，没有需要逐行复现的模型代码。

```text
文字 → token ID → token embedding + position embedding
     → 多层 TransformerBlock → out_head → 下一个 token 的分数
```

模型训练时，根据已有 token 预测下一个 token，用预测错误程度（loss）调整参数。生成时则一次选择一个新 token，再把它放回输入继续预测。

后面各章会按这条路线展开：

- 第 2 章：怎样把文字变成有位置的向量。
- 第 3 章：一个 token 怎样从其他 token 获取信息。
- 第 4 章：怎样把注意力放进完整的 GPT。
- 第 5 章：怎样计算 loss、训练和生成。
