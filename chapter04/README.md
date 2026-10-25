# 第 4 章：组装一个 GPT

先看 [chapter04.ipynb](chapter04.ipynb) 的形状演示，再看 [gpt.py](gpt.py)。

```text
token id [B,T]
  → 词向量 + 可学习的位置向量 [B,T,D]
  → TransformerBlock × N [B,T,D]
  → final_norm [B,T,D]
  → out_head [B,T,V]
```

每个 block 内有两条处理路线：多头注意力使一个 token 汇总它前面的信息；前馈网络在每个 token 内加工向量。每条路线之前做 LayerNorm，之后与原向量相加。`x + F(x)` 中的 `x` 给信息和梯度提供直接通路；`F(x)` 是这条路线学到的修改。

`out_head` 把长度为 `D` 的向量变成长度为 `V` 的预测值。所有 block 都负责逐步形成更有用的表示，最后由 `out_head` 做词表预测。模型刚创建时参数是随机的，输出没有语言能力；第 5 章才训练这些参数。

原书本章采用**可学习绝对位置 embedding**。正弦/余弦编码是原始 Transformer 论文里的另一种位置方法，不是此处 `nn.Embedding` 的计算方式。
