# 第 3 章：自注意力、因果遮挡与多头

从 [chapter03.ipynb](chapter03.ipynb) 看数值与形状，完整类在 [attention.py](attention.py)。

| 版本 | 输入 | 关键变化 |
| --- | --- | --- |
| `simplified_self_attention` | `[T,D]` | 直接用输入作 Q/K/V，没有可训练参数 |
| `SelfAttention_v1` | `[T,D]` | 三张 `nn.Parameter` 矩阵 |
| `SelfAttention_v2` | `[T,D]` | 改用三个 `nn.Linear` |
| `CausalAttention` | `[B,T,D]` | 屏蔽未来 token，训练时可 dropout |
| `MultiHeadAttentionWrapper` | `[B,T,D]` | 多个独立的因果注意力模块 |
| `MultiHeadAttention` | `[B,T,D]` | 大矩阵算 Q/K/V，然后按列分头、拼接、输出投影 |

记住 `Q @ Kᵀ → scores → softmax → weights → weights @ V → context`。`Kᵀ` 是转置，目的是让每个 query 与**所有** key 做点积。`**0.5` 是开平方；除以 `√head_dim` 有助于避免 softmax 太极端。`dim=-1` 是沿着每一行的“可关注 token”方向做 softmax。

因果 mask 固定地遮住右上角，`dropout` 则只在训练时随机丢掉一部分注意力权重。推理时先调用 `model.eval()`，dropout 就关闭。多头中的每个头都看同一组完整输入，但用自己的 Q/K/V 参数学习不同的表示。
