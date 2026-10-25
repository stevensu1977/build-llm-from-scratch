"""第 3 章：从简化版到多头因果注意力。

形状记号：B=batch 中的句子数，T=每句的 token 数，D=每个 token 的向量长度。
矩阵乘法 A @ B：A 的最后一维要等于 B 的倒数第二维。
"""

from __future__ import annotations

import torch
from torch import nn


def simplified_self_attention(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """没有可训练矩阵的教学版本；直接把 x 同时当作 Q、K、V。

输入 x: [T, D]。scores[i,j] 是第 i 个 token 对第 j 个 token 的点积。
softmax 让第 i 行变成和为 1 的权重，最后将各 token 向量加权求和。
"""
    scores = x @ x.T                 # [T,D] @ [D,T] -> [T,T]
    weights = torch.softmax(scores, dim=-1)
    context = weights @ x            # [T,T] @ [T,D] -> [T,D]
    return context, weights


class SelfAttention_v1(nn.Module):
    """书中的第一版可训练自注意力：自己声明三张参数矩阵。"""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # 每张表都把长度 d_in 的输入，映射为长度 d_out 的 Q/K/V。
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里 x 是一组 token [T,d_in]，还不是 batch。
        queries = x @ self.W_query       # [T,d_out]
        keys = x @ self.W_key
        values = x @ self.W_value
        scores = queries @ keys.T        # [T,d_out] @ [d_out,T] -> [T,T]
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        return weights @ values          # 每个 token 得到一个 context 向量


class SelfAttention_v2(nn.Module):
    """与 v1 同一个计算思路；nn.Linear 管理参数和可选 bias。"""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear 是一个模块，所以要“调用”它；不能写 x @ self.W_query。
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        scores = queries @ keys.T
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        return weights @ values


class CausalAttention(nn.Module):
    """在 v2 上加入 batch、未来位置遮挡，以及训练时的 dropout。"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # 主对角线右上方 = 未来 token。buffer 跟随模型移动设备，却不是训练参数。
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,d_in]
        _, num_tokens, _ = x.shape
        if num_tokens > self.mask.shape[0]:
            raise ValueError("输入长度超过 context_length")
        queries = self.W_query(x)                # [B,T,d_out]
        keys = self.W_key(x)
        values = self.W_value(x)
        scores = queries @ keys.transpose(1, 2) # [B,T,T]；只交换最后两维
        scores = scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        # 未来的 score 设为 -inf，经过 softmax 后，对应 weight 恰好为 0。
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        weights = self.dropout(weights)          # eval() 时自动关闭
        return weights @ values                  # [B,T,d_out]


class MultiHeadAttentionWrapper(nn.Module):
    """书中 3.6.1：让多个独立 CausalAttention 并行，再拼接输出。"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # d_out 是“每个头”的输出维度；拼接后是 num_heads*d_out。
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """书中 3.6.2：用三张大 Q/K/V 表一次算完，再按列分给各个头。"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out 必须能被 num_heads 整除")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape                # [B,T,d_in]
        if num_tokens > self.mask.shape[0]:
            raise ValueError("输入长度超过 context_length")
        # 所有头都读完整的输入 x；只是输出的 d_out 个“列”分给不同头。
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim)
        # [B,T,H,Dh] -> [B,H,T,Dh]，以便 H 个头并行计算。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = queries @ keys.transpose(2, 3)  # [B,H,T,Dh] @ [B,H,Dh,T]
        scores = scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        weights = torch.softmax(scores / self.head_dim ** 0.5, dim=-1)
        weights = self.dropout(weights)
        context = weights @ values                # [B,H,T,Dh]
        # 先把同一个 token 的各头结果排在一起，再拼回 d_out 列。
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)             # 学习怎样混合各头的信息
