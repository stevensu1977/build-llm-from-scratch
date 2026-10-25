"""第 4 章：把注意力、前馈网络和残差连接组装成 GPT。

张量形状统一写作 [B,T,D]：B 句文本，每句 T 个 token，每个 token D 个数。
每个 TransformerBlock 都返回 [B,T,D]，因此可以逐层堆叠。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from chapter03.attention import MultiHeadAttention


# 书中 GPT-2 small 的结构；未加载已有权重时，它仍只是随机初始化的模型。
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# 用于笔记本电脑上的形状与训练流程练习，而不是高质量语言模型。
TINY_CONFIG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class LayerNorm(nn.Module):
    """把每个 token 的 D 个数调整到均值约 0、方差约 1。

scale、shift 是可学习参数，帮助模型在标准化后调整数值范围。
"""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5  # 防止方差为 0 时除以 0
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x_normalized + self.shift


class GELU(nn.Module):
    """前馈网络中的非线性函数：让网络不只会做线性组合。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
        )


class FeedForward(nn.Module):
    """对每个 token 分别处理：D -> 4D -> D，不在这里混合不同 token。"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """先注意力、再前馈；每一步都把原输入加回来。

若 y=x+F(x)，反向传播时梯度可经过加法直接流回 x，
这使许多 block 叠在一起时更容易训练。
"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差 1：注意力提出的“修改量”加到原 x 上；形状都是 [B,T,D]。
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        # 残差 2：前馈网络提出另一份修改量。
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """decoder-only GPT：输入 token id，输出每个位置对整个词表的预测值。"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg.copy()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 原书与 GPT-2 用的是可学习的绝对位置表，不是固定 sin/cos。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *(TransformerBlock(cfg) for _ in range(cfg["n_layers"]))
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        # in_idx: [B,T]，里面是整数 token id。
        if in_idx.ndim != 2:
            raise ValueError("输入的形状应该是 [batch, token数]")
        _, seq_len = in_idx.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError("输入长度超过位置表的 context_length")
        tok_embeds = self.tok_emb(in_idx)  # [B,T,D]
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids) # [T,D]；加到 [B,T,D] 时自动广播
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)             # 多个 block 逐层改进每个 token 的表示
        x = self.final_norm(x)
        return self.out_head(x)            # [B,T,V]，V 是词表大小


@torch.no_grad()
def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """书中最简单的生成法：每次取最后位置概率最高的一个 token。"""
    if context_size < 1 or context_size > model.cfg["context_length"]:
        raise ValueError("context_size 必须在 1 到模型 context_length 之间")
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    # 只保留模型能看的最近一段
        logits = model(idx_cond)[:, -1, :]   # 只用最后位置预测“下一 token”
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
