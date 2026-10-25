"""第 5 章可选练习：把 Hugging Face 上的 GPT-2 权重搬入第 4 章的模型。

原书使用 TensorFlow checkpoint 和 gpt_download.py；这里换成更容易安装的
Transformers 权重来源。搬运前必须让结构与已有 GPT-2 权重匹配。
"""

from __future__ import annotations

import torch


@torch.no_grad()
def copy_hf_gpt2_weights(model, state: dict[str, torch.Tensor]) -> None:
    """从 GPT2LMHeadModel.state_dict() 复制小型 GPT-2 的全部主要权重。

HF 的 Conv1D.weight 存为 [输入列,输出列]；我们用 nn.Linear，
它的 weight 是 [输出列,输入列]，因此这些矩阵需要转置。
"""

    def copy(target: torch.Tensor, name: str, transpose: bool = False) -> None:
        source = state[name].T if transpose else state[name]
        if target.shape != source.shape:
            raise ValueError(f"{name} 形状不匹配：{tuple(source.shape)} -> {tuple(target.shape)}")
        target.copy_(source.to(device=target.device, dtype=target.dtype))

    copy(model.tok_emb.weight, "transformer.wte.weight")
    copy(model.pos_emb.weight, "transformer.wpe.weight")
    for i, block in enumerate(model.trf_blocks):
        base = f"transformer.h.{i}"
        copy(block.norm1.scale, f"{base}.ln_1.weight")
        copy(block.norm1.shift, f"{base}.ln_1.bias")
        copy(block.norm2.scale, f"{base}.ln_2.weight")
        copy(block.norm2.shift, f"{base}.ln_2.bias")

        # HF 把 Q、K、V 连在 c_attn 一张大表中；切成三段装到书中的三层。
        qkv_weight = state[f"{base}.attn.c_attn.weight"].chunk(3, dim=1)
        qkv_bias = state[f"{base}.attn.c_attn.bias"].chunk(3, dim=0)
        for layer, weight, bias in zip(
            (block.att.W_query, block.att.W_key, block.att.W_value),
            qkv_weight,
            qkv_bias,
        ):
            if layer.bias is None:
                raise ValueError("加载 GPT-2 权重时，请把 qkv_bias 设置为 True")
            layer.weight.copy_(weight.T.to(layer.weight.device, layer.weight.dtype))
            layer.bias.copy_(bias.to(layer.bias.device, layer.bias.dtype))
        copy(block.att.out_proj.weight, f"{base}.attn.c_proj.weight", transpose=True)
        copy(block.att.out_proj.bias, f"{base}.attn.c_proj.bias")
        copy(block.ff.layers[0].weight, f"{base}.mlp.c_fc.weight", transpose=True)
        copy(block.ff.layers[0].bias, f"{base}.mlp.c_fc.bias")
        copy(block.ff.layers[2].weight, f"{base}.mlp.c_proj.weight", transpose=True)
        copy(block.ff.layers[2].bias, f"{base}.mlp.c_proj.bias")
    copy(model.final_norm.scale, "transformer.ln_f.weight")
    copy(model.final_norm.shift, "transformer.ln_f.bias")
    # GPT-2 原模型把输出层与 token embedding 共用一张表。
    # 本书模型的两张表没有共享参数，但复制后其初始数值相同。
    copy(model.out_head.weight, "transformer.wte.weight")


def load_pretrained_gpt2(model, model_name: str = "gpt2") -> None:
    """首次调用会从 Hugging Face 下载权重；通常需要数百 MB 空间。"""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        raise ImportError("先运行 pip install -r requirements-pretrained.txt") from exc
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    if model.cfg["n_layers"] != hf_model.config.n_layer:
        raise ValueError("模型层数与所选 GPT-2 权重不一致")
    if model.cfg["n_heads"] != hf_model.config.n_head:
        raise ValueError("注意力头数与所选 GPT-2 权重不一致")
    copy_hf_gpt2_weights(model, hf_model.state_dict())
    model.eval()
