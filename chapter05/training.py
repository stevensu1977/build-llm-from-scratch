"""第 5 章：预测误差、训练循环、采样和 checkpoint。

训练目标是：给前面的 token，预测右移一格后的 token。
注意力中的 softmax 用来决定“看谁”；这里的交叉熵用于评估“下一个词猜得如何”。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F


def choose_device() -> torch.device:
    """优先使用 NVIDIA CUDA 或 Mac 的 MPS；否则使用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """tiktoken 把文本转为一条 batch：[1,T]。"""
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """把一条 batch 的整数 id 转回可读文本。"""
    if token_ids.shape[0] != 1:
        raise ValueError("这里只演示一条文本，请传入 [1,T]")
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """一个 batch 的交叉熵；返回单个标量，可直接调用 backward()。"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)        # [B,T,V]，还不是概率
    # 把 B 和 T 合起来：每个位置都是一条“预测下一个 token”的练习题。
    return F.cross_entropy(
        logits.flatten(0, 1),         # [B*T,V]
        target_batch.flatten(),        # [B*T]
    )


@torch.no_grad()
def calc_loss_loader(
    data_loader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """在若干个 batch 上求平均 loss。调用时应先让模型进入 eval 模式。"""
    if num_batches is not None and num_batches < 1:
        raise ValueError("num_batches 必须为正整数")
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    if limit == 0:
        raise ValueError("DataLoader 没有 batch；检查文本长度或 drop_last")
    total = 0.0
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= limit:
            break
        total += calc_loss_batch(inputs, targets, model, device).item()
    return total / limit


def evaluate_model(model, train_loader, val_loader, device, eval_iter: int):
    """训练与验证都用 eval 模式测量，并恢复原来的模式。"""
    was_training = model.training
    model.eval()
    try:
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    finally:
        model.train(was_training)
    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int = 5,
    eval_iter: int = 1,
    max_steps: int | None = None,
):
    """书中训练循环的学习版；max_steps 可给笔记本电脑设置练习上限。

一次 step：预测 → 算 loss → backward 算梯度 → optimizer.step 更新参数。
返回三个等长列表：训练 loss、验证 loss、训练时已看过的 token 数。
"""
    if num_epochs < 1 or eval_freq < 1 or eval_iter < 1:
        raise ValueError("num_epochs、eval_freq、eval_iter 必须为正整数")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps 必须为正整数")
    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens = 0
    global_step = -1
    model.train()
    for _ in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)  # 清除上一步的梯度
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()                        # 链式法则把梯度传到全部可训练参数
            optimizer.step()                        # AdamW 依据梯度修改参数
            total_tokens += inputs.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)
            if max_steps is not None and global_step + 1 >= max_steps:
                return train_losses, val_losses, tokens_seen
    return train_losses, val_losses, tokens_seen


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """每次生成一个 token；temperature>0 时随机抽样，0 时选最大值。

top_k 先只留下最有希望的 k 个候选；temperature 再调节随机程度。
推理前调用 model.eval()，关闭训练用 dropout。
"""
    if context_size < 1:
        raise ValueError("context_size 必须为正整数")
    if hasattr(model, "cfg") and context_size > model.cfg["context_length"]:
        raise ValueError("context_size 超过模型的位置表长度")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k 必须为正整数")
    if temperature < 0:
        raise ValueError("temperature 不能小于 0")
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]  # [B,V]
        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            cutoff = torch.topk(logits, k).values[:, -1, None]
            logits = torch.where(logits < cutoff, -torch.inf, logits)
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool(torch.all(idx_next == eos_id)):
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def plot_losses(
    tokens_seen: list[int],
    train_losses: list[float],
    val_losses: list[float],
):
    """画出训练与验证 loss，方便观察是否只在记忆训练文本。"""
    if not (len(tokens_seen) == len(train_losses) == len(val_losses)):
        raise ValueError("tokens_seen、train_losses、val_losses 长度必须一致")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tokens_seen, train_losses, label="Train")
    ax.plot(tokens_seen, val_losses, label="Validation", linestyle="--")
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    return fig


def save_checkpoint(path: str | Path, model, optimizer=None, **metadata) -> None:
    """保存模型权重，也可保存优化器状态以便继续训练。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    device: torch.device | str = "cpu",
) -> dict:
    """恢复权重；如传入 optimizer，也恢复它过去的更新状态。"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("metadata", {})
