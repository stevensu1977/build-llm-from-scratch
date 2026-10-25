"""生成第 2～5 章的教学 notebook。修改学习内容后可再次运行本文件。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, text: str) -> dict:
    text = inspect.cleandoc(text) + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def md(text: str) -> dict:
    return cell("markdown", text)


def code(text: str) -> dict:
    return cell("code", text)


SETUP = """
from pathlib import Path
import sys

# notebook 在项目根目录或章节目录中打开时，都能找到我们的 Python 文件。
ROOT = Path.cwd()
if not (ROOT / "chapter02").exists():
    ROOT = ROOT.parent
if not (ROOT / "chapter02").exists():
    raise FileNotFoundError("请从本项目的根目录或章节目录启动 Jupyter")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("项目目录:", ROOT)
"""


CHAPTER02 = [
    md("""
    # 第 2 章：把文本变成模型能使用的数字

    **先动手观察形状，再研究公式。** 这一章依次走过「正则分词 → 词表 → GPT-2 BPE → 滑动窗口 → 两种 embedding」。
    本章教学函数写在 `chapter02/data.py`。把 `USE_VERDICT` 设成 `True` 就会下载书中的 `the-verdict.txt`。
    """),
    code(SETUP),
    md("""
    ## 1. 用正则表达式拆开文字

    正则中的 `|` 表示“或者”；括号保留标点、`--` 和特殊 token。空格用于切分，但简易词表里不保留它。
    这意味着下面的 `decode` 只能大致还原文本，无法恢复原有空格。
    """),
    code("""
    from chapter02.data import (
        SAMPLE_TEXT, load_verdict, simple_tokenize, build_vocab,
        SimpleTokenizerV2, create_dataloader_v1
    )

    USE_VERDICT = False  # 首次练习可改为 True，下载 PDF 使用的完整短篇小说
    raw_text = load_verdict() if USE_VERDICT else SAMPLE_TEXT
    print("字符数:", len(raw_text))
    print("前几个 token:", simple_tokenize(raw_text)[:30])
    """),
    md("""
    ## 2. 给 token 编号，并试试反向查询

    `vocab` 是「token → id」字典；反向字典是「id → token」。
    `<|endoftext|>` 常作为文本边界；`<|unk|>` 代表简易词表没见过的 token。
    简易分词器的局限是：未知词都变成同一个 `<|unk|>`，因此信息会丢失。
    """),
    code("""
    vocab = build_vocab(raw_text)
    tokenizer_simple = SimpleTokenizerV2(vocab)
    ids = tokenizer_simple.encode("The fox runs. <|endoftext|> A dragon!")
    print("词表大小:", len(vocab))
    print("特殊 token 的 id:", {s: vocab[s] for s in ("<|endoftext|>", "<|unk|>")})
    print("编码:", ids)
    print("id 查词:", [tokenizer_simple.int_to_str[i] for i in ids])
    print("解码:", tokenizer_simple.decode(ids))
    """),
    md("""
    ## 3. 换成真正的 GPT-2 分词器

    `tiktoken` 使用 BPE。它可以把中文或新词拆成更小的 token；一个 token 不一定等于一个汉字或一个英文单词。
    `gpt2` 是编码方案名；`gpt-4o` 是**模型名**，可以用 `encoding_for_model` 查到对应方案。不要写成 `gpt4-o`。
    特殊 token 需明确加入 `allowed_special` 才能按一个 token 编码。
    """),
    code("""
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    chinese_ids = tokenizer.encode("狐狸跑过森林。")
    print("GPT-2 编码方案:", tokenizer.name)
    print("中文 token id:", chinese_ids)
    print("解码:", tokenizer.decode(chinese_ids))
    print("gpt-4o 对应方案:", tiktoken.encoding_for_model("gpt-4o").name)
    print("可用方案:", tiktoken.list_encoding_names())
    print("边界 token:", tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    """),
    md("""
    ## 4. 滑动窗口制造训练题目

    假设输入 `[a,b,c,d]`，目标就是 `[b,c,d,e]`。每个位置学会猜**下一个** token。
    `max_length=8` 是**每个样本**的长度；`stride=4` 是窗口每次向右走 4 格。
    这一步为将来训练整个 GPT 做准备，embedding 也会在第 5 章一起更新。
    """),
    code("""
    loader = create_dataloader_v1(
        raw_text, batch_size=2, max_length=8, stride=4,
        shuffle=False, drop_last=False, tokenizer=tokenizer
    )
    batch_x, batch_y = next(iter(loader))
    print("输入形状 [B,T]:", batch_x.shape)
    print("目标形状 [B,T]:", batch_y.shape)
    print("第一条输入:", batch_x[0].tolist())
    print("第一条目标:", batch_y[0].tolist())
    assert batch_x[0, 1:].tolist() == batch_y[0, :-1].tolist()
    """),
    md("""
    ## 5. token embedding + 可学习位置 embedding

    `nn.Embedding` 像一张可训练的查询表。词表大小 `V`、向量宽度 `D`，token 表有 `V×D` 个数。
    位置表长 `L`，有 `L×D` 个数。`P[0]` 创建时是随机数，训练中会变化；训练结束后，通常同一个位置查到相同的向量。

    词向量是 `[B,T,D]`，位置向量是 `[T,D]`。相加时位置向量会沿 batch 方向“复制”给每句话，
    最后一维 `D` 必须相同。PDF 此处使用**可学习的位置表**，不是正弦/余弦公式。
    """),
    code("""
    import torch
    from torch import nn

    D = 8
    context_length = 16  # 位置表能查到 0～15；当前样本 T=8，满足 T<=16
    token_embedding_layer = nn.Embedding(tokenizer.n_vocab, D)
    position_embedding_layer = nn.Embedding(context_length, D)
    token_vectors = token_embedding_layer(batch_x)  # [B,T,D]
    position_vectors = position_embedding_layer(torch.arange(batch_x.shape[1]))  # [T,D]
    input_vectors = token_vectors + position_vectors
    print("token 向量:", token_vectors.shape)
    print("位置向量:", position_vectors.shape)
    print("相加之后:", input_vectors.shape)
    print("位置 0 的向量 P[0]:", position_embedding_layer.weight[0].detach())
    """),
]


CHAPTER03 = [
    md("""
    # 第 3 章：一个 token 怎样“看”前面的 token？

    `Q` 是当前 token 提出的问题，`K` 用来与问题匹配，`V` 是最后取回的信息。
    流程：`Q @ Kᵀ → scores → softmax → weights → weights @ V → context`。
    用 `B` 表示 batch 大小，`T` 表示 token 数，`D` 表示向量宽度。
    """),
    code(SETUP),
    md("""
    ## 1. 简化版：把输入直接当 Q、K、V

    没有 `W_Q/W_K/W_V` 三张可训练表，但依然能根据输入向量算出注意力。
    分数是点积：每个 token 向量与句中每个 token 向量比较。
    分数高，经过 softmax 通常会得到更大权重。
    """),
    code("""
    import torch
    from chapter03.attention import (
        simplified_self_attention, SelfAttention_v1, SelfAttention_v2,
        CausalAttention, MultiHeadAttentionWrapper, MultiHeadAttention
    )

    torch.manual_seed(123)
    x = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])
    z, weights = simplified_self_attention(x)
    print("输入 [T,D]:", x.shape)
    print("权重 [T,T]:", weights.shape)
    print("每一行权重的和:", weights.sum(dim=-1))
    print("上下文向量 [T,D]:", z.shape)
    """),
    md("""
    ## 2. 可训练版本：用三张表学会生成 Q/K/V

    v1 用 `nn.Parameter`，所以写 `x @ W`；v2 用 `nn.Linear`，所以写 `W(x)`。
    `nn.Linear` 本身是模块，不是原始矩阵。它内部的 `.weight` 才是矩阵。
    在 v1/v2 中，`x` 是一整组 token `[T,D]`；因果版本会增加 batch 维。
    """),
    code("""
    v1 = SelfAttention_v1(d_in=3, d_out=4)
    v2 = SelfAttention_v2(d_in=3, d_out=4)
    print("v1 权重形状 [输入,输出]:", v1.W_query.shape)
    print("v2 Linear.weight 形状 [输出,输入]:", v2.W_query.weight.shape)
    print("v1 输出:", v1(x).shape, "v2 输出:", v2(x).shape)

    q = v2.W_query(x)
    k = v2.W_key(x)
    print("Q:", q.shape, "K 转置:", k.T.shape, "scores:", (q @ k.T).shape)
    print("√key_dim =", k.shape[-1] ** 0.5)
    """),
    md("""
    ## 3. 因果注意力：看当前和以前，不能看未来

    `[B,T,T]` 的分数表中，右上角是未来位置。mask 把那里的 score 设为负无穷；
    softmax 后这些位置的权重为 0，于是不会进入当前 token 的 context。
    `register_buffer` 保存这个固定 mask，但它不是可训练参数。dropout 只在 `train()` 模式生效。
    """),
    code("""
    batch = x.unsqueeze(0).repeat(2, 1, 1)  # [B=2,T=6,D=3]
    causal = CausalAttention(3, 4, context_length=6, dropout=0.0)
    q, k = causal.W_query(batch), causal.W_key(batch)
    scores = q @ k.transpose(1, 2)  # batch 中不能用 k.T，后者会翻转全部维度
    masked_scores = scores.masked_fill(causal.mask[:6, :6], -torch.inf)
    causal_weights = torch.softmax(masked_scores / k.shape[-1] ** 0.5, dim=-1)
    print("mask (1 代表未来):\\n", causal.mask.int())
    print("第一句话的注意力权重:\\n", causal_weights[0].round(decimals=3))
    print("输出 [B,T,d_out]:", causal(batch).shape)
    """),
    md("""
    ## 4. 多头：同一份输入，几种不同的关注方式

    `MultiHeadAttentionWrapper` 建造多个独立小头；`MultiHeadAttention` 用大矩阵一次算完再分头。
    在后一种写法中，先得到 `[B,T,d_out]`，再 `view` 为 `[B,T,H,Dh]`，其中 `d_out=H×Dh`。
    **每个头都读完整的原输入**。`view` 只是把投影后的输出列分成几组，不是删去输入 token。
    """),
    code("""
    wrapper = MultiHeadAttentionWrapper(
        d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2
    )
    multi = MultiHeadAttention(
        d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2
    )
    print("独立小头的输出 [B,T,2×2]:", wrapper(batch).shape)
    print("大矩阵分头的输出 [B,T,4]:", multi(batch).shape)
    print("头数:", multi.num_heads, "每头输出宽度:", multi.head_dim)
    print("大 Q 表形状:", multi.W_query.weight.shape)
    """),
    md("""
    `out_proj` 不负责拼接。拼接发生在 `transpose(...).contiguous().view(...)`。
    `out_proj` 是拼接**之后**的线性层，学习如何混合不同头的结果。

    对照自己手打代码，常见拼写错误有 `context_lenght`、`num_token`、`maks_bool`、`ann_scores`。
    分头后 K 的形状是 `[B,H,T,Dh]`，计算 scores 要使用 `keys.transpose(2,3)`。
    """),
]


CHAPTER04 = [
    md("""
    # 第 4 章：把注意力装成完整 GPT

    上一章的多头注意力只是一个零件。本章把它与归一化、前馈网络和残差连接组成
    `TransformerBlock`，再堆叠多个 block，最后由 `out_head` 输出词表预测值。
    """),
    code(SETUP),
    md("""
    ## 1. 归一化、非线性和前馈网络

    LayerNorm 沿一个 token 的向量列做标准化。GELU 让网络能学习非线性变化。
    FFN 的形状是 `D → 4D → D`，它对每个 token 使用同一套参数，不在这一步交换 token 信息。
    """),
    code("""
    import torch
    from chapter04.gpt import (
        LayerNorm, GELU, FeedForward, TransformerBlock,
        GPTModel, TINY_CONFIG, GPT_CONFIG_124M, generate_text_simple
    )

    torch.manual_seed(123)
    cfg = TINY_CONFIG.copy()
    toy = torch.randn(2, 5, cfg["emb_dim"])
    print("LayerNorm:", LayerNorm(cfg["emb_dim"])(toy).shape)
    print("GELU:", GELU()(toy).shape)
    print("FFN:", FeedForward(cfg)(toy).shape)
    """),
    md("""
    ## 2. 为什么写 `x + F(x)`？

    可以把 `F(x)` 想成这一层提出的修改。加法让原信息直接到下一层，
    而在反向传播时也给梯度一条直接路径。下面用最简单的 `F(x)=2x` 看数值：
    `y=x+2x=3x`，所以梯度 `dy/dx=3`。在真实 GPT 中，`F` 是注意力或 FFN。
    """),
    code("""
    small_x = torch.tensor([1.0, 2.0], requires_grad=True)
    small_y = small_x + 2 * small_x
    small_y.sum().backward()
    print("y:", small_y.detach().tolist())
    print("x 的梯度:", small_x.grad.tolist())
    block = TransformerBlock(cfg)
    print("一个 block 的输入与输出:", toy.shape, block(toy).shape)
    """),
    md("""
    ## 3. 堆叠 block，再做词表预测

    token id 的输入形状是 `[B,T]`。两个 embedding 相加后变为 `[B,T,D]`；
    每个 block 仍输出 `[B,T,D]`；`out_head` 最后给每个位置 `V` 个 logits。
    `logits` 是未归一化的预测值，第 5 章用交叉熵训练。
    这个模型只有**因果 self-attention**，因此属于 decoder-only GPT。
    """),
    code("""
    model = GPTModel(cfg)
    input_ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    logits = model(input_ids)
    print("输入:", input_ids.shape, "输出 [B,T,V]:", logits.shape)
    print("小模型参数数:", sum(p.numel() for p in model.parameters()))
    print("原书 GPT-2 small 配置的 block 数:", GPT_CONFIG_124M["n_layers"])
    """),
    md("""
    ## 4. 一次只生成一个 token

    生成时取最后一个位置的 logits，选一个 token id，接在输入后再运行。
    所有 block 逐层加工表示；最终预测由 `out_head` 完成。
    下面模型还没训练，所以输出的 id 没有语言意义。
    """),
    code("""
    model.eval()  # 关闭训练用 dropout
    prompt_ids = input_ids[:1, :3]
    generated_ids = generate_text_simple(
        model, prompt_ids, max_new_tokens=3, context_size=cfg["context_length"]
    )
    print("输入 id:", prompt_ids.tolist())
    print("生成后 id:", generated_ids.tolist())
    """),
    md("""
    **位置方案提醒**：本书与 GPT-2 用 `nn.Embedding` 学每个绝对位置的向量。
    原始 Transformer 论文用固定的正弦/余弦公式。二者都提供位置信息，
    但前者的表项需要训练；后者的数值由公式直接算出。
    """),
]


CHAPTER05 = [
    md("""
    # 第 5 章：训练模型、生成文本、保存权重

    默认使用小模型和短文本，便于在 Mac 上跑完整流程。后面提供 PDF 中较大规模的
    GPT-2 small 训练设置，以及加载已有 GPT-2 权重的可选单元；它们默认关闭。
    """),
    code(SETUP),
    md("""
    ## 1. 准备输入与“右移一格”的答案

    每个输入位置预测它后面的 token。先用短文本反复出现，让小模型能在几步之内看到例子。
    要用原书的 `the-verdict.txt`，看下面 `RUN_BOOK_EXPERIMENT` 的单元。
    """),
    code("""
    import torch
    import tiktoken
    from chapter02.data import SAMPLE_TEXT, load_verdict, create_dataloader_v1
    from chapter04.gpt import GPTModel, TINY_CONFIG, GPT_CONFIG_124M
    from chapter05.training import (
        choose_device, calc_loss_batch, calc_loss_loader, train_model_simple,
        text_to_token_ids, token_ids_to_text, generate, plot_losses,
        save_checkpoint, load_checkpoint
    )

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    text = SAMPLE_TEXT * 80
    split = int(0.9 * len(text))
    train_text, val_text = text[:split], text[split:]
    cfg = TINY_CONFIG.copy()
    cfg["context_length"] = 16
    train_loader = create_dataloader_v1(
        train_text, batch_size=4, max_length=16, stride=16,
        drop_last=True, tokenizer=tokenizer
    )
    val_loader = create_dataloader_v1(
        val_text, batch_size=4, max_length=16, stride=16,
        shuffle=False, drop_last=False, tokenizer=tokenizer
    )
    x, y = next(iter(train_loader))
    print("输入:", x.shape, x[0, :6].tolist())
    print("目标:", y.shape, y[0, :6].tolist())
    """),
    md("""
    ## 2. 从 `[B,T,V]` 算一个 loss

    `V` 是词表大小。把 `B` 和 `T` 合并为 `B×T` 道选择题，
    用交叉熵比较每题的 logits 和正确 token id。loss 越小，表示预测越准确。
    注意力里的 softmax 决定“参考哪些旧 token”；这里的交叉熵评估“下一个 token 猜得好不好”。
    """),
    code("""
    device = choose_device()
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    initial_loss = calc_loss_batch(x, y, model, device)
    print("设备:", device, "训练前 loss:", round(initial_loss.item(), 3))
    print("输入位置数:", x.numel(), "每位置都产生一次预测")
    """),
    md("""
    ## 3. 小模型训练几步

    每一步依次 `zero_grad → forward/loss → backward → optimizer.step`。
    `backward` 把梯度经过输出层、block、注意力一直传回 embedding。
    训练和验证分开算 loss，避免只会背训练文本却不知道效果如何。
    """),
    code("""
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=1, eval_freq=2, eval_iter=1, max_steps=4
    )
    print("训练 loss:", [round(n, 3) for n in train_losses])
    print("验证 loss:", [round(n, 3) for n in val_losses])
    print("已看过的 token 数:", tokens_seen)
    """),
    md("""
    训练 loss 与验证 loss 最好一起看。若训练 loss 降了、验证 loss 却升了，
    模型可能只是在背训练文本。这里仅训练 4 步，图上的点很少，主要用来学习画图方法。
    """),
    code("""
    loss_figure = plot_losses(tokens_seen, train_losses, val_losses)
    loss_figure
    """),
    md("""
    ## 4. 用温度和 top-k 生成

    `temperature=0` 总选最大 logits；正温度会抽样。`top_k=20` 先只保留得分最高的 20 个候选。
    模型只训练了几步，生成文本没有意义是正常的；这个单元要观察的是自回归的代码流程。
    """),
    code("""
    model.eval()
    start = text_to_token_ids("The fox", tokenizer).to(device)
    sampled = generate(
        model, start, max_new_tokens=10,
        context_size=cfg["context_length"], temperature=0.8, top_k=20
    )
    print(token_ids_to_text(sampled.cpu(), tokenizer))
    """),
    md("""
    ## 5. 保存和恢复

    保存的是训练后参数；如果想接着训练，还要保存 AdamW 的状态。
    恢复后在 `eval()` 模式比较同一个输入，两个模型应得到相同 logits。
    """),
    code("""
    checkpoint_path = ROOT / "chapter05" / "checkpoints" / "tiny_gpt.pt"
    save_checkpoint(checkpoint_path, model, optimizer, note="第 5 章小模型练习")
    restored = GPTModel(cfg).to(device)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=4e-4, weight_decay=0.1
    )
    metadata = load_checkpoint(checkpoint_path, restored, restored_optimizer, device)
    restored.eval()
    print("保存位置:", checkpoint_path)
    print("备注:", metadata)
    print("恢复后预测一致:", torch.allclose(model(start), restored(start)))
    """),
    md("""
    ## 6. 可选：按 PDF 的规模训练

    这会读取 `the-verdict.txt`，建立 `context_length=256`、`batch_size=2` 的数据集，
    并训练 GPT-2 small 结构 10 轮。书中叫它 124M，但由于这里输入与输出词表表项没有共享，
    实际可训练参数约 163M。Mac 18GB 可尝试运行，但速度会慢，且短小说易过拟合。
    """),
    code("""
    RUN_BOOK_EXPERIMENT = False
    if RUN_BOOK_EXPERIMENT:
        book_text = load_verdict()
        split = int(0.9 * len(book_text))
        book_cfg = GPT_CONFIG_124M.copy()
        book_cfg["context_length"] = 256
        book_train = create_dataloader_v1(
            book_text[:split], batch_size=2, max_length=256, stride=256,
            drop_last=True, tokenizer=tokenizer
        )
        book_val = create_dataloader_v1(
            book_text[split:], batch_size=2, max_length=256, stride=256,
            shuffle=False, drop_last=False, tokenizer=tokenizer
        )
        book_model = GPTModel(book_cfg).to(device)
        book_optimizer = torch.optim.AdamW(
            book_model.parameters(), lr=4e-4, weight_decay=0.1
        )
        book_train_loss, book_val_loss, book_tokens = train_model_simple(
            book_model, book_train, book_val, book_optimizer, device,
            num_epochs=10, eval_freq=5, eval_iter=5
        )
        print("最后的训练和验证 loss:", book_train_loss[-1], book_val_loss[-1])
    else:
        print("完整规模训练已跳过；小模型流程已在上面运行。")
    """),
    md("""
    ## 7. 可选：加载已经预训练的 GPT-2

    书里从 OpenAI TensorFlow checkpoint 读取权重；本项目改用 Hugging Face Transformers。
    必须使模型配置与 GPT-2 small 匹配：12 层、12 头、宽度 768、位置表 1024，并设 `qkv_bias=True`。
    首次下载需要数百 MB。普通推理中权重固定；若选择微调，权重还会更新。
    """),
    code("""
    RUN_PRETRAINED = False
    if RUN_PRETRAINED:
        from chapter05.pretrained import load_pretrained_gpt2

        pretrained_cfg = GPT_CONFIG_124M.copy()
        pretrained_cfg.update(qkv_bias=True, drop_rate=0.0)
        pretrained = GPTModel(pretrained_cfg)  # 先在 CPU 上复制权重
        load_pretrained_gpt2(pretrained, model_name="gpt2")
        pretrained = pretrained.to(device).eval()
        first_head_q = pretrained.trf_blocks[0].att.W_query.weight[
            :pretrained.trf_blocks[0].att.head_dim
        ]
        print("第 1 层第 1 头的 Q 权重形状:", first_head_q.shape)
        start = text_to_token_ids("Every effort moves you", tokenizer).to(device)
        output = generate(pretrained, start, 20, 1024, temperature=0.8, top_k=40)
        print(token_ids_to_text(output.cpu(), tokenizer))
    else:
        print("预训练权重下载已跳过。")
    """),
]


def write_notebook(chapter: str, cells: list[dict]) -> None:
    # notebook 格式 4.5 起，每个 cell 都需要一个稳定且唯一的 id。
    cells = [{**item, "id": f"{chapter}-{i:02d}"} for i, item in enumerate(cells)]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / chapter / f"{chapter}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    write_notebook("chapter02", CHAPTER02)
    write_notebook("chapter03", CHAPTER03)
    write_notebook("chapter04", CHAPTER04)
    write_notebook("chapter05", CHAPTER05)
