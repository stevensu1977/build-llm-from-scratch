# 第 5 章：训练与生成

主学习路线在 [chapter05.ipynb](chapter05.ipynb)，可复用函数在 [training.py](training.py)。

1. 滑动窗口给出输入 `[B,T]` 和右移一格的目标 `[B,T]`。
2. GPT 输出 `[B,T,V]`，其中 `V` 个数是每个候选 token 的 **logits**。
3. 交叉熵把每个位置的 logits 与正确 token id 比较，得到 loss。
4. `loss.backward()` 计算梯度；`optimizer.step()` 更新 embedding、Q/K/V、前馈、归一化和输出层等所有可训练参数。
5. 推理时一次选出一个 token，并把它接在输入末尾，再预测下一个。`temperature` 和 `top_k` 改变选择策略，不改变权重。
6. `plot_losses` 画出训练与验证 loss，帮助观察模型是否过拟合。
7. checkpoint 能保存模型权重和优化器状态，供继续训练使用。

Notebook 默认用短文本与小模型练习完整流程。`RUN_BOOK_EXPERIMENT=True` 会运行书中规模的模型训练演示，需要更多内存和时间，建议先确保小模型能运行。书中称 GPT-2 small 为“124M”，但按本书两张**没有共享参数**的词表矩阵计算，这份结构约有 163M 个参数。

[pretrained.py](pretrained.py) 是可选的权重映射练习：用 Hugging Face Transformers 的 GPT-2 权重代替书里的 TensorFlow 下载助手。对应的模型要设 `qkv_bias=True`、`context_length=1024`，并与下载权重的层数、头数和向量维度匹配。首次下载需要网络；Notebook 默认不会下载。下载后可查看每个头在大 Q/K/V 矩阵中对应的行，这些训练后的权重在普通推理中保持固定。
