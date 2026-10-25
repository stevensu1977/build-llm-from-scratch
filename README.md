# 从 PDF 逐章学习 GPT

这份项目按 Sebastian Raschka 的《Build a Large Language Model (From Scratch)》第 1～5 章整理。代码沿用书中的学习顺序，并加入适合高中生阅读的中文注释、张量形状和少量验证。原书的例子与本项目的教学改写可能在变量名或示例数字上略有不同。

你手打的 `BuildLLMFromScratch.ipynb` 保留在项目根目录；下面的章节文件是单独的学习版本。

| 目录 | 学习内容 | 主要文件 |
| --- | --- | --- |
| `chapter01` | GPT 整体流程；原书本章主要是概念 | `README.md` |
| `chapter02` | 分词、词表、滑动窗口、token 与位置 embedding | `data.py`、`chapter02.ipynb` |
| `chapter03` | 简化版、可训练版、因果、多头注意力 | `attention.py`、`chapter03.ipynb` |
| `chapter04` | LayerNorm、GELU、FFN、残差、TransformerBlock、GPTModel | `gpt.py`、`chapter04.ipynb` |
| `chapter05` | loss、训练、采样、保存与加载已有 GPT-2 权重 | `training.py`、`pretrained.py`、`chapter05.ipynb` |

## 在 Mac 上开始

推荐 Python 3.11 或 3.12。先在项目根目录安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

按照章节顺序打开 notebook。第 2 章会从原书仓库下载 `the-verdict.txt`；如果当前没有网络，可以先用 notebook 中的短文本示例阅读分词代码。

第 5 章加载已有 GPT-2 权重的可选练习需要额外安装：

```bash
python -m pip install -r requirements-pretrained.txt
```

书中第 5 章的主要训练设置是 `context_length=256`、`batch_size=2`。在 Mac 上如需使用 GPU，代码会优先选择 PyTorch 的 `mps`，否则使用 CPU。先运行小模型的练习，再把第 5 章 notebook 中的 `RUN_BOOK_EXPERIMENT` 改成 `True` 来尝试书中的完整演示。

## 阅读代码时固定问三件事

1. 输入和输出的形状是什么？
2. 哪些是训练后保留的参数，哪些只是本次输入的中间结果？
3. 这一步对下一步预测有什么作用？

## 验证

在安装 PyTorch 后，可从项目根目录运行：

```bash
python -m unittest discover -s tests -v
```

测试只覆盖形状、因果限制、梯度和保存加载等关键风险；它不是对模型生成质量的评估。
