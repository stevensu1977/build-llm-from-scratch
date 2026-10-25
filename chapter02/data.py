"""第 2 章代码：分词、词表、滑动窗口。

先用正则表达式做教学用分词，再使用 GPT-2 的 tiktoken 分词器。
教学分词器的词表很小，遇到新词只能用 <|unk|>；实际 GPT-2 使用 BPE。
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


VERDICT_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
    "main/ch02/01_main-chapter-code/the-verdict.txt"
)
VERDICT_PATH = Path(__file__).with_name("the-verdict.txt")

# 下载失败时，仍可用这段短文本练习分词与词表。
SAMPLE_TEXT = "The fox runs. The dog runs, too! A fox is quick."

# 括号会保留分隔符；-- 要放在单字符符号前面。
TOKEN_PATTERN = re.compile(r"""(<\|endoftext\|>|<\|unk\|>|--|[,.?_:;!"()']|\s+)""")
SPECIAL_TOKENS = ("<|endoftext|>", "<|unk|>")


def load_verdict(path: Path = VERDICT_PATH) -> str:
    """下载并读入原书使用的《The Verdict》文本；已下载时直接读取。"""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VERDICT_URL, path)
    return path.read_text(encoding="utf-8")


def simple_tokenize(text: str) -> list[str]:
    """把文字拆成小单元；空格只是边界，不作为教学词表里的 token。"""
    pieces = TOKEN_PATTERN.split(text)
    return [piece for piece in pieces if piece and not piece.isspace()]


def build_vocab(text: str) -> dict[str, int]:
    """按固定顺序给 token 编号，再加入两个特殊 token。"""
    words = sorted(set(simple_tokenize(text)) - set(SPECIAL_TOKENS))
    vocab = {token: token_id for token_id, token in enumerate(words)}
    for special in SPECIAL_TOKENS:
        vocab[special] = len(vocab)
    return vocab


class SimpleTokenizerV2:
    """书中简易分词器的学习版；encode 与 decode 互为主要方向的转换。"""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {token_id: token for token, token_id in vocab.items()}
        self.unk_id = vocab["<|unk|>"]

    def encode(self, text: str) -> list[int]:
        return [
            self.str_to_int.get(token, self.unk_id)
            for token in simple_tokenize(text)
        ]

    def decode(self, ids: list[int]) -> str:
        # 简易分词器曾丢掉原始空格，所以无法保证完全还原原文排版。
        text = " ".join(self.int_to_str[token_id] for token_id in ids)
        return re.sub(r"""\s+([,.?_:;!"()'])""", r"\1", text)


class GPTDatasetV1(Dataset):
    """把一长段 token 切成很多个「输入 → 下一个 token」样本。

    max_length 是每个输入样本的 token 数；stride 是窗口每次前进几格。
    target 比 input 向右错开一格，所以两者长度相同。
    """

    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        if max_length < 1 or stride < 1:
            raise ValueError("max_length 和 stride 都必须为正整数")
        token_ids = tokenizer.encode(text)
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        for start in range(0, len(token_ids) - max_length, stride):
            x = token_ids[start : start + max_length]
            y = token_ids[start + 1 : start + max_length + 1]
            self.input_ids.append(torch.tensor(x, dtype=torch.long))
            self.target_ids.append(torch.tensor(y, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer=None,
) -> DataLoader:
    """用 GPT-2 BPE 编码文本，再将多个滑动窗口样本组成 batch。"""
    if tokenizer is None:
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    if len(dataset) == 0:
        raise ValueError("文本太短：至少需要 max_length + 1 个 token")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
