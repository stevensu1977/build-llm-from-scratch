"""检验各章代码中最容易弄错的地方。运行：python -m unittest discover -s tests -v"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from chapter02.data import GPTDatasetV1, SimpleTokenizerV2, build_vocab
from chapter03.attention import CausalAttention, MultiHeadAttention, SelfAttention_v2
from chapter04.gpt import GPTModel
from chapter05.pretrained import copy_hf_gpt2_weights
from chapter05.training import calc_loss_batch, load_checkpoint, save_checkpoint


SMALL_CONFIG = {
    "vocab_size": 32,
    "context_length": 8,
    "emb_dim": 8,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


class CharacterTokenizer:
    """测试滑动窗口用的极简分词器：一个字符就是一个 token。"""

    def encode(self, text):
        return [ord(character) for character in text]


class LearningCodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_special_tokens_and_reverse_vocab(self):
        vocab = build_vocab("A fox runs.")
        tokenizer = SimpleTokenizerV2(vocab)
        ids = tokenizer.encode("A fox <|endoftext|> dragon")
        self.assertEqual(ids[2], vocab["<|endoftext|>"])
        self.assertEqual(ids[3], vocab["<|unk|>"])
        self.assertEqual(tokenizer.int_to_str[ids[2]], "<|endoftext|>")

    def test_sliding_window_targets_are_shifted(self):
        data = GPTDatasetV1("abcdefghij", CharacterTokenizer(), max_length=4, stride=2)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0][0].tolist(), [97, 98, 99, 100])
        self.assertEqual(data[0][1].tolist(), [98, 99, 100, 101])
        self.assertEqual(data[1][0].tolist(), [99, 100, 101, 102])

    def test_attention_shapes_and_future_invariance(self):
        x = torch.randn(2, 4, 3)
        v2 = SelfAttention_v2(3, 4)
        self.assertEqual(tuple(v2(x[0]).shape), (4, 4))
        for model in (
            CausalAttention(3, 4, context_length=4, dropout=0.0),
            MultiHeadAttention(3, 4, context_length=4, dropout=0.0, num_heads=2),
        ):
            model.eval()
            altered = x.clone()
            altered[:, 3, :] += 100.0
            original_output = model(x)
            altered_output = model(altered)
            self.assertEqual(tuple(original_output.shape), (2, 4, 4))
            # 未来 token 变了，前面三个位置的输出应完全不受影响。
            self.assertTrue(torch.allclose(original_output[:, :3], altered_output[:, :3]))

    def test_gpt_loss_reaches_embedding_and_attention(self):
        model = GPTModel(SMALL_CONFIG)
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (2, 5))
        y = torch.randint(0, SMALL_CONFIG["vocab_size"], (2, 5))
        self.assertEqual(tuple(model(x).shape), (2, 5, SMALL_CONFIG["vocab_size"]))
        loss = calc_loss_batch(x, y, model, torch.device("cpu"))
        loss.backward()
        self.assertGreater(model.tok_emb.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.pos_emb.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.trf_blocks[0].att.W_query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.out_head.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_restores_predictions_and_optimizer(self):
        model = GPTModel(SMALL_CONFIG)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 5))
        y = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 5))
        calc_loss_batch(x, y, model, torch.device("cpu")).backward()
        optimizer.step()
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, optimizer, chapter=5)
            restored = GPTModel(SMALL_CONFIG)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            metadata = load_checkpoint(path, restored, restored_optimizer)
            restored.eval()
            self.assertEqual(metadata, {"chapter": 5})
            self.assertTrue(torch.allclose(model(x), restored(x)))
            self.assertEqual(len(optimizer.state), len(restored_optimizer.state))

    def test_pretrained_qkv_conversion(self):
        d, v, t = 8, 32, 8
        state = {
            "transformer.wte.weight": torch.randn(v, d),
            "transformer.wpe.weight": torch.randn(t, d),
            "transformer.ln_f.weight": torch.randn(d),
            "transformer.ln_f.bias": torch.randn(d),
        }
        base = "transformer.h.0"
        for name in ("ln_1", "ln_2"):
            state[f"{base}.{name}.weight"] = torch.randn(d)
            state[f"{base}.{name}.bias"] = torch.randn(d)
        state[f"{base}.attn.c_attn.weight"] = torch.randn(d, 3 * d)
        state[f"{base}.attn.c_attn.bias"] = torch.randn(3 * d)
        state[f"{base}.attn.c_proj.weight"] = torch.randn(d, d)
        state[f"{base}.attn.c_proj.bias"] = torch.randn(d)
        state[f"{base}.mlp.c_fc.weight"] = torch.randn(d, 4 * d)
        state[f"{base}.mlp.c_fc.bias"] = torch.randn(4 * d)
        state[f"{base}.mlp.c_proj.weight"] = torch.randn(4 * d, d)
        state[f"{base}.mlp.c_proj.bias"] = torch.randn(d)
        model = GPTModel(SMALL_CONFIG)
        copy_hf_gpt2_weights(model, state)
        probe = torch.randn(2, d)
        expected_q = (
            probe @ state[f"{base}.attn.c_attn.weight"][:, :d]
            + state[f"{base}.attn.c_attn.bias"][:d]
        )
        actual_q = model.trf_blocks[0].att.W_query(probe)
        self.assertTrue(torch.allclose(expected_q, actual_q))
        self.assertTrue(torch.equal(model.out_head.weight, state["transformer.wte.weight"]))


if __name__ == "__main__":
    unittest.main()
