"""安装 requirements-pretrained.txt 后，检验本项目 GPT 与 HF GPT-2 的前向结果。"""

import unittest

import torch

from chapter04.gpt import GPTModel
from chapter05.pretrained import copy_hf_gpt2_weights

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipIf(GPT2LMHeadModel is None, "未安装可选的 Transformers")
class PretrainedMappingTests(unittest.TestCase):
    def test_random_hf_gpt2_matches_our_gpt(self):
        # 不下载真实权重。两种实现载入同一组随机参数后，预测应相同。
        torch.manual_seed(123)
        hf_cfg = GPT2Config(
            vocab_size=64, n_positions=16, n_ctx=16, n_embd=8, n_layer=2, n_head=2,
            embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
            activation_function="gelu_new",
        )
        hf_model = GPT2LMHeadModel(hf_cfg).eval()
        our_cfg = {
            "vocab_size": 64, "context_length": 16, "emb_dim": 8,
            "n_heads": 2, "n_layers": 2, "drop_rate": 0.0, "qkv_bias": True,
        }
        our_model = GPTModel(our_cfg).eval()
        copy_hf_gpt2_weights(our_model, hf_model.state_dict())
        ids = torch.tensor([[4, 8, 16, 32]])
        with torch.no_grad():
            hf_logits = hf_model(ids).logits
            our_logits = our_model(ids)
        self.assertTrue(torch.allclose(hf_logits, our_logits, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
