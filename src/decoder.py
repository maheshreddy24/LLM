"""
GOALS: # in order of priority 
    1. RoPE embeddings for PE
    2. Tokenizer
    3. KV Cache
    6. SwiGLU/... activations
    4. Fast Attention
    5. MoE
    7. Instruction Tuning
    10. Quantization
    8. RLHF (Rule based, DPO, PPO, GRPO)
    9. Efficient attention
"""
import os
import torch
import torch.nn as nn
from mha import MultiHeadAttention
import yaml
from types import SimpleNamespace
import math
from icecream import ic
from transformers import AutoTokenizer
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_scale):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_scale * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ffn_scale * d_model, d_model, bias=False),
        )
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, attention_mask, inference=False, **kwargs):
        attn, kv = self.mha(self.norm1(x), attention_mask, inference, **kwargs)
        
        if inference and kwargs.get('kv') is not None:
            x = x[:, -1:, :] + attn
        else:
            x = x + attn

        x = x + self.ffn(self.norm2(x))
        return x, kv


class Decoder(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            self.data = yaml.safe_load(file)

        self.temperature = self.data["temperature"]
        self.context_len = self.data["context_length"]
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.embedding = nn.Embedding(self.data["vocab_size"], self.data["d_model"])
        self.decoders = nn.ModuleList([
            DecoderBlock(self.data["d_model"], self.data["num_heads"], self.data["ffn_scale"])
            for _ in range(self.data["num_decoder_layers"])
        ])
        self.final_norm = nn.RMSNorm(self.data["d_model"])
        self.lm_head = nn.Linear(self.data["d_model"], self.data["vocab_size"], bias=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._kv_cache = []

    def generate(self, input_ids, max_new_tokens=None):
        max_new_tokens = max_new_tokens or (self.context_len - input_ids.shape[1])
        num_layers = self.data["num_decoder_layers"]

        self._kv_cache = [None] * num_layers

        with torch.no_grad():
            for step in range(max_new_tokens):
                assert input_ids.shape[1] <= self.context_len, (
                    # "input_ids already at/over context length; "
                    # "sliding-window KV cache eviction is not implemented"
                    "number of tokens are more than context len"
                )

                if step == 0:
                    x = self.embedding(input_ids)
                    attention_mask = torch.ones(
                        x.shape[0], x.shape[1], device=input_ids.device
                    )
                    for layer_id, decoder in enumerate(self.decoders):
                        x, kv = decoder(
                            x, attention_mask, inference=True, kv=None, cache_len=0
                        )
                        self._kv_cache[layer_id] = kv
                else:
                    new_token = input_ids[:, -1:]
                    x = self.embedding(new_token)
                    attention_mask = torch.ones(x.shape[0], 1, device=input_ids.device)
                    for layer_id, decoder in enumerate(self.decoders):
                        # K_cache.shape[2] is the number of positions already
                        # cached -> the RoPE position offset for the new token
                        cache_len = self._kv_cache[layer_id][0].shape[2]
                        # ic(x.shape)
                        # ic(attention_mask.shape)
                        x, kv = decoder(
                            x, attention_mask, inference=True,
                            kv=self._kv_cache[layer_id], cache_len=cache_len,
                        )
                        self._kv_cache[layer_id] = kv

                x = self.final_norm(x)
                logits = self.lm_head(x)
                next_token_logits = logits[:, -1, :] / self.temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if (next_token == self.tokenizer.eos_token_id).all():
                    break

        return input_ids

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        for decoder in self.decoders:
            x, _ = decoder(x, attention_mask)  # note: forward() returns (x, kv)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, self.data["vocab_size"]),
                shift_labels.view(-1),
            )

        return SimpleNamespace(loss=loss, logits=logits)