"""MiniCPM 4.7 DSpark draft math on the designated CUDA device.

The caller supplies the target's five post-layer hidden states.  The target
model, its expert cache, and speculative verification remain Strata-owned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class MiniCPMDSpark:
    TARGET_LAYERS = (3, 11, 19, 27, 35)

    def __init__(self, draft_dir: Path, target_gguf: Path, gguf_python: Path,
                 device: str = "cuda:0") -> None:
        sys.path.insert(0, str(gguf_python))
        import gguf

        self.gguf = gguf
        self.device = torch.device(device)
        self.config = json.loads((draft_dir / "config.json").read_text(encoding="utf-8"))
        cfg = self.config
        if (cfg.get("architectures") != ["Qwen3DSparkModel"] or
            tuple(cfg["dflash_config"]["target_layer_ids"]) != self.TARGET_LAYERS or
            cfg.get("block_size") != 15 or cfg.get("num_hidden_layers") != 5 or
            cfg.get("hidden_size") != 2048 or cfg.get("vocab_size") != 248144 or
            cfg["dflash_config"].get("markov_head_type") != "vanilla"):
            raise ValueError("MiniCPM 4.7 DSpark checkpoint contract mismatch")
        raw = load_file(str(draft_dir / "model.safetensors"), device="cpu")
        if len(raw) != 62 or "fc.weight" not in raw or "markov_head.markov_w2.weight" not in raw:
            raise ValueError("incomplete DSpark safetensors checkpoint")
        self.weights = {name: value.to(self.device, dtype=torch.float16)
                        for name, value in raw.items()}
        del raw
        self.reader = gguf.GGUFReader(str(target_gguf))
        target_tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.embedding = target_tensors["token_embd.weight"]
        head = target_tensors["output.weight"]
        if (tuple(self.embedding.shape) != (2048, 248144) or
            tuple(head.shape) != (2048, 248144)):
            raise ValueError("target GGUF vocabulary or hidden width differs from DSpark")
        self.head = torch.empty((248144, 2048), dtype=torch.float16, device=self.device)
        for first in range(0, 248144, 1024):
            last = min(first + 1024, 248144)
            rows = gguf.dequantize(head.data[first:last], head.tensor_type)
            self.head[first:last].copy_(torch.from_numpy(rows.astype("float16")))
        self.embedding_rows: dict[int, torch.Tensor] = {}
        self.context_k: list[torch.Tensor | None] = [None] * 5
        self.context_v: list[torch.Tensor | None] = [None] * 5
        self.context_length = 0
        self._rope_inv = 1.0 / (float(cfg["rope_parameters"]["rope_theta"]) **
                                (torch.arange(0, 128, device=self.device,
                                              dtype=torch.float32) / 128.0))

    def _weight(self, name: str) -> torch.Tensor:
        return self.weights[name]

    @staticmethod
    def _norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        return (x.float() * scale * weight.float()).to(x.dtype)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float().unsqueeze(-1) * self._rope_inv
        cos = angles.cos().to(x.dtype).unsqueeze(1)
        sin = angles.sin().to(x.dtype).unsqueeze(1)
        a, b = x[..., :128], x[..., 128:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)

    def _embedding(self, token: int) -> torch.Tensor:
        if token < 0 or token >= 248144:
            raise ValueError("DSpark embedding token outside target vocabulary")
        row = self.embedding_rows.get(token)
        if row is None:
            values = self.gguf.dequantize(self.embedding.data[token:token + 1],
                                          self.embedding.tensor_type)
            row = torch.from_numpy(values.astype("float16")).to(self.device)[0]
            if len(self.embedding_rows) > 64:
                self.embedding_rows.clear()
            self.embedding_rows[token] = row
        return row

    @torch.inference_mode()
    def reset(self) -> None:
        self.context_k = [None] * 5
        self.context_v = [None] * 5
        self.context_length = 0

    @torch.inference_mode()
    def append_context(self, target_hidden: torch.Tensor, first_position: int) -> None:
        """Append [tokens, 5, 2048] target states from layers 3/11/19/27/35."""
        if (target_hidden.ndim != 3 or tuple(target_hidden.shape[1:]) != (5, 2048)
            or first_position != self.context_length):
            raise ValueError("DSpark target state shape or position mismatch")
        count = target_hidden.shape[0]
        if not count:
            return
        states = target_hidden.to(self.device, dtype=torch.float16).reshape(count, 10240)
        projected = F.linear(states, self._weight("fc.weight"))
        normed = self._norm(projected, self._weight("hidden_norm.weight"))
        positions = torch.arange(first_position, first_position + count, device=self.device)
        for layer in range(5):
            prefix = f"layers.{layer}.self_attn."
            k = F.linear(normed, self._weight(prefix + "k_proj.weight"))
            v = F.linear(normed, self._weight(prefix + "v_proj.weight"))
            k = self._norm(k.reshape(count, 2, 256),
                           self._weight(prefix + "k_norm.weight"))
            k = self._rope(k, positions)
            v = v.reshape(count, 2, 256)
            self.context_k[layer] = (k if self.context_k[layer] is None
                                     else torch.cat((self.context_k[layer], k), dim=0))
            self.context_v[layer] = (v if self.context_v[layer] is None
                                     else torch.cat((self.context_v[layer], v), dim=0))
        self.context_length += count

    @torch.inference_mode()
    def draft(self, anchor_token: int, first_position: int,
              n_tokens: int = 15) -> tuple[list[int], list[float]]:
        """Draft a single noncausal query block and its sequential Markov picks."""
        if (first_position != self.context_length or
            n_tokens < 1 or n_tokens > 15):
            raise ValueError("DSpark query does not follow committed context")
        query_ids = [anchor_token] + [self.config["dflash_config"]["mask_token_id"]] * (n_tokens - 1)
        hidden = torch.stack([self._embedding(token) for token in query_ids], dim=0)
        positions = torch.arange(first_position, first_position + n_tokens, device=self.device)
        residual = None
        for layer in range(5):
            prefix = f"layers.{layer}."
            residual = hidden if residual is None else residual + hidden
            normed = self._norm(residual, self._weight(prefix + "input_layernorm.weight"))
            attn_prefix = prefix + "self_attn."
            q = F.linear(normed, self._weight(attn_prefix + "q_proj.weight"))
            k = F.linear(normed, self._weight(attn_prefix + "k_proj.weight"))
            v = F.linear(normed, self._weight(attn_prefix + "v_proj.weight"))
            q = self._norm(q.reshape(n_tokens, 8, 256),
                           self._weight(attn_prefix + "q_norm.weight"))
            k = self._norm(k.reshape(n_tokens, 2, 256),
                           self._weight(attn_prefix + "k_norm.weight"))
            q, k = self._rope(q, positions), self._rope(k, positions)
            v = v.reshape(n_tokens, 2, 256)
            context_k, context_v = self.context_k[layer], self.context_v[layer]
            if context_k is not None:
                k = torch.cat((context_k, k), dim=0)
                v = torch.cat((context_v, v), dim=0)
            # The five checkpoint layers use full, noncausal DFlash attention.
            q = q.permute(1, 0, 2).unsqueeze(0)
            k = k.permute(1, 0, 2).unsqueeze(0).repeat_interleave(4, dim=1)
            v = v.permute(1, 0, 2).unsqueeze(0).repeat_interleave(4, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                  is_causal=False)
            attn = attn.squeeze(0).permute(1, 0, 2).reshape(n_tokens, 2048)
            attn = F.linear(attn, self._weight(attn_prefix + "o_proj.weight"))
            residual = residual + attn
            normed = self._norm(residual, self._weight(prefix + "post_attention_layernorm.weight"))
            gate = F.linear(normed, self._weight(prefix + "mlp.gate_proj.weight"))
            up = F.linear(normed, self._weight(prefix + "mlp.up_proj.weight"))
            hidden = F.linear(F.silu(gate) * up,
                              self._weight(prefix + "mlp.down_proj.weight"))
        hidden = self._norm(residual + hidden, self._weight("norm.weight"))
        base = F.linear(hidden, self.head)
        picks: list[int] = []
        confidence: list[float] = []
        previous = anchor_token
        for index in range(n_tokens):
            markov = self._weight("markov_head.markov_w1.weight")[previous]
            bias = F.linear(markov, self._weight("markov_head.markov_w2.weight"))
            selected = int(torch.argmax(base[index] + bias).item())
            if self.config["dflash_config"].get("enable_confidence_head", False):
                estimate = F.linear(torch.cat((hidden[index], markov)),
                                    self._weight("confidence_head.proj.weight"),
                                    self._weight("confidence_head.proj.bias"))
                confidence.append(float(torch.sigmoid(estimate).item()))
            picks.append(selected)
            previous = selected
        return picks, confidence
