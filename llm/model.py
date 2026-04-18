from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    mlp_type: str = "relu"  # relu | swiglu
    use_rope: bool = False


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def _build_norm(config: TransformerConfig, dim: int) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(dim)
    return LayerNorm(dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


def _apply_rope(x: torch.Tensor, seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, theta)  # [seq, head_dim/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)[None, None, :, :]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)[None, None, :, :]
    x_rope = (x * cos) + (_rotate_half(x) * sin)
    return x_rope.to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = Linear(config.d_model, config.d_model)
        self.k_proj = Linear(config.d_model, config.d_model)
        self.v_proj = Linear(config.d_model, config.d_model)
        self.out_proj = Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if self.head_dim % 2 == 0 and getattr(self, "use_rope", False):
            q = _apply_rope(q, seq_len, self.head_dim, x.device)
            k = _apply_rope(k, seq_len, self.head_dim, x.device)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v

        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(ctx)


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.mlp_type = config.mlp_type
        if self.mlp_type == "swiglu":
            self.fc1 = Linear(config.d_model, config.d_ff)
            self.fc_gate = Linear(config.d_model, config.d_ff)
            self.fc2 = Linear(config.d_ff, config.d_model)
        else:
            self.fc1 = Linear(config.d_model, config.d_ff)
            self.fc2 = Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            x1 = self.fc1(x)
            xg = self.fc_gate(x)
            silu_x1 = x1 * torch.sigmoid(x1)
            return self.fc2(silu_x1 * xg)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = _build_norm(config, config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.attn.use_rope = config.use_rope
        self.ln2 = _build_norm(config, config.d_model)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        self.pos_embedding = nn.Parameter(torch.empty(config.max_seq_len, config.d_model))
        nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = _build_norm(config, config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Input sequence length exceeds max_seq_len")

        tok = self.token_embedding[input_ids]
        pos = self.pos_embedding[torch.arange(seq_len, device=input_ids.device)]
        x = self.drop(tok + pos.unsqueeze(0))

        for block in self.layers:
            x = block(x)

        x = self.ln_f(x)
        logits = x @ self.token_embedding.t()
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.config.max_seq_len :]
            logits = self(ctx)[:, -1, :]
            if repetition_penalty > 1.0:
                # Penalize logits for tokens already generated in each sample.
                for b in range(out.size(0)):
                    seen = torch.unique(out[b])
                    logits[b, seen] = logits[b, seen] / repetition_penalty
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1).values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask = cumulative_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None and torch.all(next_id.squeeze(-1) == eos_token_id):
                break
        return out
