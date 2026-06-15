import math
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch


@dataclass
class GPT_Config:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool


class GPT_Model(nn.Module):
    def __init__(self, config: GPT_Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    self.config.vocab_size, self.config.n_embd
                ),  # (context_widow,embd) #Token table
                wpe=nn.Embedding(
                    self.config.block_size, self.config.n_embd
                ),  # (context_widow,embd) #Position table
                dropout=nn.Dropout(self.config.dropout),
                heads=nn.ModuleList(
                    Block(config=config) for _ in range(config.n_layer)
                ),  # List of layers for our transformer
                norm_layer=nn.LayerNorm(self.config.n_embd, bias=self.config.bias),
            )
        )
        self.lm_layer = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )  # Projects our embeddings to tokenize vocab to then run softmax
        self.lm_layer.weight = self.transformer.wte.weight

        # Init weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("oproj.weight") or name.endswith("proj_layer.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        B, T = tokens.size()
        assert T <= self.config.block_size, (
            f"sequence {T} > block_size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)

        # Lookup embeddings and add
        tok_emb = self.transformer.wte(tokens)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.dropout(tok_emb + pos_emb)

        for block in self.transformer.heads:
            x = block(x)

        # final norm
        x = self.transformer.norm_layer(x)

        if targets is not None:  # Fwd pass during trainign
            logits = self.lm_layer(x)  # (B,T,vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:  # Fwd during inference
            logits = self.lm_layer(x[:, [-1], :])
            loss = None
        return logits, loss

    def num_params(self, non_embedding=True):
        """Total parameter count."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n


""" Mutlihead Self-Attention layer"""


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.oproj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

    def forward(self, x):
        B, T, d = x.size()
        head_dim = d // self.n_head
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B,H,T,d)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B,H,T,d)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B,H,T,d)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, d)
        )  # (B,T,d) #For notes: B is batch size(# of sequeces), T is context length or blocksize, d is embedding dimensions of each token
        return self.oproj(y)


"""MLP Block with 2 layers and GELU activation"""


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc_layer = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj_layer = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, d = x.size()
        x = self.fc_layer(x)
        x = self.gelu(x)
        x = self.proj_layer(x)
        x = self.dropout(x)
        return x


"""Attention and MLP block representing 1 layer"""


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Forward residuals
        x = x + self.mlp(self.ln_2(x))
        return x
