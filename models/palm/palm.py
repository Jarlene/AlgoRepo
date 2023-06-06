import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from models.base import Base
from layers.Layers import LayerNorm, Residual, RotaryEmbedding, FeedForward


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        self.fused_dims = (attn_inner_dim, dim_head,
                           dim_head)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(
            dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = FeedForward(dim=dim, swish=True, glu=True, mult=ff_mult)

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(x)


class PaLM(Base):
    def __init__(self, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4, **kwargs) -> None:
        super(PaLM, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Embedding(num_tokens, dim),
            *[
                Residual(ParallelTransformerBlock(
                    dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult))
                for _ in range(depth)
            ],
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )
        self.net[-1].weight = self.net[0].weight

        nn.init.normal_(self.net[0].weight, std=0.02)

    def forward(self, x):
        logits = self.net(x)
        return logits

    def loss(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(rearrange(logits, "b c n -> b n c"), y)

    def metric(self, x, y, **kwargs):
        return {'loss': self.loss(x, y)}
