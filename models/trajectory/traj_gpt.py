import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Layers import FeedForward, LayerNorm, RotaryEmbedding
from typing import Optional, Tuple, Dict
from einops import rearrange
from models.base import Base
from torch import einsum


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


def convert_head_mask_to_5d(head_mask, num_hidden_layers):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(
            0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        # We can specify head_mask for each layer
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    assert head_mask.dim(
    ) == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    return head_mask


class GPTAttention(nn.Module):

    def __init__(self, args, is_cross_attention=False, layer_idx=None) -> None:
        super().__init__()

        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.embed_dim = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.is_cross_attention:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.fused_proj = nn.Linear(
                self.embed_dim, self.embed_dim * 2, bias=False)
        else:
            self.fused_proj = nn.Linear(
                self.embed_dim, self.embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((args.max_length, args.max_length), dtype=torch.bool)).view(
                1, 1, args.max_length, args.max_length
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / \
            torch.full([], value.size(-1) ** 0.5,
                       dtype=attn_weights.dtype, device=attn_weights.device)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length -
                                    query_length: key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full(
                [], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None):
        query, key, value = self.fused_proj(
            hidden_states).split(self.embed_dim, dim=-1)

        query = rearrange(query, "b n s (h d) -> b h s n d", h=self.num_heads)
        key = rearrange(key, "b n s (h d) -> b h s n d", h=self.num_heads)
        value = rearrange(value, "b n s (h d) -> b h s n d", h=self.num_heads)

        attn_output, attn_weights = self.attn(
            query, key, value, attention_mask)
        attn_output = rearrange(attn_output, "b h s n d -> b n s (h d)")
        attn_output = self.proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights


class Block(nn.Module):
    def __init__(self, args, layer_idx=None) -> None:
        super().__init__()

        hidden_size = args.hidden_size
        self.ln_1 = LayerNorm(hidden_size)
        self.atten = GPTAttention(args, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, hidden_size,
                              swish=True, dropout=args.ff_pdrop, glu=True)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs, attn_weights = self.atten(
            hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = attn_outputs + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.ff(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states, attn_weights)
        return outputs


class TrajGPT(Base):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.embed_dim = args.hidden_size
        self.ego_embedding = nn.Linear(7, self.embed_dim)
        self.agent_embedding = nn.Linear(8, self.embed_dim)
        self.num_of_agent = args.num_of_agent

        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = nn.ModuleList([Block(args, layer_idx=i)
                               for i in range(args.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(self.embed_dim)
        self.ln = LayerNorm(self.embed_dim)
        self.ego_weight = nn.Parameter(torch.rand(
            1, self.num_of_agent, args.hidden_size, args.hidden_size))
        self.agent_weight = nn.Parameter(torch.rand(
            self.num_of_agent - 1, self.num_of_agent, args.hidden_size, args.hidden_size))
        self.ego_output = nn.Linear(args.hidden_size, 7)
        self.agent_output = nn.Linear(args.hidden_size, 8)

        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("ego_weight", self.ego_weight, persistent=True)
        self.register_buffer(
            "agent_weight", self.agent_weight, persistent=True)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def get_mask(self, n: int, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self,
                ego_x: torch.Tensor,
                agent_x: torch.Tensor):

        n = ego_x.shape[-2]
        b = ego_x.shape[0]

        device = ego_x.device
        # ego_x shape is [batch_size x length x 7]

        ego = self.ego_embedding(ego_x)
        pos_emb = self.get_rotary_embedding(n, ego_x.device)
        ego = apply_rotary_pos_emb(pos_emb, ego)  # [batch_size x length x dim]
        ego = ego.unsqueeze(-2)  # [batch_size x length x 1 x dim]

        # agent shape is [batch_size x length x num_agent x 8]
        agent = self.agent_embedding(agent_x)
        pos_emb = pos_emb.unsqueeze(1)
        # [batch_size x length x num_agent x dim]
        agent = apply_rotary_pos_emb(pos_emb, agent)

        x = torch.cat([ego, agent], dim=-2)
        x = self.drop(x)
        hidden_states = x
        attention_mask = self.get_mask(n, device)
        atten_outputs = []
        for i, block in enumerate(self.h):
            hidden_states, attn_outputs = block(
                hidden_states=hidden_states, attention_mask=attention_mask)
            atten_outputs.append(attn_outputs)

        x = self.ln(hidden_states)
        ego_ouput = einsum("b n s d, a s d d -> b n a d", x, self.ego_weight)
        ego_ouput = self.ego_output(ego_ouput)  # [batch_size x length x 1 x 7]
        agent_output = einsum(
            "b n s d, a s d d -> b n a d", x, self.agent_weight)
        agent_output = self.agent_output(agent_output)
        return ego_ouput, agent_output, atten_outputs

    def loss(self, feature, target):
        ego_x, agent_x = feature[0], feature[1]
        ego_y, agent_y = target[0], target[1]
        ego_logits, agent_logits, _ = self.forward(ego_x, agent_x)
        ego_logits = ego_logits.squeeze(-2)
        ego_loss = F.mse_loss(ego_logits, ego_y)
        agent_loss = F.mse_loss(agent_logits, agent_y)
        return ego_loss + agent_loss
