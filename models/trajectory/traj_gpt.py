import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Layers import FeedForward, LayerNorm, RotaryEmbedding, MoERouterLayer
from typing import Optional, Dict, Tuple
from einops import rearrange
from models.base import Base
from torch import einsum
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from metric.planning_metrics import AverageDisplacementError, FinalDisplacementError, AverageHeadingError, FinalHeadingError


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class GPTAttention(nn.Module):

    def __init__(self, args, layer_idx=None) -> None:
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.embed_dim = args.hidden_size
        self.split_head = args.split_head
        if self.split_head:
            self.num_heads = args.num_heads
            self.head_dim = self.embed_dim // self.num_heads
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
        if self.split_head:
            query = rearrange(
                query, "b n ... (h d) -> b h ... n d", h=self.num_heads)
            key = rearrange(
                key, "b n ... (h d) -> b h ... n d", h=self.num_heads)
            value = rearrange(
                value, "b n ... (h d) -> b h ... n d", h=self.num_heads)
        else:
            query = rearrange(query, "b n ... h d -> b h ... n d")
            key = rearrange(key, "b n ... h d -> b h ... n d")
            value = rearrange(value, "b n ... h d -> b h ... n d")

        attn_output, attn_weights = self.attn(
            query, key, value, attention_mask)
        if self.split_head:
            attn_output = rearrange(
                attn_output, "b h ... n d -> b n ... (h d)")
        else:
            attn_output = rearrange(attn_output, "b h ... n d -> b n ... h d")
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
        self.ffn = FeedForward(hidden_size,
                               dropout=args.ff_pdrop, glu=True)
        if args.use_moe:
            self.ffn = MoERouterLayer(num_experts=args.num_experts, topk=args.topk, expert_capacity=args.expert_capacity,
                                      hidden_size=hidden_size, router_jitter_noise=args.router_jitter_noise, expert=self.ffn)

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
        feed_forward_hidden_states, _ = self.ffn(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states, attn_weights)
        return outputs


class TrajGPT(Base):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.embed_dim = args.hidden_size
        self.ego_embedding = nn.Linear(
            args.ego_attribs, self.embed_dim, bias=False)
        self.agent_embedding = nn.Linear(
            args.agent_attribs, self.embed_dim, bias=False)

        if args.use_lane:
            self.lanes_embedding = nn.Linear(
                args.num_of_lane_path_point * 3 + args.lanes_attribs, self.embed_dim, bias=False)
            self.num_of_lanes = args.num_of_lanes
        else:
            self.num_of_lanes = 0

        self.num_of_all = self.num_of_lanes + args.num_of_agents
        self.ego_weight = nn.Parameter(torch.ones(
            1, self.num_of_all + 1, args.hidden_size, args.hidden_size))

        self.agent_weight = nn.Parameter(torch.ones(
            args.num_of_agents, self.num_of_all + 1, args.hidden_size, args.hidden_size))

        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = nn.ModuleList([Block(args, layer_idx=i)
                               for i in range(args.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(self.embed_dim)
        self.ln = LayerNorm(self.embed_dim)
        self.ego_proj = nn.Linear(args.hidden_size, args.ego_attribs)
        self.agent_proj = nn.Linear(args.hidden_size, args.agent_attribs)

        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("mask", None, persistent=False)
        self.metrics = {'mae': MeanAbsoluteError(),
                        'mape': MeanAbsolutePercentageError(),
                        'mse': MeanSquaredError(),
                        'ade': AverageDisplacementError(),
                        'fde': FinalDisplacementError(),
                        'ahe': AverageHeadingError(),
                        'fhe': FinalHeadingError()}

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def get_mask(self, n: int, device):
        if self.mask is not None:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self,
                ego: torch.Tensor,
                agents: Optional[torch.Tensor] = None,
                lanes: Optional[torch.Tensor] = None):

        # x shape is [batch_size x length x 7]
        b, n, device = ego.shape[0], ego.shape[1],  ego.device

        ego = self.ego_embedding(ego)  # [batch_size x length x dim]
        pos_emb = self.get_rotary_embedding(n, device)
        ego = apply_rotary_pos_emb(pos_emb, ego)
        x = ego.unsqueeze(-2)  # [batch_size x length x 1 x dim]

        if agents is not None:
            # [batch_size x length x num_of_agents x dim]
            agents = self.agent_embedding(agents)
            pos_emb = pos_emb.unsqueeze(1)
            agents = apply_rotary_pos_emb(pos_emb, agents)
            x = torch.cat([x, agents], dim=-2)

        if lanes is not None:
            lanes = lanes.view(b, n, self.num_of_lanes, -1)
            # [batch_size x length x num_of_lanes x dim]
            lanes = self.lanes_embedding(lanes)
            lanes = apply_rotary_pos_emb(pos_emb, lanes)
            x = torch.cat([x, lanes], dim=-2)

        x = self.drop(x)
        hidden_states = x
        attention_mask = self.get_mask(n, device)
        atten_outputs = []
        for i, block in enumerate(self.h):
            hidden_states, attn_outputs = block(
                hidden_states=hidden_states, attention_mask=attention_mask)
            atten_outputs.append(attn_outputs)
        x = self.ln(hidden_states)

        # [batch_size x length x (num_of_lanes + num_of_agents +1) x dim]
        if agents is None and lanes is None:
            ego_out = self.ego_proj(x.squeeze(-2))
            angent_out = None
        elif agents is not None and lanes is None:
            diff = self.ego_weight.shape[2] - self.num_of_lanes
            ego_out = einsum("b n s d, a s d d -> b n a d",
                             x, self.ego_weight[:, :, :diff, :])
            angent_out = einsum(
                "b n s d, a s d d -> b n a d", x, self.agent_weight[:, :, :diff, :])
        elif agents is None and lanes is not None:
            diff = self.num_of_lanes + 1
            ego_out = einsum("b n s d, a s d d -> b n a d",
                             x, self.ego_weight[:, :, :diff, :])
            angent_out = None
        else:
            ego_out = einsum("b n s d, a s d d -> b n a d", x, self.ego_weight)
            angent_out = einsum(
                "b n s d, a s d d -> b n a d", x, self.agent_weight)
            angent_out = self.agent_proj(angent_out)
        return ego_out, angent_out

    def loss(self,
             ego: torch.Tensor,
             agents: Optional[torch.Tensor] = None,
             lanes: Optional[torch.Tensor] = None,
             y_ego: Optional[torch.Tensor] = None,
             y_agents: Optional[torch.Tensor] = None):

        ego_logits, agents_logits = self.forward(ego, agents, lanes)
        loss = F.mse_loss(ego_logits, y_ego)
        if y_agents is not None and agents_logits is not None:
            loss += F.mse_loss(agents_logits, y_agents)
        return loss

    def metric(self,
               ego: torch.Tensor,
               agents: Optional[torch.Tensor] = None,
               lanes: Optional[torch.Tensor] = None,
               y_ego: Optional[torch.Tensor] = None,
               y_agents: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        res = {}
        logits, _ = self.forward(ego, agents, lanes)
        for k, m in self.metrics.items():
            m.to(ego.device)
            m.update(logits, y_ego)
            res[k] = m.compute()

        return res

    def reset(self):
        for k, m in self.metrics.items():
            m.reset()
