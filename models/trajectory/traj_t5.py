import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from layers.Layers import LayerNorm, RotaryEmbedding, SelfAttention, CrossAttention, MoERouterLayer
from typing import Optional, Tuple, Dict
from einops import rearrange, einsum
from models.base import Base
import math
from argparse import Namespace
import copy
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from metric.planning_metrics import AverageDisplacementError, FinalDisplacementError, AverageHeadingError, FinalHeadingError


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.wi_0 = nn.Linear(args.hidden_size, args.d_ff, bias=False)
        self.wi_1 = nn.Linear(args.hidden_size, args.d_ff, bias=False)
        self.wo = nn.Linear(args.d_ff, args.hidden_size, bias=False)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.act = NewGELUActivation()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(args)
        self.layer_norm = T5LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, args):
        super(T5Block, self).__init__()
        self.is_decoder = args.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(SelfAttention(hidden_size=args.hidden_size,
                          num_heads=args.num_heads, split_head=args.split_head))
        if self.is_decoder:
            self.layer.append(CrossAttention(hidden_size=args.hidden_size,
                                             num_heads=args.num_heads, split_head=args.split_head))

        if args.use_moe:
            ffn = T5LayerFF(args)
            moe = MoERouterLayer(num_experts=args.num_experts, topk=args.topk, expert_capacity=args.expert_capacity,
                                 hidden_size=args.hidden_size, router_jitter_noise=args.router_jitter_noise, expert=ffn)
            self.layer.append(moe)
        else:
            ffn = T5LayerFF(args)
            self.layer.append(ffn)

    def forward(
            self,
            hidden_states,
            key_value=None,):
        hidden_states = self.layer[0](hidden_states)

        if self.is_decoder and key_value is not None:
            hidden_states = self.layer[1](
                hidden_states, key_value)

        hidden_states, _ = self.layer[-1](hidden_states)
        return hidden_states


class T5Stack(nn.Module):
    def __init__(self, args) -> None:
        super(T5Stack, self).__init__()
        self.is_decoder = args.is_decoder
        self.block = nn.ModuleList([T5Block(args)
                                   for i in range(args.num_layers)])
        self.final_layer_norm = T5LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self,
                input: torch.Tensor,
                key_value=None,
                ):
        hidden_states = self.dropout(input)

        for i, layer_module in enumerate(self.block):
            if self.is_decoder and key_value is not None:
                key_value = key_value.to(hidden_states.device)
                hidden_states = layer_module(hidden_states, key_value)
            else:
                hidden_states = layer_module(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TrajT5Model(Base):
    def __init__(self, args: Namespace) -> None:
        super(TrajT5Model, self).__init__()

        self.ego_embedding = nn.Linear(
            args.ego_attribs, args.hidden_size, bias=False)
        self.agent_embedding = nn.Linear(
            args.agent_attribs, args.hidden_size, bias=False)

        self.ego_decoder_start_token = nn.Parameter(
            torch.rand(args.ego_attribs))
        self.agents_decoder_start_token = nn.Parameter(
            torch.rand(args.num_of_agents, args.agent_attribs))

        if args.use_lane:
            self.lanes_embedding = nn.Linear(
                args.lanes_attribs, self.hidden_size, bias=False)
            self.num_of_lanes = args.num_of_lanes
        else:
            self.num_of_lanes = 0

        self.num_of_all = self.num_of_lanes + args.num_of_agents

        self.ego_weight = nn.Parameter(torch.ones(
            1, self.num_of_all + 1, args.hidden_size, args.hidden_size))

        self.agent_weight = nn.Parameter(torch.ones(
            args.num_of_agents, self.num_of_all + 1, args.hidden_size, args.hidden_size))

        encoder_config = copy.deepcopy(args)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(args)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = T5Stack(decoder_config)

        self.ln = LayerNorm(args.hidden_size)
        self.ego_output = nn.Linear(args.hidden_size, args.ego_attribs)
        self.agent_output = nn.Linear(args.hidden_size, args.agent_attribs)

        self.metrics = {'mae': MeanAbsoluteError(),
                        'mape': MeanAbsolutePercentageError(),
                        'mse': MeanSquaredError(),
                        'ade': AverageDisplacementError(),
                        'fde': FinalDisplacementError(),
                        'ahe': AverageHeadingError(),
                        'fhe': FinalHeadingError()}

        self.rotary_emb = RotaryEmbedding(args.hidden_size)
        self.register_buffer("pos_emb", None, persistent=False)

    def shift_right(self, data: torch.Tensor, shift_data: torch.Tensor):
        shifted_input_ids = data.new_zeros(data.shape)
        shifted_input_ids[:, 1:, ...] = data[:, :-1, ...].clone()
        shifted_input_ids[:, 0, ...] = shift_data
        return shifted_input_ids

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self,
                ego: torch.Tensor,
                y_ego:  Optional[torch.Tensor] = None,
                agents: Optional[torch.Tensor] = None,
                y_agents: Optional[torch.Tensor] = None,
                lanes: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        b, n, device = ego.shape[0], ego.shape[1],  ego.device
        ego = self.ego_embedding(ego)
        pos_emb = self.get_rotary_embedding(n, device)
        ego = apply_rotary_pos_emb(pos_emb, ego)
        x = ego.unsqueeze(-2)  # [batch_size x length x 1 x dim]

        if agents is not None:
            agents = self.agent_embedding(agents)
            # [batch_size x length x num_of_agents x dim]
            pos_emb = pos_emb.unsqueeze(1)
            agents = apply_rotary_pos_emb(pos_emb, agents)
            x = torch.cat([x, agents], dim=-2)

        if lanes is not None:
            lanes = self.lanes_embedding(lanes)
            # [batch_size x length x num_of_lanes x dim]
            if pos_emb.dim() != 3:
                pos_emb = pos_emb.unsqueeze(1)
            lanes = apply_rotary_pos_emb(pos_emb, lanes)
            x = torch.cat([x, lanes], dim=-2)

        # [batch_size x length x (num_of_lanes +num_of_agents +1) x dim]
        encoder_output = self.encoder(x)

        ego_out = None
        if y_ego is not None and y_agents is not None:
            y_ego = self.shift_right(y_ego, self.ego_decoder_start_token)
            y_ego = self.ego_embedding(y_ego)
            y_ego = y_ego.unsqueeze(-2)
            y_agents = self.shift_right(
                y_agents, self.agents_decoder_start_token)
            y_agents = self.agent_embedding(y_agents)
            x = torch.cat([y_ego, y_agents], dim=-2)

            if lanes is not None:
                n = x.shape[1]
                lanes = lanes[:, -1, ...]
                lanes = self.lanes_embedding(lanes).unsqueeze(1)
                lanes = lanes.repeat(1, n, 1, 1)
                x = torch.cat([x, lanes], dim=-2)
        elif y_ego is not None and y_agents is None:
            y_ego = self.shift_right(y_ego, self.ego_decoder_start_token)
            y_ego = self.ego_embedding(y_ego)
            x = y_ego.unsqueeze(-2)
            if lanes is not None:
                n = x.shape[1]
                lanes = lanes[:, -1, ...]
                lanes = self.lanes_embedding(lanes).unsqueeze(1)
                lanes = lanes.repeat(1, n, 1, 1)
                x = torch.cat([x, lanes], dim=-2)

        ego_out = self.decoder(x, encoder_output)
        output = self.ln(ego_out)

        if agents is None and lanes is None:
            ego_out = self.ego_output(output.squeeze(-2))
        elif agents is not None and lanes is None:
            diff = self.ego_weight.shape[1] - self.num_of_lanes + 1
            ego_out = einsum(
                output, self.ego_weight[:, :diff, :, :], "b n s d, a s d d -> b n a d")
            ego_out = self.ego_output(ego_out).squeeze(-2)
        elif agents is None and lanes is not None:
            diff = self.num_of_lanes + 1
            ego_out = einsum(
                output, self.ego_weight[:, :diff, :, :], "b n s d, a s d d -> b n a d")
            ego_out = self.ego_output(ego_out).squeeze(-2)
        else:
            ego_out = einsum(output, self.ego_weight,
                             "b n s d, a s d d -> b n a d")
            # [batch_size x length x  ego_attribs]
            ego_out = self.ego_output(ego_out).squeeze(-2)

        agent_out = None
        if y_agents is not None and agents is not None:
            if lanes is None:
                diff = self.ego_weight.shape[1] - self.num_of_lanes + 1
                agent_out = einsum(output, self.agent_weight[:, :diff, :, :],
                                   "b n s d, a s d d -> b n a d")
            else:
                agent_out = einsum(output, self.agent_weight,
                                   "b n s d, a s d d -> b n a d")

                # [batch_size x length x num_of_agent x agent_attribs]
            agent_out = self.agent_output(agent_out)

        return ego_out, agent_out

    def loss(self,
             ego: torch.Tensor,
             y_ego:  Optional[torch.Tensor] = None,
             agents: Optional[torch.Tensor] = None,
             y_agents: Optional[torch.Tensor] = None,
             lanes: Optional[torch.Tensor] = None):
        ego_logits, agent_logits = self.forward(
            ego, y_ego, agents, y_agents, lanes)
        loss = 0.0
        if y_ego is not None:
            loss = F.mse_loss(ego_logits, y_ego)
        if agents is not None and y_agents is not None:
            loss += F.mse_loss(agent_logits, y_agents)
        return loss

    def metric(self,
               ego: torch.Tensor,
               y_ego:  Optional[torch.Tensor] = None,
               agents: Optional[torch.Tensor] = None,
               y_agents: Optional[torch.Tensor] = None,
               lanes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        res = {}
        ego_logits, agent_logits = self.forward(
            ego, y_ego, agents, y_agents, lanes)
        for k, m in self.metrics.items():
            m.to(ego.device)
            m.update(ego_logits, y_ego)
            res[k] = m.compute()
        return res

    def reset(self):
        for k, m in self.metrics.items():
            m.reset()
