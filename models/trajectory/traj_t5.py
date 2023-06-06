import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from layers.Layers import FeedForward, LayerNorm, RotaryEmbedding, SelfAttention, CrossAttention
from typing import Optional, Tuple, Dict
from einops import rearrange, einsum
from models.base import Base
import math
from argparse import Namespace
import copy
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError


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
        self.wi_0 = nn.Linear(args.embed_dim, args.d_ff, bias=False)
        self.wi_1 = nn.Linear(args.embed_dim, args.d_ff, bias=False)
        self.wo = nn.Linear(args.d_ff, args.embed_dim, bias=False)
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
        self.layer_norm = T5LayerNorm(args.embed_dim)
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
        self.layer.append(SelfAttention(args))
        if self.is_decoder:
            self.layer.append(CrossAttention(args))

        self.layer.append(T5LayerFF(args))

    def forward(
            self,
            hidden_states,
            key_value=None,):
        hidden_states = self.layer[0](hidden_states)

        if self.is_decoder and key_value is not None:
            hidden_states = self.layer[1](
                hidden_states, key_value)

        hidden_states = self.layer[-1](hidden_states)
        return hidden_states


class T5Stack(nn.Module):
    def __init__(self, args) -> None:
        super(T5Stack, self).__init__()
        self.is_decoder = args.is_decoder
        self.block = nn.ModuleList([T5Block(args)
                                   for i in range(args.num_layers)])
        self.final_layer_norm = T5LayerNorm(args.embed_dim)
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

        self.decoder_start_token = nn.Parameter(torch.rand(1))

        self.ego_embed = nn.Linear(args.ego_attribs, args.embed_dim)
        self.agent_embed = nn.Linear(args.agent_attribs, args.embed_dim)
        # self.map_embed = nn.Linear()
        self.self_atten = SelfAttention(args)
        self.cross_attent = CrossAttention(args)

        encoder_config = copy.deepcopy(args)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(args)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = T5Stack(decoder_config)
        self.ln = LayerNorm(args.embed_dim)
        self.ego_weight = nn.Parameter(torch.rand(
            1, args.num_of_agent, args.embed_dim, args.embed_dim))
        self.agent_weight = nn.Parameter(torch.rand(
            args.num_of_agent, args.num_of_agent, args.embed_dim, args.embed_dim))

        self.ego_output = nn.Linear(args.embed_dim, args.ego_attribs)
        self.agent_output = nn.Linear(args.embed_dim, args.agent_attribs)

        self.metrics = {'mae': MeanAbsoluteError(),
                        'mape': MeanAbsolutePercentageError(),
                        'mse': MeanSquaredError()}

    def shift_right(self, data: torch.Tensor):
        shifted_input_ids = data.new_zeros(data.shape)
        shifted_input_ids[..., 1:] = data[..., :-1].clone()
        shifted_input_ids[..., 0] = self.decoder_start_token
        return shifted_input_ids

    def forward(self,
                ego_input: torch.Tensor,
                agent_input: torch.Tensor,
                ego_feature_output: Optional[torch.Tensor] = None,
                agent_feature_output: Optional[torch.Tensor] = None,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ego_input_embed = self.ego_embed(ego_input)
        agent_input_embed = self.agent_embed(agent_input)

        ego_input_embed = self.self_atten(ego_input_embed)
        agent_input_embed = self.self_atten(agent_input_embed)

        x = self.cross_attent(ego_input_embed, agent_input_embed)

        encoder_output = self.encoder(x)

        ego_out = None
        if ego_feature_output is not None:
            ego_feature_output = self.shift_right(ego_feature_output)
            ego_feature_output = self.ego_embed(ego_feature_output)
            ego_out = self.decoder(ego_feature_output, encoder_output)
            ego_out = self.ln(ego_out)
            ego_out = einsum(ego_out, self.ego_weight,
                             "b n s d, a s d d -> b n a d",)

            # [batch_size x length x  ego_attribs]
            ego_out = self.ego_output(ego_out).squeeze(-2)

        agent_out = None
        if agent_feature_output is not None:
            agent_feature_output = self.shift_right(agent_feature_output)
            agent_feature_output = self.agent_embed(agent_feature_output)
            agent_out = self.decoder(agent_feature_output, encoder_output)
            agent_out = self.ln(agent_out)
            agent_out = einsum(agent_out, self.agent_weight,
                               "b n s d, a s d d -> b n a d",)

            # [batch_size x length x num_of_agent x agent_attribs]
            agent_out = self.agent_output(agent_out)

        return ego_out, agent_out

    def loss(self,
             ego_input: torch.Tensor,
             agent_input: torch.Tensor,
             ego_feature_output: Optional[torch.Tensor] = None,
             agent_feature_output: Optional[torch.Tensor] = None):
        ego_logits, agent_logits, = self.forward(
            ego_input, agent_input, ego_feature_output, agent_feature_output)
        loss = 0.0
        if ego_feature_output is not None and ego_logits is not None:
            ego_loss = F.mse_loss(ego_logits, ego_feature_output)
            loss += ego_loss
        if agent_feature_output is not None and agent_logits is not None:
            agent_loss = F.mse_loss(agent_logits, agent_feature_output)
            loss += agent_loss
        return loss

    def metric(self,
               ego_input: torch.Tensor,
               agent_input: torch.Tensor,
               ego_feature_output: Optional[torch.Tensor] = None,
               agent_feature_output: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        res = {}
        ego_logits, agent_logits = self.forward(
            ego_input, agent_input, ego_feature_output, agent_feature_output)
        for k, m in self.metrics.items():
            m.to(ego_input.device)
            m.update(ego_logits, ego_feature_output)
            m.update(agent_logits, agent_feature_output)
            res[k] = m.compute()

        return res

    def reset(self):
        for k, m in self.metrics.items():
            m.reset()
