
import torch
import torch.nn as nn
from argparse import Namespace
from layers.Layers import DualAttenion, Transpose


class CARD(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super(CARD, self).__init__()
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.embed_dim = args.embed_dim
        self.task_name = args.task_name
        patch_num = int((args.seq_len - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.pos_embed = nn.Parameter(
            torch.randn(patch_num, args.embed_dim)*1e-2)
        self.total_token_number = self.patch_num + 1
        args.total_token_number = self.total_token_number

        # embeding layer related
        self.input_projection = nn.Linear(self.patch_len, args.embed_dim)
        self.input_dropout = nn.Dropout(args.dropout)
        self.cls = nn.Parameter(torch.randn(1, args.embed_dim)*1e-2)

        # mlp decoder
        self.out_proj = nn.Linear(
            (patch_num+1+self.model_token_number)*args.embed_dim, args.pred_len)

        # dual attention encoder related
        self.Attentions_over_token = nn.ModuleList(
            [DualAttenion(args) for i in range(args.hiden_layer_num)])
        self.Attentions_over_channel = nn.ModuleList(
            [DualAttenion(args, over_channel=True) for i in range(args.hiden_layer_num)])
        self.Attentions_mlp = nn.ModuleList(
            [nn.Linear(args.embed_dim, args.embed_dim) for i in range(args.hiden_layer_num)])
        self.Attentions_dropout = nn.ModuleList(
            [nn.Dropout(args.dropout) for i in range(args.hiden_layer_num)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1, 2),
                                                            nn.BatchNorm1d(args.embed_dim,
                                                                           momentum=args.momentum),
                                                            Transpose(1, 2)) for i in range(args.hiden_layer_num)])

    def forward(self, z: torch.Tensor):
        b, c, s = z.shape
        # inputs nomralization
        z_mean = torch.mean(z, dim=(-1), keepdims=True)
        z_std = torch.std(z, dim=(-1), keepdims=True)
        z = (z - z_mean)/(z_std + 1e-4)
        # tokenization
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(
            self.input_projection(zcube)) + self.pos_embed
        cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
        z_embed = torch.cat((cls_token, z_embed), dim=-2)
        # dual attention encoder
        inputs = z_embed
        b, c, t, h = inputs.shape
        for a_2, a_1, mlp, drop, norm in zip(self.Attentions_over_token, self.Attentions_over_channel, self.
                                             Attentions_mlp, self.Attentions_dropout, self.Attentions_norm):
            output_1 = a_1(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c, t, -1)).reshape(b, c, t, -1)
            inputs = outputs
        # mlp decoder
        z_out = self.out_proj(outputs.reshape(b, c, -1))
        # denomrlaization
        z = z_out * (z_std+1e-4) + z_mean
        return z
