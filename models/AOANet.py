
import torch
import torch.nn as nn
from models.BaseModel import Base
from argparse import Namespace
from layers.FeatureEmbedding import FeatureEmbedding


class GeneralizedInteraction(nn.Module):
    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(
            0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(
            input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum("bnh,bnd->bnhd",
                                     B_0.repeat(1, self.input_subspaces, 1),
                                     B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim))  # b x (field*in) x d x d
        fusion = torch.matmul(outer_product.permute(
            0, 2, 3, 1), self.alpha)  # b x d x d x out
        fusion = self.W * fusion.permute(0, 3, 1, 2)  # b x out x d x d
        B_i = torch.matmul(fusion, self.h).squeeze(-1)  # b x out x d
        return B_i


class GeneralizedInteractionNet(nn.Module):
    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces,
                                                            num_subspaces,
                                                            num_fields,
                                                            embedding_dim)
                                     for i in range(num_layers)])

    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i


class AOANet(Base):
    def __init__(self, feat_map, args: Namespace) -> None:
        super(AOANet, self).__init__(**args)
        self.args = args
        self.featureEmbedding = FeatureEmbedding(feat_map, args.embedding_dim)

    def forward(self, input):
        x = self.featureEmbedding(input)

    def loss(self, X, Y):
        pass

    def metric(self, X, Y):
        pass
