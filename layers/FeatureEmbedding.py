import torch
from torch import nn
from typing import List, Dict


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 features: List[Dict],
                 embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.embeds = {}
        self.feat_nums = 0
        for f in features:
            name = f['name']
            type = f['type']

            if type == 'categorical':
                max_idx = f['max_idx']
                padding_idx = f.get("padding_idx")
                pretrain = f.get('pretrain')
                if pretrain:
                    embedding_matrix = torch.load(pretrain)
                else:
                    embedding_matrix = nn.Embedding(max_idx,
                                                    embedding_dim,
                                                    padding_idx=padding_idx)
                self.feat_nums += embedding_dim
            elif type == 'numeric':
                need_embed = f.get('emb')
                numeric_size = f.get('size')
                if need_embed:
                    embedding_matrix = nn.Linear(
                        numeric_size if numeric_size is not None else 1, embedding_dim, bias=False)
                    self.feat_nums += embedding_dim
                else:
                    def embedding_matrix(x): return x
                    self.feat_nums += 1
            elif type == 'sequence':
                max_idx = f['max_idx']
                padding_idx = f.get("padding_idx")
                pretrain = f.get('pretrain')
                if pretrain:
                    embedding_matrix = torch.load(pretrain)
                else:
                    embedding_matrix = nn.Embedding(max_idx,
                                                    embedding_dim,
                                                    padding_idx=padding_idx)
                self.feat_nums += embedding_dim
            else:
                assert (False)
            self.embeds[name] = embedding_matrix

    def forward(self, x: Dict):
        res = {}
        for k, y in x.items():
            res[k] = self.embeds[k](y)
        return self.dictToTensor(res)

    def dictToTensor(self, x: Dict):
        data_list = []
        for k, v in x.items():
            data_list.append(v)

        return torch.cat(data_list, dim=1)
