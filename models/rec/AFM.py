from models.base import Base
from layers.Layers import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine
from typing import Dict
import torch
from torchmetrics import AUROC, Precision, Recall, Accuracy


class AFM(Base):
    def __init__(self, args):
        super(AFM, self).__init__()
        self.args = args
        self.num_fields = len(args.field_dims)
        self.embedding = FeaturesEmbedding(args.field_dims, args.embed_dim)
        self.linear = FeaturesLinear(args.field_dims)
        self.afm = AttentionalFactorizationMachine(
            args.embed_dim, args.attn_size, args.dropouts)
        self.criterion = torch.nn.BCELoss()
        self.metrics = {'auc': AUROC(task="binary"),
                        'precision': Precision(task="binary", threshold=self.args.threshold),
                        'recall': Recall(task="binary", threshold=self.args.threshold),
                        'acc': Accuracy(task="binary", threshold=self.args.threshold)
                        }

    def forward(self, x):
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))

    def loss(self, x, y):
        pred = self.forward(x)
        loss = self.criterion(pred, y.float())
        return loss

    def metric(self,  x, y, **kwargs) -> Dict[str, torch.Tensor]:
        res = {}
        pred = self.forward(x)
        for k, m in self.metrics.items():
            m.to(x.device)
            m.update(pred, y)
            res[k] = m.compute()

        return res

    def reset(self):
        for k, m in self.metrics.items():
            m.reset()
