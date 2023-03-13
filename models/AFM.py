from models.BaseModel import Base
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
        self.metrics = {}
        if self.args.metrics is not None:
            for m in self.args.metrics:
                if m == 'auc':
                    self.metrics[m] = AUROC(task="binary")
                if m == 'precision':
                    self.metrics[m] = Precision(
                        task="binary", threshold=self.args.threshold)
                if m == 'recall':
                    self.metrics[m] = Recall(
                        task="binary", threshold=self.args.threshold)
                if m == 'acc':
                    self.metrics[m] = Accuracy(
                        task="binary", threshold=self.args.threshold)

    def forward(self, x):
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))

    def loss(self, x, y):
        pred = self.forward(x)
        loss = self.criterion(pred, y.float())
        return loss

    def metric(self,  x, y, **kwargs) -> Dict[str, torch.Tensor]:
        res = {}
        if len(self.metrics) > 0:
            pred = self.forward(x)
            for k, m in self.metrics.items():
                m.update(pred, y)
                res[m] = m.compute()

        return res

    def reset(self) -> Dict[str, torch.Tensor]:
        res = {}
        if len(self.metrics) > 0:
            for k, m in self.metrics.items():
                res[k] = m.compute()
                m.reset()

        return res
