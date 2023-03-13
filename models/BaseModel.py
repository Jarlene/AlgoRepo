import torch
import torch.nn as nn
import numpy as np
from abc import abstractclassmethod
from typing import List, Union, Any, Tuple, Optional, Dict

import torch.nn.functional as F


class Base(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(Base, self).__init__()

    @abstractclassmethod
    def forward(self, **kwargs) -> torch.Tensor:
        pass

    @abstractclassmethod
    def loss(self,  **kwargs) -> torch.Tensor:
        pass

    @abstractclassmethod
    def metric(self, ** kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractclassmethod
    def reset(self, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def compute_cl_loss(self, p: torch.Tensor, q: torch.Tensor, temperature, debiased=True, tau_plus=0.1):
        batch_size = p.size(0)

        def get_negative_mask(batch_size):
            negative_mask = torch.ones(
                (batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0

            negative_mask = torch.cat((negative_mask, negative_mask), 0)
            return negative_mask

        # neg score
        out = torch.cat([p, q], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size).to(p.device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(p * q, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)
        loss = (- torch.log(pos / (pos + Ng))).mean()
        if torch.isnan(loss) or not torch.isfinite(loss):
            print("cl loss is nan")
        return loss

    def compute_kl_loss(self, p: torch.Tensor, q: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                          F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                          F.softmax(p, dim=-1), reduction='none')
        if pad_mask is not None:
            p_loss.masked_fill(pad_mask, 0.)
            q_loss.masked_fill(pad_mask, 0.)
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss
