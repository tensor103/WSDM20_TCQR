import copy
from typing import *

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model

# implementation of transformer encoder
from models.metric_rank_recall import RankRecall, MRR


# seaborn.set_context(context="talk")


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

'''
The base model for task routing task
'''
class ModelBase(Model):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)
        self.rank_recall = RankRecall()
        self.mrr = MRR()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"mrr": self.mrr.get_metric(reset)}