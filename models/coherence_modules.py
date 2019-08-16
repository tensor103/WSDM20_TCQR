import random
from typing import *

import torch
import torch.nn as nn
from allennlp.models import Model
import br_utils
import sys
sys.path.insert(0, "../")
# from .. import br_util
#

'''
Inner Product Coherence Function
Note: time embed is ignored
'''
class CoherenceInnerProd(Model):
    def __init__(self):
        super().__init__(None)

    def forward(self, token_temp_embed, time_embed, temp_ctx_embed, no_ctx=False) -> torch.Tensor:

        if no_ctx:
            coherence = torch.einsum('nd,md->nm', [token_temp_embed, temp_ctx_embed])  # (n, m)
        else:
            coherence = torch.einsum('nd,nmd->nm', [token_temp_embed, temp_ctx_embed])  # (n, m)

        return coherence


'''
Bilinear Coherence Function
Note: time embed is ignored
'''
class CoherenceBiLinear(Model):
    def __init__(self, dim):
        super().__init__(None)
        self.projection = nn.Linear(dim, dim)

    def forward(self, token_temp_embed, time_embed, temp_ctx_embed, no_ctx=True) -> torch.Tensor:

        # use bilinear to map nd, md -> nm
        # n * nd * x
        proj_token_embed = self.projection(token_temp_embed)
        if no_ctx:
            coherence = torch.einsum('nd,md->nm', [proj_token_embed, temp_ctx_embed])  # (n, m)
        else:
            coherence = torch.einsum('nd,nmd->nm', [proj_token_embed, temp_ctx_embed])  # (n, m)

        return coherence
