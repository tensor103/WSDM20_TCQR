import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
import br_utils
from models.bert_sent_pooler import BertSentencePooler
# implementation of transformer encoder
from models.coherence_modules import CoherenceInnerProd
from models.loss_modules import CoherenceLoss, TripletLoss, HTempLoss, MarginRankLoss
from models.model_base import ModelBase, clones, LayerNorm
from models.temp_ctx_model import MHTempCtxAttention
from models.time_encoder import TimeEncoder

class TempCtxAttentionNS(nn.Module):
    def __init__(self, h, d_model, d_query, d_time, dropout=0.1):
        "Take in model size and number of heads."
        super(TempCtxAttentionNS, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.query_linear = nn.Linear(d_query, d_model)
        self.time_linear = nn.Linear(d_time, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def temp_ctx_attention(self, query, key, value, time, mask=None, dropout=None, att_l=False):
        "Compute 'Scaled Dot Product Attention'"
        # query(n, h, d); key/value(m, h, d); time(n,m,h,d)

        d_k = query.size(-1)

        # Solution 1: nmhd, nmhd => mhnn; mhnn, nmhd => nmhd
        # scores = torch.einsum('nmhd,qmhd->mhnq', [query, key]) / math.sqrt(d_k)  # (m, h, n, n)
        #
        # p_attn = F.softmax(scores, dim=-1)  # (m, h, n, n)
        # if dropout is not None:
        #     p_attn = dropout(p_attn)
        #
        # temp_ctx_result = torch.einsum('mhnn,nmhd->nmhd', [p_attn, value])  # (n, m, h, d)

        # Solution 2: nmhd, nmhd => nmhdd; nmhdd, nmhd => nmhd
        ctx_scores = torch.einsum('nhd,mhq->nmhdq', [query, key]) / math.sqrt(d_k)  # (n, m, h, d, d)

        ctx_attn = F.softmax(ctx_scores, dim=-1)  # (n, m, h, d, d)
        if dropout is not None:
            ctx_attn = dropout(ctx_attn)

        ctx_value = torch.einsum('nmhdq,mhq->nmhd', [ctx_attn, value])  # (n, m, h, d)

        temp_scores = torch.einsum('nhd,nmhq->nmhdq', [time, ctx_value]) / math.sqrt(d_k)  # (n, m, h, d, d)

        temp_attn = F.softmax(temp_scores, dim=-1)  # (n, m, h, d, d)
        if dropout is not None:
            temp_attn = dropout(temp_attn)

        temp_ctx_result = torch.einsum('nmhdq,nmhq->nmhd', [temp_attn, ctx_value])  # (n, m, h, d)

        return temp_ctx_result, temp_attn


    def forward(self, query, key, value, time, mask=None, att_l=False):

        # time(n, dt); query(n,d); key/value -- (n, m, d)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n = query.size(0)
        m = key.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.query_linear(query).view(n, self.h, self.d_k)  # (n, h, d_k)
        time = self.time_linear(time).view(n, self.h, self.d_k)  # (n, h, d_k)
        key = self.linears[0](key).view(m, self.h, self.d_k)  # (m, h, d_k)
        value = self.linears[1](value).view(m, self.h, self.d_k)  # (m, h, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.temp_ctx_attention(query, key, value, time, mask=mask,
                                               dropout=self.dropout)  # x -- (n, m, h, d_k)

        # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)
        x = x.contiguous().view(n, m, self.h * self.d_k)  # (n, m, h * d_k)
        return self.linears[-1](x)  # (n, m, h * d_k)


#
# class MHTempCtxAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MHTempCtxAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def temp_ctx_attention(self, query, key, value, mask=None, dropout=None, att_l=False):
#         "Compute 'Scaled Dot Product Attention'"
#         # query/key/value -- (n, m, h, d)
#
#         d_k = query.size(-1)
#
#         # Solution 1: nmhd, nmhd => mhnn; mhnn, nmhd => nmhd
#         # scores = torch.einsum('nmhd,qmhd->mhnq', [query, key]) / math.sqrt(d_k)  # (m, h, n, n)
#         #
#         # p_attn = F.softmax(scores, dim=-1)  # (m, h, n, n)
#         # if dropout is not None:
#         #     p_attn = dropout(p_attn)
#         #
#         # temp_ctx_result = torch.einsum('mhnn,nmhd->nmhd', [p_attn, value])  # (n, m, h, d)
#
#         # Solution 2: nmhd, nmhd => nmhdd; nmhdd, nmhd => nmhd
#         scores = torch.einsum('nmhd,nmhq->nmhdq', [query, key]) / math.sqrt(d_k)  # (n, m, h, d, d)
#
#         p_attn = F.softmax(scores, dim=-1)  # (m, h, n, n)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#
#         temp_ctx_result = torch.einsum('nmhdq,nmhq->nmhd', [p_attn, value])  # (n, m, h, d)
#
#         return temp_ctx_result, p_attn
#
#     def forward(self, query, key, value, mask=None, att_l=False):
#
#         # query/key/value -- (n, m, d)
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         n, m, _ = query.size()
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query = self.linears[0](query).view(n, m, self.h, self.d_k)  # (n, m, h, d_k)
#         key = self.linears[1](key).view(n, m, self.h, self.d_k)  # (n, m, h, d_k)
#         value = self.linears[2](value).view(n, m, self.h, self.d_k)  # (n, m, h, d_k)
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = self.temp_ctx_attention(query, key, value, mask=mask,
#                                                dropout=self.dropout)  # x -- (n, m, h, d_k)
#
#         # 3) "Concat" using a view and apply a final linear.
#         # x = x.transpose(1, 2).contiguous() \
#         #     .view(nbatches, -1, self.h * self.d_k)
#         x = x.contiguous().view(n, m, self.h * self.d_k)  # (n, m, h * d_k)
#         return self.linears[-1](x)  # (n, m, h * d_k)



class TempCtxAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(TempCtxAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def ctx_attention(self, query, key, value, mask=None, dropout=None, att_l=False):
        "Compute 'Scaled Dot Product Attention'"
        # query -- (n, h, l, d)
        # query -- (n, h, d) new version
        # key/value -- (m, h, k, d)

        d_k = query.size(-1)

        if att_l:
            scores = torch.einsum('nhld,mhkd->nmhlk', [query, key]) / math.sqrt(d_k)  # (n, m, h, l, k)
            # average pooling on the dimension l
            scores = torch.mean(scores, -2)  # (n, m, h, k)
        else:
            scores = torch.einsum('nhd,mhkd->nmhk', [query, key]) / math.sqrt(d_k)  # (n, m, h, k)

        p_attn = F.softmax(scores, dim=-1)  # (n, m, h, k)
        if dropout is not None:
            p_attn = dropout(p_attn)

        ctx_result = torch.einsum('nmhk,mhkd->nmhd', [p_attn, value])  # (n, m, h, d)
        return ctx_result, p_attn

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n = query.size(0)
        m = key.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        try:
            query = self.linears[0](query).view(n, self.h, self.d_k)  # (n, h, d_k)
        except:
            print(query.size())
            query = self.linears[0](query).view(n, self.h, self.d_k)  # (n, h, d_k)

        key = self.linears[1](key).view(m, -1, self.h, self.d_k).transpose(1, 2)  # (m, h, k, d_k)
        value = self.linears[2](value).view(m, -1, self.h, self.d_k).transpose(1, 2)  # (m, h, k, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.ctx_attention(query, key, value, mask=mask,
                                          dropout=self.dropout)  # x -- (n, m, h, d_k)

        # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)
        x = x.contiguous().view(n, m, self.h * self.d_k)  # (n, m, h * d_k)
        return self.linears[-1](x)  # (n, m, h * d_k)



'''
ShiftTempAttention
'''
class ShiftTempAttention(Model):
    def __init__(self, num_authors: int, dim: int, date_span: Any, num_shift: int, span: int, ignore_time: bool):
        super().__init__(None)

        # skills dim
        self.num_authors, self.sk_dim = num_authors, dim

        # shifted temporal attentions
        self.num_shift, self.span_size = num_shift, span
        self.temp_att_modules = clones(MHTempCtxAttention(h=8, d_model=self.sk_dim), self.num_shift)
        self.temp_layer_norms = clones(nn.LayerNorm(self.sk_dim), self.num_shift - 1)

        # temporal encoder
        self.time_encoder = TimeEncoder(self.sk_dim, dropout=0.1, span=span, date_range=date_span)

        self.ignore_time = ignore_time

    def forward(self, token_embed: torch.Tensor, temp_ctx_embed: torch.Tensor, date: Any) -> torch.Tensor:

        # n -- batch number
        # m -- author number
        # d -- hidden dimension
        # k -- skill number
        # l -- text length
        # p -- pos/neg author number in one batch
        history_embeds = []
        token_embed = token_embed.unsqueeze(1).expand(-1, self.num_authors, -1)  # (n, m, d)
        for i in range(self.num_shift):
            # transfer the date into time embedding
            time_embed = [self.time_encoder.get_time_encoding(d, num_shift=i) for d in date]
            time_embed = torch.stack(time_embed, dim=0)  # (n, d)
            time_embed = time_embed.unsqueeze(1).expand(-1, self.num_authors, -1)  # (n, m, d)

            # Option 1
            # temp_ctx_embed_te = temp_ctx_embed + time_embed
            # temp_ctx_embed_te = self.temp_att_modules[i](temp_ctx_embed_te, temp_ctx_embed_te,
            #                                              temp_ctx_embed_te) + temp_ctx_embed

            token_temp_embed = token_embed if self.ignore_time else token_embed + time_embed

            # token_temp_embed = token_embed + time_embed
            temp_ctx_embed_te = self.temp_att_modules[i](token_temp_embed, temp_ctx_embed,
                                                         temp_ctx_embed) + temp_ctx_embed

            if i == self.num_shift - 1:
                temp_ctx_embed = temp_ctx_embed_te
            else:
                temp_ctx_embed = self.temp_layer_norms[i](temp_ctx_embed_te)  # (n, m, d)

            history_embeds.append(temp_ctx_embed_te)

        history_embed = torch.stack(history_embeds, dim=1)  # (n, s, m, d)

        return temp_ctx_embed, history_embed


