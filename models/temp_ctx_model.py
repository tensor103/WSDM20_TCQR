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

import br_utils
from models.bert_sent_pooler import BertSentencePooler
# implementation of transformer encoder
from models.coherence_modules import CoherenceInnerProd
from models.loss_modules import CoherenceLoss, TripletLoss, HTempLoss, MarginRankLoss
from models.model_base import ModelBase, clones
from models.time_encoder import TimeEncoder
from pandas import Timestamp

# seaborn.set_context(context="talk")

def temp_ctx_attention(query, key, value, mask=None, dropout=None, att_l=False):
    "Compute 'Scaled Dot Product Attention'"
    # query/key/value -- (n, m, h, d)

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
    scores = torch.einsum('nmhd,nmhq->nmhdq', [query, key]) / math.sqrt(d_k)  # (n, m, h, d, d)

    p_attn = F.softmax(scores, dim=-1)  # (m, h, n, n)
    if dropout is not None:
        p_attn = dropout(p_attn)

    temp_ctx_result = torch.einsum('nmhdq,nmhq->nmhd', [p_attn, value])  # (n, m, h, d)

    return temp_ctx_result, p_attn


class MHTempCtxAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MHTempCtxAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, att_l=False):

        # query/key/value -- (n, m, d)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n, m, _ = query.size()

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query).view(n, m, self.h, self.d_k)  # (n, m, h, d_k)
        key = self.linears[1](key).view(n, m, self.h, self.d_k)  # (n, m, h, d_k)
        value = self.linears[2](value).view(n, m, self.h, self.d_k)  # (n, m, h, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = temp_ctx_attention(query, key, value, mask=mask,
                                 dropout=self.dropout)  # x -- (n, m, h, d_k)

        # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)
        x = x.contiguous().view(n, m, self.h * self.d_k)  # (n, m, h * d_k)
        return self.linears[-1](x)  # (n, m, h * d_k)


'''
Starting my own TempCtxModel
'''
class TempCtxModel(ModelBase):
    def __init__(self, num_authors: int, out_sz: int,
                 vocab: Vocabulary, date_span: Any):
        super().__init__(vocab)

        # init word embedding
        bert_embedder = PretrainedBertEmbedder(
            pretrained_model="bert-base-uncased",
            top_layer_only=True,  # conserve memory
        )
        self.date_span = date_span
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                      # we'll be ignoring masks so we'll need to set this to True
                                                      allow_unmatched_keys=True)

        self.encoder = BertSentencePooler(vocab, self.word_embeddings.get_output_dim())

        self.num_authors = num_authors

        # skills dim
        self.num_sk, self.sk_dim = 20, 768
        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.num_sk, self.sk_dim), requires_grad=True)  # (m, k, d)

        self.ctx_attention = TempCtxAttention(h=8, d_model=self.sk_dim)
        self.temp_ctx_attention = MHTempCtxAttention(h=8, d_model=self.sk_dim)

        self.attention = nn.Parameter(torch.randn(self.word_embeddings.get_output_dim(), self.sk_dim), requires_grad=True)

        # temporal context
        self.time_encoder = TimeEncoder(self.sk_dim, dropout=0.1, span=1, date_range=date_span)

        # layer_norm
        self.ctx_layer_norm = LayerNorm(self.sk_dim)

        # loss related
        # self.cohere_loss = CoherenceLoss(self.encoder.get_output_dim(), out_sz)
        self.triplet_loss = TripletLoss(self.encoder.get_output_dim(), out_sz)
        self.htemp_loss = HTempLoss(self.encoder.get_output_dim(), out_sz)
        self.rank_loss = MarginRankLoss(self.encoder.get_output_dim(), out_sz)

        self.coherence_func = CoherenceInnerProd()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, answerers: Any, date: Any, accept_usr: Any) -> torch.Tensor:

        # n -- batch number
        # m -- author number
        # d -- hidden dimension
        # k -- skill number
        # l -- text length
        # p -- pos/neg author number in one batch
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)  # (n, l, d)
        token_hidden = self.encoder(embeddings, mask).transpose(-1, -2)  # (n, l, d)

        token_embed = torch.mean(token_hidden, 1).squeeze(1)  # (n, d)
        # token_embed = token_hidden[:, :, -1]
        author_ctx_embed = self.ctx_attention(token_embed, self.author_embeddings, self.author_embeddings)  # (n, m, d)

        # add layer norm for author context embedding
        author_ctx_embed = self.ctx_layer_norm(author_ctx_embed)  # (n, m, d)

        # transfer the date into time embedding
        # TODO: use answer date for time embedding
        time_embed = gen_time_encoding(self.time_encoder, answerers, date, embeddings.size(2), self.num_authors, train_mode=self.training)
        # time_embed = [self.time_encoder.get_time_encoding(i) for i in date]
        # time_embed = torch.stack(time_embed, dim=0)  # (n, d)
        # time_embed = time_embed.unsqueeze(1).expand(-1, self.num_authors, -1)  # (n, m, d)

        author_ctx_embed_te = author_ctx_embed + time_embed
        author_tctx_embed = self.temp_ctx_attention(time_embed, author_ctx_embed, author_ctx_embed)  # (n, m, d)
        # author_tctx_embed = self.temp_ctx_attention(author_ctx_embed_te, author_ctx_embed_te, author_ctx_embed_te)  # (n, m, d)

        # get horizontal temporal time embeddings
        # htemp_embeds = []
        # truth = [[j[0] for j in i] for i in answerers]
        # for i, d in enumerate(date):
        #     pos_labels = br_utils.to_cuda(torch.tensor(truth[i]))
        #     post_time_embeds = self.time_encoder.get_post_encodings(d)  # (t, d)
        #     post_time_embeds = post_time_embeds.unsqueeze(1).expand(-1, pos_labels.size(0), -1)  # (t, pos, d)
        #
        #     pos_embed = author_ctx_embed[i, pos_labels, :]  # (pos, d)
        #     pos_embed = pos_embed.unsqueeze(0).expand(post_time_embeds.size(0), -1, -1)  # (t, pos, d)
        #     author_post_ctx_embed_te = pos_embed + post_time_embeds
        #     # author_post_ctx_embed = self.temp_ctx_attention(author_post_ctx_embed_te, author_post_ctx_embed_te, author_post_ctx_embed_te)  # (t, pos, d)
        #     author_post_ctx_embed = self.temp_ctx_attention(post_time_embeds, pos_embed, pos_embed)  # (t, pos, d)
        #     htemp_embeds.append(author_post_ctx_embed)
        # htemp_loss = self.htemp_loss(token_embed, htemp_embeds)

        # generate loss
        # loss, coherence = self.rank_loss(token_embed, author_tctx_embed, answerers)
        loss, coherence = self.rank_loss(token_embed, author_tctx_embed, answerers, accept_usr)
        # loss += 0.5 * htemp_loss

        # coherence = self.coherence_func(token_embed, None, author_tctx_embed)
        output = {"loss": loss, "coherence": coherence}

        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        truth = [[j[0] for j in i] for i in answerers]

        # self.rank_recall(predict, truth)
        # self.mrr(predict, truth)
        self.mrr(predict, accept_usr)


        return output
