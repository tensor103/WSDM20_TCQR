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
from models.model_base import ModelBase, clones, LayerNorm
from models.multihead_ctx_model import TempCtxAttention
from models.time_encoder import TimeEncoder, gen_time_encoding
from pandas import Timestamp

# seaborn.set_context(context="talk")


'''
Starting my own TempQuestCtxModel
'''
class TempQuestCtxModel(ModelBase):
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
        self.use_cat = False
        if self.use_cat:
            self.num_sk, self.sk_dim, self.time_dim = 20, 768, 32
            self.author_dim = self.sk_dim + self.time_dim
        else:
            self.num_sk, self.sk_dim, self.time_dim = 20, 768, 768
            self.author_dim = self.sk_dim

        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.num_sk, self.author_dim), requires_grad=True)  # (m, k, d)

        self.ctx_attention = TempCtxAttention(h=8, d_model=self.author_dim)
        # self.temp_ctx_attention = MHTempCtxAttention(h=8, d_model=self.sk_dim)

        # self.attention = nn.Parameter(torch.randn(self.word_embeddings.get_output_dim(), self.sk_dim), requires_grad=True)

        # temporal context
        self.time_encoder = TimeEncoder(self.time_dim, dropout=0.1, span=1, date_range=date_span)

        # layer_norm
        self.ctx_layer_norm = LayerNorm(self.author_dim)

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

        # transfer the date into time embedding
        # TODO: use answer date for time embedding
        time_embed = gen_time_encoding(self.time_encoder, date)

        if self.use_cat:
            token_temp_embed = torch.cat((token_embed, time_embed), 1)
        else:
            token_temp_embed = token_embed + time_embed
        author_ctx_embed = self.ctx_attention(token_temp_embed, self.author_embeddings, self.author_embeddings)  # (n, m, d)

        # add layer norm for author context embedding
        author_ctx_embed = self.ctx_layer_norm(author_ctx_embed)  # (n, m, d)

        # generate loss
        loss, coherence = self.rank_loss(token_temp_embed, author_ctx_embed, answerers, accept_usr)

        output = {"loss": loss, "coherence": coherence}

        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        truth = [[j[0] for j in i] for i in answerers]

        # self.rank_recall(predict, truth)
        # self.mrr(predict, truth)
        self.mrr(predict, accept_usr)


        return output
