from typing import *

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask

from models.bert_sent_pooler import BertSentencePooler
# implementation of transformer encoder
from models.loss_modules import CoherenceLoss, TripletLoss, TemporalLoss, MarginRankLoss
from models.model_base import ModelBase, clones
from models.multihead_ctx_model import TempCtxAttention
from models.temp_ctx_attention import ShiftTempAttention
from models.temp_ctx_model import MHTempCtxAttention
from models.time_encoder import TimeEncoder

# seaborn.set_context(context="talk")


'''
Starting my own ShiftTempModel
'''
class ShiftTempModel(ModelBase):
    def __init__(self, num_authors: int, out_sz: int,
                 vocab: Vocabulary, date_span: Any,
                 num_shift: int, span: int):
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
        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.num_sk, self.sk_dim),
                                              requires_grad=True)  # (m, k, d)

        self.ctx_attention = TempCtxAttention(h=8, d_model=self.sk_dim)
        # layer_norm
        self.ctx_layer_norm = nn.LayerNorm(self.sk_dim)

        self.shift_temp_att = ShiftTempAttention(self.num_authors, self.sk_dim, date_span, num_shift, span)

        # self.cohere_loss = CoherenceLoss(self.encoder.get_output_dim(), out_sz)
        self.triplet_loss = TripletLoss(self.encoder.get_output_dim(), out_sz)
        self.temp_loss = TemporalLoss(self.encoder.get_output_dim(), out_sz)
        self.rank_loss = MarginRankLoss(self.encoder.get_output_dim(), out_sz)

        self.weight_temp = 0.3
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, answerers: Any, date: Any, accept_usr: Any, att_l=False) -> torch.Tensor:

        # n -- batch number
        # m -- author number
        # d -- hidden dimension
        # k -- skill number
        # l -- text length
        # p -- pos/neg author number in one batch
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        token_hidden = self.encoder(embeddings, mask).transpose(-1, -2)  # (n, d, l)

        token_embed = torch.mean(token_hidden, 1).squeeze(1)  # (n, d)
        # token_embed = token_hidden[:, :, -1]
        if att_l:
            author_ctx_embed = self.ctx_attention(token_hidden, self.author_embeddings, self.author_embeddings,
                                                  att_l=att_l)
        else:
            author_ctx_embed = self.ctx_attention(token_embed, self.author_embeddings, self.author_embeddings,
                                                  att_l=att_l)  # (n, m, d)

        # add layer norm for author context embedding
        author_ctx_embed = self.ctx_layer_norm(author_ctx_embed)  # (n, m, d)

        temp_ctx_embed, history_temp_embeds = self.shift_temp_att(author_ctx_embed, date)  # (n, m, d), (

        # generate loss
        # loss, coherence = self.cohere_loss(token_embed, temp_ctx_embed, label)
        # triplet_loss, coherence = self.triplet_loss(token_embed, temp_ctx_embed, label)
        triplet_loss, coherence = self.rank_loss(token_embed, temp_ctx_embed, answerers, accept_usr)

        truth = [[j[0] for j in i] for i in answerers]
        temp_loss = self.temp_loss(token_embed, history_temp_embeds, truth)
        loss = triplet_loss + temp_loss * self.weight_temp

        output = {"loss": loss, "coherence": coherence}

        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)

        # print("Truth:", accept_usr)
        self.mrr(predict, accept_usr)

        return output

