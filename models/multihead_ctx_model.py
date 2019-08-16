import math
from typing import *

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

from models.bert_sent_pooler import BertSentencePooler
# implementation of transformer encoder
from models.loss_modules import CoherenceLoss
from models.model_base import ModelBase, clones


# seaborn.set_context(context="talk")
from models.temp_ctx_attention import TempCtxAttention

'''
Starting my own MultiheadCtxModel
'''
class MultiHeadCtxModel(ModelBase):
    def __init__(self, num_authors: int, out_sz: int,
                 vocab: Vocabulary):
        super().__init__(vocab)

        # init word embedding
        bert_embedder = PretrainedBertEmbedder(
            pretrained_model="bert-base-uncased",
            top_layer_only=True,  # conserve memory
        )
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                      # we'll be ignoring masks so we'll need to set this to True
                                                      allow_unmatched_keys=True)

        self.encoder = BertSentencePooler(vocab, self.word_embeddings.get_output_dim())

        self.num_authors = num_authors

        # skills dim
        self.num_sk, self.sk_dim, self.time_dim = 20, 768, 32
        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.num_sk, self.sk_dim),
                                              requires_grad=True)  # (m, k, d)

        self.multihead_att = TempCtxAttention(h=8, d_model=self.sk_dim)

        self.attention = nn.Parameter(torch.randn(self.word_embeddings.get_output_dim(), self.sk_dim),
                                      requires_grad=True)
        # nn.Linear(self.word_embeddings.get_output_dim(), self.sk_dim)

        self.cohere_loss = CoherenceLoss(self.encoder.get_output_dim(), out_sz)

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: Any, date: Any, att_l=False) -> torch.Tensor:

        # n -- batch number
        # m -- author number
        # d -- hidden dimension
        # k -- skill number
        # l -- text length
        # p -- pos/neg author number in one batch

        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        token_hidden = self.encoder(embeddings, mask).transpose(-1, -2)  # (n, d, l)

        token_embed = torch.mean(token_hidden, 1).squeeze()  # (n, d)
        # token_embed = token_hidden[:, :, -1]
        if att_l:
            author_ctx_embed = self.multihead_att(token_hidden, self.author_embeddings, self.author_embeddings)
        else:
            author_ctx_embed = self.multihead_att(token_embed, self.author_embeddings, self.author_embeddings)

        # generate loss
        loss, coherence = self.cohere_loss(token_embed, author_ctx_embed, label)
        output = {"loss": loss, "coherence": coherence}

        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        self.rank_recall(predict, label)

        return output
