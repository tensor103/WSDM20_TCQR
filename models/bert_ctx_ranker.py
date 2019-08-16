import random
from typing import *

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from overrides import overrides
from torch.autograd import Variable
from statistics import mean

from models.bert_sent_pooler import BertSentencePooler
from models.loss_modules import CoherenceLoss, TripletLoss, MarginRankLoss
from models.model_base import ModelBase


class BertCtxRanker(ModelBase):
    def __init__(self, args, num_authors: int, out_sz: int,
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
        self.num_sk, self.sk_dim = 20, 768
        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.sk_dim, self.num_sk), requires_grad=True)  # (m, d, k)

        self.attention = nn.Parameter(torch.randn(self.word_embeddings.get_output_dim(), self.sk_dim), requires_grad=True)
        # nn.Linear(self.word_embeddings.get_output_dim(), self.sk_dim)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        # self.loss = nn.CrossEntropyLoss()

        # loss related
        # self.cohere_loss = CoherenceLoss(self.encoder.get_output_dim(), out_sz)
        self.triplet_loss = TripletLoss(self.encoder.get_output_dim(), out_sz)
        self.rank_loss = MarginRankLoss(self.encoder.get_output_dim(), out_sz)

    def build_author_ctx_embed(self, token_hidden, author_embeds):

        # token_hidden (n, d, l)
        # author_embeds (m, d, k)

        n, _, l = token_hidden.shape
        m = author_embeds.shape[0]

        F_sim = torch.einsum('ndl,de,mek->nmlk', [token_hidden, self.attention, author_embeds])
        F_tanh = self.tanh(F_sim.contiguous().view(n * m, l, self.num_sk))  # (n * m, l, k)
        F_tanh = F_tanh.view(n, m, l, self.num_sk)  # (n, m, l, k)
        g_u = torch.mean(F_tanh, 2)  # (n, m, k)
        a_u = self.softmax(g_u)  # (n, m, k)

        author_ctx_embed = torch.einsum('mdk,nmk->nmd', [author_embeds, a_u])  # (n, m, d)

        return author_ctx_embed

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
        token_hidden = self.encoder(embeddings, mask)  # (n, d, l)

        author_ctx_embed = self.build_author_ctx_embed(token_hidden, self.author_embeddings)  # (n, m, d)
        token_embed = torch.mean(token_hidden, 2)  # (n, d)

        # coherence = torch.einsum('nd,nmd->nm', [token_embed, author_ctx_embed])  # (n, m)
        # loss, coherence = self.cohere_loss(token_embed, author_ctx_embed, label)
        # loss, coherence = self.triplet_loss(token_embed, author_ctx_embed, label)
        loss, coherence = self.rank_loss(token_embed, author_ctx_embed, answerers, accept_usr)

        # generate positive loss
        # all_labels = list(range(self.num_authors))
        # loss = 0
        # for i, pos_labels in enumerate(label):
        #
        #     num_pos = len(pos_labels)
        #     if num_pos == 0:
        #         continue
        #
        #     # BR-DEV relation
        #     pos_labels = torch.tensor(pos_labels)
        #     if torch.cuda.is_available(): pos_labels = pos_labels.cuda()
        #     pos_coherence = coherence[i, pos_labels]
        #     pos_loss = torch.sum(-torch.log(self.sigmoid(pos_coherence))) / num_pos
        #
        #     neg_labels = torch.tensor([item for item in all_labels if item not in pos_labels])
        #     num_neg = len(neg_labels)
        #     if torch.cuda.is_available(): neg_labels = neg_labels.cuda()
        #     neg_coherence = coherence[i, neg_labels]
        #     neg_loss = torch.sum(-torch.log(self.sigmoid(-neg_coherence))) / num_neg
        #
        #     loss += (pos_loss + neg_loss)
        #
        #     # DEV-DEV relation
        #     pos_authors = author_ctx_embed[i, pos_labels]  # (pos, d)
        #     neg_authors = author_ctx_embed[i, neg_labels]  # (neg, d)
        #
        #     auth_pos_coherence = torch.einsum('pd,qd->pq', [pos_authors, pos_authors])  # (pos, pos)
        #     auth_neg_coherence = torch.einsum('pd,nd->pn', [pos_authors, neg_authors])  # (pos, neg)
        #
        #     log_sig_auth = -torch.log(self.sigmoid(auth_pos_coherence))
        #     auth_pos_loss = (torch.sum(log_sig_auth) - torch.sum(torch.diagonal(log_sig_auth, 0)))
        #     if num_pos > 1:
        #         auth_pos_loss /= (num_pos * num_pos - num_pos)
        #
        #     auth_neg_loss = torch.sum(-torch.log(self.sigmoid(-auth_neg_coherence))) / (num_pos * num_neg)
        #
        #     # loss += (auth_pos_loss + auth_neg_loss)
        #     loss += (auth_pos_loss)
        #
        #     if torch.isnan(loss):
        #         raise ValueError("nan loss encountered")

        output = {"loss": loss, "coherence": coherence}
        # output = {"class_logits": class_logits}
        # output["loss"] = self.loss(class_logits, label)

        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        truth = [[j[0] for j in i] for i in answerers]


        self.mrr(predict, accept_usr)
        return output
