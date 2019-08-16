from typing import *

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask

from models.bert_sent_pooler import BertSentencePooler
from models.loss_modules import CoherenceLoss, TripletLoss
from models.model_base import ModelBase


class BertNoCtxRanker(ModelBase):
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
        self.author_embeddings = nn.Parameter(torch.randn(num_authors, self.sk_dim), requires_grad=True)  # (m, d)

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


    def build_coherence(self, token_hidden, author_embeds):

        # token_hidden (n, d, l)
        # author_embeds (m, d)

        n, _, l = token_hidden.shape
        m = author_embeds.shape[0]

        token_embed = torch.mean(token_hidden, 2)  # (n, d)

        coherence = torch.einsum('nd,md->nm', [token_embed, author_embeds])  # (n, m)

        return coherence


    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: Any, date: Any) -> torch.Tensor:

        # n -- batch number
        # m -- author number
        # d -- hidden dimension
        # k -- skill number
        # l -- text length
        # p -- pos/neg author number in one batch

        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        token_hidden = self.encoder(embeddings, mask)  # (n, d, l)
        token_embed = torch.mean(token_hidden, 2)  # (n, d)

        # coherence = self.build_coherence(token_hidden, self.author_embeddings)

        # generate positive loss
        # all_labels = list(range(self.num_authors))
        # loss = 0
        # for i, pos_labels in enumerate(label):
        #
        #     pos_labels = torch.tensor(pos_labels)
        #     if torch.cuda.is_available(): pos_labels = pos_labels.cuda()
        #     pos_coherence = coherence[i, pos_labels]
        #     pos_loss = torch.sum(-torch.log(self.sigmoid(pos_coherence))) / len(pos_labels)
        #
        #     neg_labels = torch.tensor([item for item in all_labels if item not in pos_labels])
        #     if torch.cuda.is_available(): neg_labels = neg_labels.cuda()
        #     neg_coherence = coherence[i, neg_labels]
        #     neg_loss = torch.sum(-torch.log(self.sigmoid(-neg_coherence))) / len(neg_labels)
        #
        #     loss += (pos_loss + neg_loss)
        #     pass

        # generate negative loss

        # # positive author embeddings
        # pos_author_embeds, pos_size = self.gen_pos_author_embeds(label)  # (n, p, d, k)
        #
        # # negative author embeddings
        # neg_size = pos_size  # choose negative samples the same as positive size
        # neg_author_embeds = self.gen_neg_author_embeds(label, neg_size)
        #
        # pos_coherence = self.build_coherence(token_hidden, pos_author_embeds)
        # neg_coherence = self.build_coherence(token_hidden, neg_author_embeds)
        #
        # pos_loss = torch.sum(torch.sum(torch.log(self.sigmoid(-pos_coherence)))) / pos_size
        # neg_loss = torch.sum(torch.sum(torch.log(self.sigmoid(neg_coherence)))) / neg_size

        # loss = pos_loss + neg_loss

        # loss, coherence = self.cohere_loss(token_embed, self.author_embeddings, label, no_ctx=True)
        loss, coherence = self.triplet_loss(token_embed, self.author_embeddings, label, no_ctx=True)

        output = {"loss": loss, "coherence": coherence}
        predict = np.argsort(-coherence.detach().cpu().numpy(), axis=1)
        self.rank_recall(predict, label)

        return output
