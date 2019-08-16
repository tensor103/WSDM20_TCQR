import random
from typing import *

import torch
import torch.nn as nn
from allennlp.models import Model
import br_utils
import sys

from models.coherence_modules import CoherenceInnerProd, CoherenceBiLinear

sys.path.insert(0, "../")
# from .. import br_util
#

'''
CoherenceLoss
'''
class CoherenceLoss(Model):
    def __init__(self, dim: int, out_sz: int):
        super().__init__(None)

        # loss related
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Linear(dim, out_sz)

    def forward(self, token_embed, temp_ctx_embed: torch.Tensor, label: Any, no_ctx=False, att_l=False) -> torch.Tensor:

        if no_ctx:
            coherence = torch.einsum('nd,md->nm', [token_embed, temp_ctx_embed])  # (n, m)
            num_authors = temp_ctx_embed.size(0)
        else:
            coherence = torch.einsum('nd,nmd->nm', [token_embed, temp_ctx_embed])  # (n, m)
            num_authors = temp_ctx_embed.size(1)

        # generate positive loss
        all_labels = list(range(num_authors))
        loss = 0
        for i, pos_labels in enumerate(label):

            pos_labels = torch.tensor(pos_labels)
            if torch.cuda.is_available():
                pos_labels = pos_labels.cuda()
            pos_coherence = coherence[i, pos_labels]
            pos_loss = torch.sum(-torch.log(self.sigmoid(pos_coherence))) / len(pos_labels)

            neg_labels = torch.tensor([item for item in all_labels if item not in pos_labels])
            if torch.cuda.is_available(): neg_labels = neg_labels.cuda()
            neg_coherence = coherence[i, neg_labels]
            neg_loss = torch.sum(-torch.log(self.sigmoid(-neg_coherence))) / len(neg_labels)

            loss += (pos_loss + neg_loss)

        if torch.isnan(loss):
            author_ctx_embed = self.ctx_attention(token_embed, self.author_embeddings, self.author_embeddings,
                                                  att_l=att_l)
            print("pos_labels:", pos_labels)
            raise ValueError("nan loss encountered")

        return loss, coherence

'''
TripletLoss
'''
class TripletLoss(Model):
    def __init__(self, dim: int, out_sz: int):
        super().__init__(None)

        # loss related
        self.margin = 1.0
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Linear(dim, out_sz)

        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, token_temp_embed, temp_ctx_embed: torch.Tensor, labels: Any, no_ctx=False, att_l=False) -> torch.Tensor:

        from br_utils import to_cuda

        if no_ctx:
            num_authors, dim = temp_ctx_embed.size()
            coherence = torch.einsum('nd,md->nm', [token_temp_embed, temp_ctx_embed])  # (n, m)
        else:
            _, num_authors, dim = temp_ctx_embed.size()
            coherence = torch.einsum('nd,nmd->nm', [token_temp_embed, temp_ctx_embed])  # (n, m)

        MAX_NUM_TRIPLET = 1000
        # generate positive loss
        all_labels = list(range(num_authors))
        loss = 0
        for i, pos_labels in enumerate(labels):

            # pos -- number of positive samples
            # neg -- number of negative samples
            num_pos, num_neg = len(pos_labels), num_authors - len(pos_labels)
            # choose a smaller negative sample size to avoid exceeding the MAX_NUM_TRIPLET
            num_neg = min(num_neg, int(MAX_NUM_TRIPLET / num_pos))
            neg_label_cands = [item for item in all_labels if item not in pos_labels]
            neg_labels = random.sample(neg_label_cands, num_neg)

            anchor_embed = token_temp_embed[i, :]  # (d, 1)

            pos_labels = to_cuda(torch.tensor(pos_labels))
            neg_labels = to_cuda(torch.tensor(neg_labels))

            if no_ctx:
                pos_embed = temp_ctx_embed[pos_labels, :]  # (pos, d)
                neg_embed = temp_ctx_embed[neg_labels, :]  # (neg, d)
            else:
                pos_embed = temp_ctx_embed[i, pos_labels, :]  # (pos, d)
                neg_embed = temp_ctx_embed[i, neg_labels, :]  # (neg, d)

            pos_cohere = torch.einsum('d,pd->p', [anchor_embed, pos_embed])  # (pos, 1)
            neg_cohere = torch.einsum('d,nd->n', [anchor_embed, neg_embed])  # (neg, 1)

            pos_cohere_ex = pos_cohere.unsqueeze(1).expand(num_pos, num_neg)  # (pos, neg)
            neg_cohere_ex = neg_cohere.unsqueeze(0).expand(num_pos, num_neg)  # (pos, neg)

            # triplet_loss(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)
            zero_tensor = to_cuda(torch.zeros(num_pos, num_neg))
            triplet_loss = torch.max(-pos_cohere_ex + neg_cohere_ex + self.margin, zero_tensor)
            loss += torch.sum(triplet_loss) / (num_pos * num_neg)

        if torch.isnan(loss):
            author_ctx_embed = self.ctx_attention(token_temp_embed, self.author_embeddings, self.author_embeddings,
                                                  att_l=att_l)
            print("pos_labels:", pos_labels)
            raise ValueError("nan loss encountered")

        return loss, coherence

'''
Horizontal Temporal Loss

Horizontal temporal loss is used in the temporal based context-aware models, including temp_ctx
Given the hidden representation of the temporal layers by the sequence of time: z_0, z_1, ..., z_t
# temporal_loss(a, z_i, z_{i+1}) = max(d(a, z_i) - d(a, z_{i+1}) + margin, 0)
'''
class HTempLoss(Model):
    def __init__(self, dim: int, out_sz: int):
        super().__init__(None)

        # loss related
        self.margin = 1.0
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Linear(dim, out_sz)

        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, token_embed, htemp_embeds: torch.Tensor) -> torch.Tensor:

        # z_t > z_{t+1}
        # history_embed - (n, s, m, d) where s is shift size
        # batch_size, num_shift, _, num_dim = history_embed.size()  # s -- size of shift

        loss = 0
        for i, htemp_embed in enumerate(htemp_embeds):

            num_temp, num_pos, _ = htemp_embed.size()

            for t in range(num_temp):
                # pos -- number of positive samples
                # neg -- number of negative samples
                anchor_embed = token_embed[i, :]  # (d, 1)

                # avoid to compare z_1 to z_0 since z_0 has no temporal information
                pos_embed = htemp_embed[:-1, :, :]  # (s-1, pos, d)
                neg_embed = htemp_embed[1:, :, :]  # (s-1, pos, d)

                pos_cohere = torch.einsum('d,spd->sp', [anchor_embed, pos_embed])  # (s-1, pos)
                neg_cohere = torch.einsum('d,spd->sp', [anchor_embed, neg_embed])  # (s-1, pos)

                zero_tensor = br_utils.to_cuda(torch.zeros(num_temp - 1, num_pos))
                temp_loss = torch.max(-pos_cohere + neg_cohere + self.margin, zero_tensor)
                loss += torch.sum(temp_loss) / ((num_temp - 1) * num_pos)

        return loss


'''
TemporalLoss

Temporal loss can be only used in the temporal based context-aware models, including shift_temp and mspan_temp
Given the hidden representation of the temporal layers by the sequence of time: z_0, z_1, ..., z_t
# temporal_loss(a, z_i, z_{i+1}) = max(d(a, z_i) - d(a, z_{i+1}) + margin, 0)
'''
class TemporalLoss(Model):
    def __init__(self, dim: int, out_sz: int):
        super().__init__(None)

        # loss related
        self.margin = 1.0
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Linear(dim, out_sz)

        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, token_embed, history_embed: torch.Tensor, labels: Any) -> torch.Tensor:

        # z_t > z_{t+1}
        from br_utils import to_cuda

        # history_embed - (n, s, m, d) where s is shift size
        batch_size, num_shift, _, num_dim = history_embed.size()  # s -- size of shift

        loss = 0
        for i, pos_labels in enumerate(labels):

            num_pos = len(pos_labels)
            # pos -- number of positive samples
            # neg -- number of negative samples
            anchor_embed = token_embed[i, :]  # (d, 1)

            pos_labels = to_cuda(torch.tensor(pos_labels))
            # pos_embed = history_embed[i, :-1, pos_labels, :]  # (s-1, pos, d)
            # neg_embed = history_embed[i, 1:, pos_labels, :]  # (s-1, pos, d)

            # avoid to compare z_1 to z_0 since z_0 has no temporal information
            pos_embed = history_embed[i, 1:-1, pos_labels, :]  # (s-1, pos, d)
            neg_embed = history_embed[i, 2:, pos_labels, :]  # (s-1, pos, d)

            pos_cohere = torch.einsum('d,spd->sp', [anchor_embed, pos_embed])  # (s-1, pos)
            neg_cohere = torch.einsum('d,spd->sp', [anchor_embed, neg_embed])  # (s-1, pos)

            zero_tensor = to_cuda(torch.zeros(num_shift - 2, num_pos))
            temp_loss = torch.max(-pos_cohere + neg_cohere + self.margin, zero_tensor)
            loss += torch.sum(temp_loss) / ((num_shift - 2) * num_pos)

        return loss

'''
TripletLoss
'''
class MarginRankLoss(Model):
    def __init__(self, dim: int, out_sz: int):
        super().__init__(None)

        # loss related
        self.pos_margin, self.neg_margin = 1.0, 10.0
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.projection = nn.Linear(dim, out_sz)
        self.coherence_func = CoherenceInnerProd()
        # self.coherence_func = CoherenceBiLinear(dim)

        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, token_temp_embed, temp_ctx_embed: torch.Tensor, labels: Any, accept_usr: Any, no_ctx=False, att_l=False) -> torch.Tensor:

        from br_utils import to_cuda

        if no_ctx:
            num_authors, dim = temp_ctx_embed.size()
        else:
            _, num_authors, dim = temp_ctx_embed.size()

        coherence = self.coherence_func(token_temp_embed, None, temp_ctx_embed, no_ctx)

        MAX_NUM_TRIPLET = 1000
        # generate positive loss
        all_labels = list(range(num_authors))
        loss = 0
        for i, pos_labels in enumerate(labels):

            acc_label = accept_usr[i][0]
            noacc_pos_labels = [label for label in pos_labels if label[0] != acc_label]
            # if len(noacc_pos_labels) + 1 != num_pos:
            #     print("accept user is not in answerers!!!!!!!")
            #     print(acc_label, pos_labels)
            sorted_labels = sorted(noacc_pos_labels, key=lambda tup: tup[1], reverse=True)
            pos_labels = [acc_label] + [i[0] for i in sorted_labels]

            # pos -- number of positive samples
            # neg -- number of negative samples
            num_pos, num_neg = len(pos_labels), num_authors - len(pos_labels)

            # choose a smaller negative sample size to avoid exceeding the MAX_NUM_TRIPLET
            num_neg = min(num_neg, int(MAX_NUM_TRIPLET / num_pos))
            neg_label_cands = [item for item in all_labels if item not in pos_labels]
            neg_labels = random.sample(neg_label_cands, num_neg)

            anchor_embed = token_temp_embed[i, :]  # (d, 1)

            pos_labels = to_cuda(torch.tensor(pos_labels))
            neg_labels = to_cuda(torch.tensor(neg_labels))

            if no_ctx:
                pos_embed = temp_ctx_embed[pos_labels, :]  # (pos, d)
                neg_embed = temp_ctx_embed[neg_labels, :]  # (neg, d)
            else:
                pos_embed = temp_ctx_embed[i, pos_labels, :]  # (pos, d)
                neg_embed = temp_ctx_embed[i, neg_labels, :]  # (neg, d)

            pos_cohere = torch.einsum('d,pd->p', [anchor_embed, pos_embed])  # (pos, 1)
            neg_cohere = torch.einsum('d,nd->n', [anchor_embed, neg_embed])  # (neg, 1)

            pos_cohere_ex = pos_cohere.unsqueeze(1).expand(num_pos, num_neg)  # (pos, neg)
            neg_cohere_ex = neg_cohere.unsqueeze(0).expand(num_pos, num_neg)  # (pos, neg)

            # loss between positive samples and negative samples
            # triplet_loss(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)
            zero_tensor = to_cuda(torch.zeros(num_pos, num_neg))
            triplet_loss = torch.max(-pos_cohere_ex + neg_cohere_ex + self.neg_margin, zero_tensor)
            loss += torch.sum(triplet_loss) / (num_pos * num_neg)

            # ==== deprecated: ordered loss between positive samples ====
            # zero_tensor = to_cuda(torch.zeros(num_pos, num_pos))
            # if num_pos > 1:
            #     pos_cohere_h = pos_cohere.unsqueeze(1).expand(num_pos, num_pos)
            #     pos_cohere_v = pos_cohere.unsqueeze(0).expand(num_pos, num_pos)
            #     triplet_loss = torch.max(-pos_cohere_h + pos_cohere_v + self.pos_margin, zero_tensor)
            #     loss += torch.sum(torch.triu(triplet_loss, diagonal=1)) / (num_pos * (num_pos - 1) / 2)
            # ================================================

            # ==== loss between accepted user and positive samples ====
            # Notice: the accepted user is in the first position of positive labels

            if num_pos > 1:
                zero_tensor = to_cuda(torch.zeros(1, num_pos - 1))
                acc_cohere_ex = pos_cohere[0].unsqueeze(0).unsqueeze(1).expand(1, num_pos - 1)  # (1, non_acc)
                nacc_cohere_ex = pos_cohere[1:].unsqueeze(0).expand(1, num_pos - 1)  # (1, non_acc)
                acc_triplet_loss = torch.max(-acc_cohere_ex + nacc_cohere_ex + self.pos_margin, zero_tensor)
                loss += torch.sum(acc_triplet_loss) / (num_pos - 1)
            # ================================================

        if torch.isnan(loss):
            author_ctx_embed = self.ctx_attention(token_temp_embed, self.author_embeddings, self.author_embeddings,
                                                  att_l=att_l)
            print("pos_labels:", pos_labels)
            raise ValueError("nan loss encountered")

        return loss, coherence