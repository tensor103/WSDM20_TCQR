from typing import *

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


class LSTMClassifier(Model):
    def __init__(self, args, out_sz: int,
                 vocab: Vocabulary):
        super().__init__(vocab)

        # prepare embeddings
        token_embedding = Embedding(num_embeddings=args.max_vocab_size + 2,
                                    embedding_dim=300, padding_index=0)
        self.word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(self.word_embeddings.get_output_dim(),
                                                                     hidden_size=64, bidirectional=True, batch_first=True))

        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

        return output