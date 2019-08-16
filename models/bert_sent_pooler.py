import torch
from allennlp.modules import Seq2VecEncoder
from overrides import overrides

class BertSentencePooler(Seq2VecEncoder):
    def __init__(self, vocab, dim):
        super().__init__(vocab)
        self.bert_dim = dim

    def forward(self, embs: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        return torch.transpose(embs, 1, 2)

    @overrides
    def get_output_dim(self) -> int:
        return self.bert_dim