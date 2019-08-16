from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("rank_recall")
class RankRecall(Metric):
    """
    Ranking recall
    """
    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ConfigurationError("Tie break in Categorical Accuracy "
                                     "can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.sum_recall_k = 0.
        self.total_count = 0.

    def __call__(self, predictions, gold_labels):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        # num_classes = predictions.size(-1)
        # if gold_labels.dim() != predictions.dim() - 1:
        #     raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
        #                              "found tensor of shape: {}".format(predictions.size()))
        # if (gold_labels >= num_classes).any():
        #     raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
        #                              "the number of classes.".format(num_classes))

        for i, labels in enumerate(gold_labels):

            rank = predictions[i, :]
            # precision k
            k = len(labels)
            rk_hit = sum([1 for j in rank[:k] if j in labels]) / k
            self.sum_recall_k += rk_hit

        self.total_count += len(gold_labels)


    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            recall_k = float(self.sum_recall_k) / float(self.total_count)
        else:
            recall_k = 0.0
        if reset:
            self.reset()
        return recall_k

    @overrides
    def reset(self):
        self.sum_recall_k = 0.0
        self.total_count = 0.0


@Metric.register("mrr")
class MRR(Metric):
    """
    Ranking recall
    """
    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ConfigurationError("Tie break in Categorical Accuracy "
                                     "can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.sum_mrr = 0.
        self.total_count = 0.

    def __call__(self, predictions, gold_labels):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """

        for i, truth in enumerate(gold_labels):
            pred = predictions[i, :]
            # precision k
            for i, usr_idx in enumerate(pred, start=1):
                if usr_idx in truth:
                    self.sum_mrr += float(1 / i)
                    break
        self.total_count += len(gold_labels)


    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            mrr = float(self.sum_mrr) / float(self.total_count)
        else:
            mrr = 0.0
        if reset:
            self.reset()
        return mrr

    @overrides
    def reset(self):
        self.sum_mrr = 0.0
        self.total_count = 0.0