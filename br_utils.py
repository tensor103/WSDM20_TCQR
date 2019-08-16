
import os
from datetime import datetime

import torch
from torch.autograd import Variable

from models.tempq_ctx_model import TempQuestCtxModel
from models.tempx_ctx_model import TempXCtxModel


def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def save_snapshot(snap_path, save_prefix, model, vocab):

    if not os.path.isdir(snap_path):
        os.makedirs(snap_path)
    save_prefix = os.path.join(snap_path, save_prefix)
    save_path = '{}.pt'.format(save_prefix)

    print("Saving model to", save_path)
    with open(save_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(save_prefix + "vocab")

#save_snapshot(args.snapshot_path, "bert_rank", model, vocab)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def init_model(args, model_type, num_authors, vocab, encoder, max_vocab_size, date_span, ignore_time, num_sk):

    from models.bert_classifier import BertClassifier
    from models.bert_ctx_ranker import BertCtxRanker
    from models.bert_noctx_ranker import BertNoCtxRanker
    from models.lstm_classifier import LSTMClassifier
    from models.mspan_temp_model import MultiSpanTempModel
    from models.multihead_ctx_model import MultiHeadCtxModel
    from models.shift_temp_model import ShiftTempModel
    from models.temp_ctx_model import TempCtxModel

    model = None
    if model_type == "mspan_temp":
        model = MultiSpanTempModel(
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab,
            date_span=date_span,
            num_shift=args.num_shift,
            spans=args.spans,
            encoder=encoder,
            max_vocab_size=max_vocab_size,
            ignore_time=ignore_time,
            num_sk=num_sk
        )
    elif model_type == "shift_temp":
        model = ShiftTempModel(
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab,
            date_span=date_span,
            num_shift=args.num_shift,
            span=1
        )
    elif model_type == "tempx_ctx":
        model = TempXCtxModel(
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab,
            date_span=date_span
        )
    elif model_type == "tempq_ctx":
        model = TempQuestCtxModel(
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab,
            date_span=date_span
        )
    elif model_type == "temp_ctx":
        model = TempCtxModel(
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab,
            date_span=date_span
        )
    elif model_type == "mh_ctx":
        model = MultiHeadCtxModel(
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )
    elif model_type == "bert_ctx":
        model = BertCtxRanker(
            args=args,
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )
    elif model_type == "bert_noctx":
        model = BertNoCtxRanker(
            args=args,
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )
    elif model_type == "lstm_classifier":
        model = LSTMClassifier(
            args=args,
            out_sz=num_authors,
            vocab=vocab
        )
    elif model_type == "bert_classifier":
        model = BertClassifier(
            args=args,
            out_sz=num_authors,
            vocab=vocab
        )

    return model


# auto complete the snapshot file path
def complete_snapshot(snapshot_path, snapshot):
    if not snapshot.startswith(snapshot_path):
        snapshot = snapshot_path + snapshot

    if not snapshot.endswith('.th'):
        snapshot = snapshot + "/best.th"
    return snapshot


# detect model name from path
def detect_model_from_path(snapshot):
    snap_folder = snapshot.split("/")[-2]
    model = "_".join(snap_folder.split("_")[0:2])
    return model


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))