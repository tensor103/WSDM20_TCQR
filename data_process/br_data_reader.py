from typing import *

import numpy as np
import pandas as pd
from allennlp.data import Instance, Field
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides


def load_br_data(data_root, assign_id_file, encoder, train_type, max_seq_len, toy_data, is_rank_task=True):
    # the token indexer is responsible for mapping tokens to integers

    def bert_tokenizer(s: str):
        return token_indexer.wordpiece_tokenizer(s)[:max_seq_len - 2]

    def normal_tokenizer(x: str):
        return [w.text for w in
                SpacyWordSplitter(language='en_core_web_sm',
                                  pos_tags=False).split_words(x)[:max_seq_len]]

    if encoder == "bert":
        token_indexer = PretrainedBertIndexer(
            pretrained_model="bert-base-uncased",
            max_pieces=max_seq_len,
            do_lowercase=True,
        )
        tokenizer = bert_tokenizer
    else:
        token_indexer = SingleIdTokenIndexer()
        tokenizer = normal_tokenizer

    # init dataset reader
    reader = BRDataReader(
        assignee_id_file=assign_id_file,
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        is_rank_task=is_rank_task,
        toy_data=toy_data
    )

    if train_type == "seq":
        train_pkl, val_pkl, test_pkl = "train.pkl", "val.pkl", "test.pkl"
    elif train_type == "cross":
        train_pkl, val_pkl, test_pkl = "train_cross.pkl", "val_cross.pkl", "test_cross.pkl"
    # elif train_type == "single":
    #     train_pkl, test_pkl = "train_single.pkl", "test_single.pkl"

    train_ds, val_ds, test_ds = (reader.read(data_root / fname) for fname in [train_pkl, val_pkl, test_pkl])

    return reader, train_ds, val_ds, test_ds


class BRDataReader(DatasetReader):
    def __init__(self, assignee_id_file,
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = 300,
                 is_rank_task: bool = True,
                 toy_data: bool = False) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len
        self.is_rank_task = is_rank_task

        self.assignee_id_file = assignee_id_file
        self.num_authors = -1
        self.toy_data = toy_data

        # from_date, last
        self.from_date, self.to_date = pd.Timestamp(2100, 1, 1).tz_localize('US/Eastern'), \
                                       pd.Timestamp(1900, 1, 1).tz_localize('US/Eastern')

    @overrides
    def text_to_instance(self, tokens: List[Token], id: str,
                         authors: np.ndarray, assignee: int, date: pd.Timestamp) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        id_field = MetadataField(id)
        fields["id"] = id_field

        # if labels is None:
        #     labels = np.zeros(len(label_cols))
        # label_field = ArrayField(array=labels)
        # fields["label"] = label_field

        if self.is_rank_task:
            fields["label"] = MetadataField(authors)
        else:
            assignee_field = LabelField(assignee, skip_indexing=True)
            fields["label"] = assignee_field

        fields["date"] = MetadataField(date)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:

        df = pd.read_pickle(file_path)

        # if config.testing:
        if self.toy_data:
            df = df.head(100)

        for i, row in df.iterrows():

            # title & description
            title = (row["title"])
            desc = (row["desc"])
            date = row["create_time"]
            tokens = [Token(x) for x in self.tokenizer(row["title"] + row["desc"])]
            author_list = list(set(row["author_list"]))
            if len(author_list) == 0:
                continue
            assignee = row["assignee"]

            if date < self.from_date: self.from_date = date
            if date > self.to_date: self.to_date = date

            yield self.text_to_instance(
                tokens, row["id"], author_list, assignee, date
            )

    def get_author_num(self):
        if self.num_authors < 0:
            self.num_authors = sum([1 for i in open(self.assignee_id_file, "r").readlines() if i.strip()])
        return self.num_authors

    def get_time_span(self):
        return self.from_date, self.to_date


