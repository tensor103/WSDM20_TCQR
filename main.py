import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from GPUtil import GPUtil
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer

from br_utils import init_model, complete_snapshot, detect_model_from_path
from data_process.br_data_reader import load_br_data
from data_process.stackex_data_reader import load_stackex_data
from eval import eval_model, eval_temp_model
from pandas import Timestamp
logging.basicConfig(level=logging.ERROR)


parser = argparse.ArgumentParser(description='Bug Triaging Model')

# training parameters
# use lr=1e-4 for context based models, but lr=1e-3 for non-context models
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate [default: 1e-4]')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay [default: 0]')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for train [default: 50]')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training [default: 64]')
parser.add_argument('--seed', type=int, default=1, help='seed of random numbers [default: 1]')
parser.add_argument('--device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('--toy_data', action='store_true', default=False,
                    help='use an extremely small dataset for testing purpose [default: False]')
# parser.add_argument('--val_ratio', type=float, default=0.3, help='validation rate [default: 0.3]')
parser.add_argument('--data_type', type=str, default='seq',
                    help='choose training type to run [options: seq, cross, rand, single]')

parser.add_argument('--max_seq_len', type=int, default=300,
                    help='maximum sequence length which is necessary to limit memory usage [default: 200]')
parser.add_argument('--max_vocab_size', type=int, default=100000, help='maximum vocabulary size [default: 100000]')

parser.add_argument('--is_rank_task', type=bool, default=True,
                    help='whether use the ranking task, otherwise it is classification task [default: True]')

# datasets
parser.add_argument('--dataset', type=str, default='stackex_ai',
                    help='choose dataset to run [options: HIVE, COLLECTIONS, COMPRESS, stackex_ai, stackex_bioinformatics, '
                         'stackex_3dprinting, stackex_ebooks, stackex_history, stackex_philosophy]')
# model
parser.add_argument('--model', type=str, default="mspan_temp",
                    help='models: mspan_temp, shift_temp, tempx_ctx, tempq_ctx, temp_ctx, mh_ctx, bert_ctx, bert_noctx, '
                         'lstm_classifier, bert_classifier [default: bert_classifier]')
parser.add_argument('--bio', type=str, default="",
                    help='short bio to distinguish other models [default: ""]')
parser.add_argument('--num_sk', type=int, default=20, help='the number of user expertise areas [default: 20]')


# multi-span and shifted temporal models
parser.add_argument('--num_shift', type=int, default=3, help='the number of shifted time encoding [default: 10]')
parser.add_argument('--spans', type=str, default="1,2,3", help='the list string of spans [default: 1,2,3]')
parser.add_argument('--ignore_time', action='store_true', default=False, help='For ablation study: whether to use time in the model')

# data
parser.add_argument('--data_path', type=str, default='./data/', help='the data directory')
parser.add_argument('--encoder', type=str, default='bert', help='the type of encoder, bert or lstm')
parser.add_argument('--snapshot_path', type=str, default="./snapshot/",
                    help='path of snapshot models [default: ./snapshot/]')

# testing
parser.add_argument('--test', action='store_true', default=False, help='use test mode')
parser.add_argument('--temp_test', action='store_true', default=False, help='use temporal test mode')
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot, if only directory is given, use the best.th [default: None]')

# bundle testing
parser.add_argument('--bt', action='store_true', default=False, help='use bundle test mode')
parser.add_argument('--bt_path', type=str, default=None, help='file folder of model snapshots [default: None]')
parser.add_argument('--bt_max', type=int, default=100, help='maximum models are tested in the bt mode [default: 50]')
# parser.add_argument('--bt_min', type=int, default=0, help='maximum models are tested in the bt mode [default: 0]')

args = parser.parse_args()

# update args and print
args.cuda = torch.cuda.is_available()

PROJECT = args.dataset
# DATA_ROOT = Path("../dataset/bugtriaging/") / PROJECT
DATA_ROOT = Path(args.data_path) / PROJECT
USER_ID_FILE = DATA_ROOT / "user_ids.txt"
torch.manual_seed(args.seed)

# initialize the snapshot path
now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")
bio_str = "[" + args.bio + "]" if args.bio else ""
if not args.test and not args.temp_test:
    args.snapshot_path = args.snapshot_path + args.model + bio_str + "_" + PROJECT + "_" + args.data_type + "_" + date_time + "/"
if not args.toy_data and not args.test and not os.path.isdir(args.snapshot_path):
    os.makedirs(args.snapshot_path)

if len(args.spans) > 0:
    args.spans = [int(i) for i in args.spans.split(",")]
else:
    args.spans = []

print("========== Parameter Settings ==========")
for arg in vars(args):
    print(arg, "=", getattr(args, arg), sep="")
print("========== ========== ==========")

# load data
print("Loading data:", DATA_ROOT, " (", args.data_type, ")", sep="")
if PROJECT.startswith("stackex"):
    reader, train_ds, val_ds, test_ds = load_stackex_data(DATA_ROOT, USER_ID_FILE, args.encoder, args.data_type,
                                                          args.max_seq_len, args.toy_data)
else:
    reader, train_ds, val_ds, test_ds = load_br_data(DATA_ROOT, USER_ID_FILE, args.encoder, args.data_type,
                                                     args.max_seq_len, args.toy_data)
num_authors = reader.get_author_num()
print("User Number: ", str(num_authors))

# choose available gpu
if args.device < 0:
    avail_gpus = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.05)
    # avail_gpus = GPUtil.getFirstAvailable(maxMemory=0.9, maxLoad=0.9)
    if len(avail_gpus) == 0:
        print("No GPU available!!!")
        sys.exit()
    args.device = avail_gpus[0]
    print("### Use GPU", str(args.device), "###")
torch.cuda.set_device(args.device)


# val_size = int(len(train_ds) * args.val_ratio)
# val_ds = train_ds[-val_size:]
# train_ds = train_ds[:-val_size]

# if args.test_ratio != 1:
#     print("Testing ratio:", args.test_ratio)
#     test_size = int(len(test_ds) * args.test_ratio)
#     test_ds = test_ds[:test_size]


# prepare vocabulary
vocab = Vocabulary.from_instances(train_ds, max_vocab_size=args.max_vocab_size)

# prepare iterator
# iterator = BucketIterator(batch_size=args.batch_size,
#                           sorting_keys=[("tokens", "num_tokens")],
#                           )
iterator = BasicIterator(batch_size=args.batch_size)
# tell the iterator how to numericalize the text data
iterator.index_with(vocab)


date_span = reader.get_time_span()
print("Date span: ", date_span[0], "--", date_span[1])
if not args.test and not args.bt and not args.temp_test:
    print("Training Model ...")

    # initialize model
    model = init_model(args, args.model, num_authors, vocab, args.encoder, args.max_vocab_size, date_span, args.ignore_time, args.num_sk)

    # move model to gpu
    if args.cuda:
        model.cuda()

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    serialization_dir = args.snapshot_path if not args.toy_data else None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        serialization_dir=serialization_dir,
        validation_dataset=val_ds,
        validation_metric="+mrr",
        cuda_device=args.device if args.cuda else -1,
        num_epochs=args.epochs,
        num_serialized_models_to_keep=args.epochs,
        # grad_norm=0.5,
        # grad_clipping=1
    )
    metrics = trainer.train()

    # evaluate model after training
    print("loading best model for evaluation")
    model = init_model(args, args.model, num_authors, vocab, args.encoder, args.max_vocab_size, date_span, args.ignore_time, args.num_sk)
    with open(args.snapshot_path + "best.th", 'rb') as f:
        model.load_state_dict(torch.load(f))
    if args.cuda:
        model.cuda(args.device)
    print("Evaluation in validation data.")
    eval_model(model, vocab, val_ds, args.batch_size, args.device if args.cuda else -1)
    print("Evaluation in testing data.")
    eval_model(model, vocab, test_ds, args.batch_size, args.device if args.cuda else -1)

elif args.test:  # test the single model
    # Evaluation
    print("Evaluation ...")

    if not args.snapshot:
        print("No snapshot is provided!")
        exit(0)

    # auto-fill snapshot path
    args.snapshot = complete_snapshot(args.snapshot_path, args.snapshot)

    # auto-detect model
    args.model = detect_model_from_path(args.snapshot)

    # load model from file
    model = init_model(args, args.model, num_authors, vocab, args.encoder, args.max_vocab_size, date_span, args.ignore_time, args.num_sk)

    print("Loading model: ", args.snapshot)
    with open(args.snapshot, 'rb') as f:
        model.load_state_dict(torch.load(f))
    if args.cuda:
        model.cuda(args.device)

    # run evaluation
    # eval_model(model, vocab, test_ds, args.batch_size, args.device if args.cuda else -1)
    eval_model(model, vocab, train_ds, args.batch_size, args.device if args.cuda else -1)

elif args.bt:  # bundle test the models for all the epochs
    if not args.bt_path:
        print("Specify the snapshot directory!")
        sys.exit(-1)

    snapshot_dir = "./snapshot/" + args.bt_path + "/"
    output_file = snapshot_dir + "exp_result.txt"
    output_file = open(output_file, 'a')
    print("snapshot directory:", snapshot_dir)

    for i in range(args.bt_max):
        print("=== Testing model", str(i), "===")

        # check the file existence
        model_name = "model_state_epoch_" + str(i) + ".th"
        snapshot_file = snapshot_dir + model_name
        while 1:
            try:
                with open(snapshot_file, 'rb') as f:
                    model = init_model(args, args.model, num_authors, vocab, args.encoder, args.max_vocab_size, date_span, args.ignore_time, args.num_sk)
                    model.load_state_dict(torch.load(f))
                if args.cuda:
                    model.cuda(args.device)

                # run evaluation
                print("start evaluating", model_name)
                output_str = "====== model " + str(i) + " ======\n"
                output_str += eval_model(model, vocab, test_ds, args.batch_size, args.device if args.cuda else -1)
                output_file.write(output_str)
                output_file.flush()
                break
            except FileNotFoundError:
                # Keep preset values
                # print("file doesn't exist, sleep and retry.")
                print("sleep...", end=" ")
                time.sleep(5)
    output_file.close()
elif args.temp_test:

    if not args.snapshot:
        print("No snapshot is provided!")
        exit(0)

    # auto-fill snapshot path
    args.snapshot = complete_snapshot(args.snapshot_path, args.snapshot)

    # auto-detect model
    args.model = detect_model_from_path(args.snapshot)

    # load model from file
    model = init_model(args, args.model, num_authors, vocab, args.encoder, args.max_vocab_size, date_span, args.ignore_time, args.num_sk)

    with open(args.snapshot, 'rb') as f:
        model.load_state_dict(torch.load(f))
    if args.cuda:
        model.cuda(args.device)

    # start to generate temporal synthetic data
    temp_usr_ids = [5, 8, 10, 15, 16, 18, 20, 25, 26, 28, 29, 30, 32, 33, 36, 38]
    qids = [216, 544, 754]
    for qid in qids:
        print("qid:", qid)
        temp_ds, date_range = [], range(15)
        sample_data = train_ds[qid]
        for i in date_range:
            date = sample_data['date'].metadata
            year, month, day = date.year, date.month - 1, date.day
            new_date = Timestamp(year + int((month + i) / 12), (month + i) % 12 + 1, day)
            new_sample = copy.deepcopy(sample_data)
            new_sample['date'].metadata = new_date
            temp_ds.append(new_sample)
            pass

        eval_temp_model(model, vocab, temp_ds, args.batch_size, args.device if args.cuda else -1)

    pass
# predicted = np.argmax(test_preds, axis=1)
# show_results(truth, predicted)
pass
