## Introduction
This is a Pytorch implementation of temporal context-aware question routing model, as described in our WSDM 2020 paper submission:

**Temporal Context-Aware Representation Learning for Question Routing**

## Requirement
* python 3
* pytorch > 1.0
* numpy

<!-- ## Run the demo
```
python3 main.py
``` -->

## Datasets

We employed six real-world CQA datasets from [StackExchange](https://stackexchange.com/) to evaluate the performance of our model. All the datasets are publicly available from their [website](https://archive.org/details/stackexchange).

The details of the datasets are presented as follows, including the number of questions, number of answerers, and their start and end dates.

Dataset | Questions | Answerers | Time Range
------------ | ------------- | ------------ | -------------
*ai* | 1130 | 163 | 2016.08-2019.06
*bioinformatics* | 915 | 107 | 2017.05-2019.05
*3dprinting* | 963 | 120 | 2016.01-2019.05
*ebooks* | 368 | 74 | 2013.12-2019.05
*history* | 4807 | 473 | 2011.05-2019.05
*philosophy* | 4295 | 658 | 2011.04-2019.06

Each dataset contains all questions and their corresponding answer records, including the lists of answerers and the respondent who provided the accepted answer. Also, both question content and question raising timestamp are included in the datasets. We reserved the latest 20\% of the data in the order of question raising time for the testing set and randomly split the remainder between 70\% for training and 10\% for validation. Both the answerers and the accepted answer for each question will be used as ground truth for evaluating the performance of our question routing model.

You can specify a dataset as follows:
```
python3 main.py --dataset="stackex_ai"
```
#### Data Processing
You can run the data preprocessing scripts before training the model under the data_process folder as follows:
```
python3 stackex_preprocess.py --data_type='seq' --dataset='stackex_ai'
```
## Usage
```
python3 main.py -h
```

You will get:

```
Bug Triaging Model

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate [default: 1e-4]
  --weight_decay WEIGHT_DECAY
                        weight decay [default: 0]
  --epochs EPOCHS       number of epochs for train [default: 50]
  --batch_size BATCH_SIZE
                        batch size for training [default: 64]
  --seed SEED           seed of random numbers [default: 1]
  --device DEVICE       device to use for iterate data, -1 mean cpu [default:
                        -1]
  --toy_data            use an extremely small dataset for testing purpose
                        [default: False]
  --data_type DATA_TYPE
                        choose training type to run [options: seq, cross,
                        rand, single]
  --max_seq_len MAX_SEQ_LEN
                        maximum sequence length which is necessary to limit
                        memory usage [default: 200]
  --max_vocab_size MAX_VOCAB_SIZE
                        maximum vocabulary size [default: 100000]
  --is_rank_task IS_RANK_TASK
                        whether use the ranking task, otherwise it is
                        classification task [default: True]
  --dataset DATASET     choose dataset to run [options: HIVE, COLLECTIONS,
                        COMPRESS, stackex_ai, stackex_bioinformatics,
                        stackex_3dprinting, stackex_ebooks, stackex_history,
                        stackex_philosophy]
  --model MODEL         models: mspan_temp, shift_temp, tempx_ctx, tempq_ctx,
                        temp_ctx, mh_ctx, bert_ctx, bert_noctx,
                        lstm_classifier, bert_classifier [default:
                        bert_classifier]
  --num_sk NUM_SK       the number of user expertise areas [default: 20]
  --num_shift NUM_SHIFT
                        the number of shifted time encoding [default: 10]
  --spans SPANS         the list string of spans [default: 1,2,3]
  --ignore_time         For ablation study: whether to use time in the model
  --data_path DATA_PATH
                        the data directory
  --encoder ENCODER     the type of encoder, bert or lstm
  --snapshot_path SNAPSHOT_PATH
                        path of snapshot models [default: ./snapshot/]
  --test                use test mode
  --temp_test           use temporal test mode
  --snapshot SNAPSHOT   filename of model snapshot, if only directory is
                        given, use the best.th [default: None]
```

## Train
You can start the training of our model as follows:
```
python3 main.py --lr=1e-5 --batch_size=16 --dataset="stackex_ai"
```

## Test
After finish training the model, you can perform testing as follows:
```
python3 main.py --test --batch_size=4 --dataset="stackex_ai" --snapshot="./snapshot.pt"
```
The snapshot option means where to load your trained model. If you don't assign it, the model will start from scratch.
