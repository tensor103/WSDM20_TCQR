import argparse
import operator
import random

import pandas as pd
from datetime import datetime
import pytz
from bs4 import BeautifulSoup
from lxml import etree
from random import shuffle
# sys.path.append("./")


parser = argparse.ArgumentParser(description='Bug Triaging Model')

parser.add_argument('--ratio_train', type=float, default=0.7, help='validation rate [default: 0.7]')
parser.add_argument('--ratio_val', type=float, default=0.1, help='validation rate [default: 0.1]')
parser.add_argument('--data_type', type=str, default='seq',
                    help='choose training type to run [options: seq, rand]')
parser.add_argument('--dataset', type=str, default='stackex_bioinformatics',
                    help='choose dataset to run [options: stackex_ai, stackex_bioinformatics, stackex_3dprinting, '
                         'stackex_ebooks, stackex_history, stackex_philosophy]')
parser.add_argument('--data_path', type=str, default='../data/', help='the data directory')

args = parser.parse_args()


if args.data_type == "seq":
    type_suffix = ""
elif args.data_type == "rand":
    type_suffix = "_rand"
elif args.data_type == "temptest":
    type_suffix = "_temptest"

PROJECT_PATH = args.data_path + args.dataset + "/"
TRAIN_FILE = PROJECT_PATH + "train" + type_suffix + ".pkl"
VAL_FILE = PROJECT_PATH + "val" + type_suffix + ".pkl"
TEST_FILE = PROJECT_PATH + "test" + type_suffix + ".pkl"
USER_ID_FILE = PROJECT_PATH + "user_ids.txt"

def clean_html(x):
    return BeautifulSoup(x, 'lxml').get_text()


def parse_question(elem):
    id = elem['Id']
    title = elem['Title']
    content = clean_html(elem['Body'])
    date = parse_date(elem['CreationDate'])

    if 'AcceptedAnswerId' in elem:
        accepted_ans_id = elem['AcceptedAnswerId']
    else:
        accepted_ans_id = ""
    # print(accepted_ans_id)
    return id, title, content, date, accepted_ans_id

def parse_answer(elem):
    parent_id = elem['ParentId']
    id = elem['Id']
    content = clean_html(elem['Body'])
    user_id = ""
    if 'OwnerUserId' in elem:
        user_id = elem['OwnerUserId']
    elif 'OwnerDisplayName' in elem:
        user_id = elem['OwnerDisplayName']
    else:
        pass
    score = elem['Score']
    date = parse_date(elem['CreationDate'])

    return parent_id, id, content, user_id, score, date

# identify the cold users
def get_active_usrs(post_file, min_usr_freq=5):

    parser = etree.iterparse(post_file, events=('end',), tag='row')
    usr_freq_dict = {}
    for i, (_, elem) in enumerate(parser):
        attr = dict(elem.attrib)
        if attr['PostTypeId'] == '2':  # answer post
            parent_id, id, content, user_name, score, date = parse_answer(attr)

            if not user_name:
                continue

            if user_name in usr_freq_dict:
                usr_freq_dict[user_name] += 1
            else:
                usr_freq_dict[user_name] = 1

    active_usrs, temp_users = [], []
    for usr_name, freq in usr_freq_dict.items():
        if freq >= min_usr_freq:
            active_usrs.append(usr_name)
        if freq == 1:
            temp_users.append(usr_name)
    return active_usrs, temp_users[:20]


# load data
def load_data(data_type, min_usr_freq=5):
    question_dict, question_order = {}, []
    usr_idx_dict = {}  # usr
    post_file = PROJECT_PATH + 'Posts.xml'
    active_usrs, temp_usrs = get_active_usrs(post_file, min_usr_freq=min_usr_freq)
    temp_question_ids, temp_usr_ids = [], []  # questions used for temporal test

    parser = etree.iterparse(post_file, events=('end',), tag='row')
    for i, (_, elem) in enumerate(parser):
        attr = dict(elem.attrib)

        # Output to separate files
        if attr['PostTypeId'] == '1':  # question post
            id, title, content, date, ans_id = parse_question(attr)
            # if question doesn't contain the accepted answer, skip the question
            if not ans_id:
                continue
            question_dict[id] = {"id": id, "title": title, "content": content, "date": date, "answers": [], "accept_ans": ans_id}
            question_order.append(id)
            pass
        elif attr['PostTypeId'] == '2':  # answer post
            parent_id, id, content, usr_name, score, date = parse_answer(attr)

            if (parent_id not in question_dict) or (not usr_name):
                continue

            if (usr_name not in active_usrs) and (usr_name not in temp_usrs):
                continue

            # assign user id
            if usr_name not in usr_idx_dict:
                usr_idx = len(usr_idx_dict.keys())
                usr_idx_dict[usr_name] = usr_idx
            else:
                usr_idx = usr_idx_dict[usr_name]

            answer_tuple = (id, content, usr_idx, score, date)
            question_dict[parent_id]["answers"].append(answer_tuple)

            if usr_name in temp_usrs:
                temp_question_ids.append(parent_id)



    # filter questions without any answers
    question_dict, num_removed = filter_unanswer_question(question_dict)

    # add data by its different data orders
    if data_type == "rand":
        shuffle(question_order)

    questions, temp_questions = [], []
    for id in question_order:
        if id not in question_dict:
            continue

        if id in temp_question_ids:
            temp_questions.append(question_dict[id])
        else:
            questions.append(question_dict[id])

    temp_usr_ids = [usr_idx_dict[i] for i in temp_usrs if i in usr_idx_dict]
    print("Total questions: ", str(len(questions)), " Users: ", str(len(usr_idx_dict.keys())), sep="")
    print("Temp Users IDs:", " ".join([str(i) for i in temp_usr_ids]))
    return questions, usr_idx_dict, temp_questions, temp_usr_ids


def filter_unanswer_question(question_dict):
    remove_ids = []
    for id, question in question_dict.items():

        answers = question["answers"]
        if len(answers) == 0:
            remove_ids.append(id)
            continue

    for id in remove_ids:
        del question_dict[id]
    return question_dict, len(remove_ids)


def parse_date(date_str):
    # # sample: Thu, 20 Sep 2018 23:46:28 +0000
    # return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    # sample: 2018-10-18T10:45:18.660
    return pytz.utc.localize(datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f'))


questions, user_dict, temp_questions, temp_usrs = load_data(args.data_type)

# write user id file
user_id_file = open(USER_ID_FILE, 'w')
sorted_user_ids = sorted(user_dict.items(), key=operator.itemgetter(1))
for i in sorted_user_ids:
    user_id_file.write(str(i[0]) + "\t" + str(i[1]) + "\n")

data_size = len(questions)
num_train, num_val = int(data_size * args.ratio_train), int(data_size * args.ratio_val)


train_indices = random.sample(range(num_train + num_val), num_train)
val_indices = [i for i in range(num_train + num_val) if i not in train_indices]

train_questions = [questions[i] for i in train_indices]
val_questions = [questions[i] for i in val_indices]

if args.data_type == "temptest":
    train_df = pd.DataFrame(train_questions + temp_questions)
else:
    train_df = pd.DataFrame(train_questions)
val_df = pd.DataFrame(val_questions)
test_df = pd.DataFrame(questions[num_train + num_val:])

# save data to data frame file
train_df.to_pickle(TRAIN_FILE)
val_df.to_pickle(VAL_FILE)
test_df.to_pickle(TEST_FILE)



