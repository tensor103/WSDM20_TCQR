import operator

import numpy as np
import pandas as pd


def load_data(file_path):
    df = pd.read_pickle(file_path)

    author_freq_dict = {}
    for i, row in df.iterrows():

        # title & description
        title = (row["title"])
        desc = (row["desc"])
        date = row["create_time"]
        authors = list(set(row["author_list"]))
        if len(authors) == 0:
            continue

        for author in authors:
            if author in author_freq_dict:
                author_freq_dict[author] += 1
            else:
                author_freq_dict[author] = 1

        assignee = row["assignee"]

        # print(date.strftime("%Y-%m-%d"), authors, assignee)

    sorted_x = sorted(author_freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    for i in sorted_x:
        print(i[0], '\t', i[1])


def main():
    DATA_ROOT = "../data/"
    # PROJECT = "COLLECTIONS"
    PROJECT = "HIVE"
    PROJECT_PATH = DATA_ROOT + PROJECT + "/"

    TRAIN_CROSS_FILE = PROJECT_PATH + "train_cross.pkl"
    TEST_CROSS_FILE = PROJECT_PATH + "test_cross.pkl"


    print("=========== Training set ===========")
    load_data(TRAIN_CROSS_FILE)

    print("=========== Testing set ===========")
    load_data(TEST_CROSS_FILE)

if __name__ == "__main__":
    main()
