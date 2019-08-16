from pathlib import Path

from data_process.stackex_data_reader import load_stackex_data

data_path = "../data/"
PROJECT = "stackex_history"
# DATA_ROOT = Path("../dataset/bugtriaging/") / PROJECT
DATA_ROOT = Path(data_path) / PROJECT
data_type = "seq"
USER_ID_FILE = DATA_ROOT / "user_ids.txt"
max_seq_len = 300
toy_data = False
encoder = "bert"


reader, train_ds, val_ds, test_ds = load_stackex_data(DATA_ROOT, USER_ID_FILE, encoder, data_type,
                                                      max_seq_len, toy_data)

temp_usr = 5  # 5, 8, 10
temp_usrs = [5, 8, 10, 15, 16, 18, 20, 21, 25, 26, 28, 29, 30, 32, 33, 36, 38]
# temp_usrs = [5, 8, 10, 32]
all_answerers = []
for qi, rec in enumerate(train_ds):
    date = rec["date"].metadata
    answerers = [i[0] for i in rec["answerers"].metadata]
    if len(answerers) > 5:
        print(qi, len(answerers))

    # for temp_usr in temp_usrs:
    #     if temp_usr in answerers:
    #         print(temp_usr, date, answerers)
    # all_answerers += answerers
    # print(answerers, "|", rec["accept_usr"].metadata[0])

freq = sum([1 for i in all_answerers if i == 5])
pass