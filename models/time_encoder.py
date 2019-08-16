import math
import sys
from datetime import timedelta

import seaborn
import torch
import torch.nn as nn
from torch.autograd import Variable

import br_utils

seaborn.set_context(context="talk")
sys.path.insert(0, "../")
# from br_util import to_variable
from pandas import Timestamp


def gen_time_encoding(time_encoder, dates):
    # transfer the date into time embedding
    time_embed = [time_encoder.get_time_encoding(i) for i in dates]
    time_embed = torch.stack(time_embed, dim=0)  # (n, d)
    return br_utils.to_variable(time_embed, requires_grad=False)

class TimeEncoder:
    """Implement the Time Encoding Function"""

    def __init__(self, d_model, dropout, span, date_range):
        super(TimeEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.decay_rate = 1
        self.span = span

        # support month span only
        self.from_date, self.to_date = date_range
        self.date_margin = 10
        max_time_span = self.temporal_index(self.to_date) + 1 + self.date_margin

        # Compute the time encodings once in log space.
        self.time_encode = torch.zeros(max_time_span, d_model)
        position = torch.arange(0., max_time_span).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.time_encode[:, 0::2] = torch.sin(position * div_term)
        self.time_encode[:, 1::2] = torch.cos(position * div_term)

    def temporal_index(self, date):
        month_span = (date.year - self.from_date.year) * 12 + date.month - self.from_date.month
        idx = int(month_span / self.span) + self.date_margin
        return idx

    def get_all_encoding(self):
        return self.time_encode

    def get_post_encodings(self, date):
        temp_idx = self.temporal_index(date)
        post_encodings = self.time_encode[temp_idx:, :]
        from br_utils import to_variable
        return to_variable(post_encodings, requires_grad=False)

    def get_time_encoding(self, date, num_shift=0):
        time_encode = self.time_encode[self.temporal_index(date), :]
        for i in range(1, num_shift + 1):
            # print("temp_idx:", self.temporal_index(date) - i, self.temporal_index(date) + i)
            time_encode_i_pre = self.time_encode[self.temporal_index(date) - i, :]
            time_encode_i_post = self.time_encode[self.temporal_index(date) + i, :]
            time_encode += (time_encode_i_pre + time_encode_i_post) * (self.decay_rate ** i)
        time_encode /= (num_shift * 2 + 1)
        from br_utils import to_variable
        return to_variable(time_encode, requires_grad=False)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

def test_time_encoding():
    date_range = (Timestamp(2008, 9, 2), Timestamp(2019, 4, 2))
    # date_range = (Timestamp(2008, 9, 2), Timestamp(2008, 12, 2))
    time_encoder_mgr = TimeEncoder(786, 0.1, span=1, date_range=date_range)

    e1 = time_encoder_mgr.get_time_encoding(Timestamp(2018, 1, 1))
    e2 = time_encoder_mgr.get_time_encoding(Timestamp(2018, 2, 1))
    e3 = time_encoder_mgr.get_time_encoding(Timestamp(2018, 3, 1))
    e4 = time_encoder_mgr.get_time_encoding(Timestamp(2018, 4, 1))
    e5 = time_encoder_mgr.get_time_encoding(Timestamp(2018, 5, 1))
    e6 = time_encoder_mgr.get_time_encoding(Timestamp(2018, 6, 1))

    norm = 1
    dist_12 = torch.dist(e1, e2, norm)
    dist_13 = torch.dist(e1, e3, norm)
    dist_14 = torch.dist(e1, e4, norm)
    dist_15 = torch.dist(e1, e5, norm)
    dist_16 = torch.dist(e1, e6, norm)

    dist_23 = torch.dist(e2, e3, norm)
    dist_24 = torch.dist(e2, e4, norm)
    dist_25 = torch.dist(e2, e5, norm)
    dist_26 = torch.dist(e2, e6, norm)

    new_idx = time_encoder_mgr.temporal_index(Timestamp(2019, 1, 2))
    date_list = [Timestamp(2008, 9, 2), Timestamp(2008, 10, 2), Timestamp(2008, 11, 2), Timestamp(2008, 12, 2), Timestamp(2009, 1, 2),
                 Timestamp(2009, 2, 2), Timestamp(2009, 3, 2), Timestamp(2009, 4, 2), Timestamp(2009, 5, 2), Timestamp(2009, 6, 2),
                 Timestamp(2009, 7, 2), Timestamp(2009, 8, 2), Timestamp(2009, 9, 2), Timestamp(2009, 10, 2), Timestamp(2009, 11, 2),
                 Timestamp(2009, 12, 2)]
    for date in date_list:
        print(date, time_encoder_mgr.temporal_index(date))


def draw_time_encoding():
    date_range = (Timestamp(2008, 9, 2), Timestamp(2019, 4, 2))
    time_encoder_mgr = TimeEncoder(786, 0.1, span=1, date_range=date_range)
    encoding = time_encoder_mgr.get_all_encoding()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(encoding)
    ax.set_xlabel('dimensions')
    ax.set_ylabel('time units')
    fig.show()
    fig.savefig("./time_encoding.pdf", bbox_inches='tight')


def main():
    draw_time_encoding()
    # test_time_encoding()

if __name__ == "__main__":
    main()

