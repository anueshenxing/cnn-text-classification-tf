# coding=utf8
import datetime
import time
import tensorflow as tf
from CNN_LSTM_zhang.CNN_LSTM_train import *
from CNN_zhang.CNN_train import *
from RNN_zhang.LSTMModel import *
from util.load_data import *
reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    data_dir = ["news_title_category_with_keywords.p", "news_title_category.p"]
    CNN_LSTM()
    for d_dir in data_dir:
        LSTM(d_dir)
        CNN(d_dir)


