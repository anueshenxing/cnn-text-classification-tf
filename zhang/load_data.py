# coding=utf8
import sys
import cPickle
import numpy as np
from collections import defaultdict

reload(sys)
sys.setdefaultencoding("utf-8")


def news_ctg_dict():
    ctg_dict = defaultdict(int)
    ctg_list = ['society', 'edu', 'sports', 'travel', 'military', 'finance', 'tech', 'food', 'health', 'car',
                'entertainment']
    for index in range(len(ctg_list)):
        ctg_dict[ctg_list[index]] = index
    return ctg_dict

def load_word_ix():
    predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    word_ix_dir = predir + "wordtoix_and_ixtoword_true.p"
    word_ix_p = cPickle.load(open(word_ix_dir, 'rb'))
    wordtoix, ixtoword = word_ix_p[0], word_ix_p[1]
    return wordtoix, ixtoword

def load_data():
    predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    data_dir = predir + "news_title_category.p"
    data_p = cPickle.load(open(data_dir, 'rb'))
    data = data_p[0]
    wordtoix, ixtoword = load_word_ix()
    ctg_dict = news_ctg_dict()

    data_by_id = []
    data_label = []
    max_len = 0
    for index in range(len(data)):
        sentence_by_id = []
        sentence_label = np.zeros(11)
        # 获取新闻类别标签
        news_ctg = data[index].split("\n")[0].split(" ")[0]
        sentence_label[ctg_dict[news_ctg]] = 1.
        # print sentence_label
        data_label.append(sentence_label)

        # 将新闻标题句子编号化
        for word in data[index].split("\n")[0].split(" ")[1:]:
            if word != '':
                sentence_by_id.append(wordtoix[word])
        # print sentence_by_id
        if len(sentence_by_id) > max_len:
            max_len = len(sentence_by_id)
        data_by_id.append(sentence_by_id)
    # print max_len
    for index in range(len(data_by_id)):
        one_data = data_by_id[index]
        for i in range(max_len - len(one_data)):
            one_data.append(542255)
            # print len(one_data)
    return np.array(data_by_id), np.array(data_label)

def shuffled_data(x, y):
    np.random.seed(10)
    shuffled_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    train_data_num = len(y)/10 * 8
    data_train, train_label = x_shuffled[:train_data_num], y_shuffled[:train_data_num]
    data_test, test_label = x_shuffled[train_data_num:], y_shuffled[train_data_num:]
    return data_train, train_label, data_test, test_label


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    # wordtoix, ixtoword = load_word_ix()
    # data = load_data()
    # for i in data[0].split("\n")[0].split(" ")[1:]:
    #     print i + ": " + str(wordtoix[i])
    # ids = [wordtoix[i] for i in data[0].split("\n")[0].split(" ")[1:]]
    # print ids
    load_data()
