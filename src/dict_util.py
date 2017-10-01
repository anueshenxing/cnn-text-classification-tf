# coding=utf-8

import global_params as gp
import cPickle
import os

from collections import defaultdict


def generate_word_dict(news_list):
    is_success = False
    vocab = defaultdict(float)
    for news in news_list:
        news_title = news.get('news_title', [])
        news_content = news.get('news_content', [])
        news_word_list = news_title + news_content
        for word in news_word_list:
            vocab[word] += 1

    get_word_by_id = defaultdict(int)
    get_id_by_word = defaultdict(int)

    id = 0
    for w in vocab.keys():
        get_id_by_word[w] = id
        get_word_by_id[id] = w
        id += 1

    cPickle.dump(get_word_by_id,
                 open(gp.PRODUCE_FILES_PREFIX_DIR + gp.GET_WORD_BY_ID, "wb"),
                 True)
    cPickle.dump(get_id_by_word,
                 open(gp.PRODUCE_FILES_PREFIX_DIR + gp.GET_ID_BY_WORD, "wb"),
                 True)

    word_is_key = gp.PRODUCE_FILES_PREFIX_DIR + gp.GET_WORD_BY_ID
    id_is_key = gp.PRODUCE_FILES_PREFIX_DIR + gp.GET_ID_BY_WORD

    if os.path.exists(word_is_key) and os.path.exists(id_is_key):
        is_success = True

    return is_success


def generate_word_vector_dict():
    pass


if __name__ == '__main__':
    news_list = [{"news_title": ["one", "two", "three"], "news_content": ["four", "five", "six"]},
                 {"news_title": ["seven", "eight", "nine"], "news_content": ["ten", "eleven", "twilve"]}]

    print generate_word_dict(news_list)

