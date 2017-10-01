# coding=utf-8

import jieba.posseg as pseg
import global_params as gp


def split_sentence_to_word(sentence):
    result = []
    if sentence is None:
        return None
    sent = remove_line_breaks(sentence)
    word_flag_dict = pseg.cut(sent)
    for item in word_flag_dict:
        word = (item.word, item.flag)
        result.append(word)
    return result


def remove_line_breaks(sentence):
    temp = sentence.split('\n')
    sentence_without_line_breaks = ""
    for item in temp:
        sentence_without_line_breaks += item
    return sentence_without_line_breaks


def stopword_filter(word_list):
    words_without_stopwords = []
    stopword_list = load_stopwords()
    for word in word_list:
        if word not in stopword_list:
            words_without_stopwords.append(word)
    return words_without_stopwords


def load_stopwords():
    stopword_list = []
    stopwords_file = open(gp.USED_FILES_PREFIX_DIR + gp.STOP_WORDS_FILE, 'rb')
    for word in stopwords_file.readlines():
        stopword_list.append(word.split('\n')[0])
    return stopword_list


if __name__ == '__main__':
    print load_stopwords()
#   sentence = "健，云南玉溪塔集原董事，曾是有名的中烟草大王。\n 1928年，他出生于一个农民家庭。少年时期参加革命因反右不力被打成右派；"
#   print split_sentence_to_word(sentence)

