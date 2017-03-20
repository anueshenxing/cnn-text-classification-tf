# coding=utf8
import sys
import cPickle
import tensorflow as tf
import numpy as np
reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    W_dir = predir + "word_vec_dict_true.p"
    word_ix_dir = predir + "wordtoix_and_ixtoword_true.p"
    data_t_dir = predir + "news_title_category.p"
    data_t_k_dir = predir + "news_title_category_with_keywords.p"
    data_t_k_p = cPickle.load(open(data_t_k_dir, 'rb'))
    data_t_p = cPickle.load(open(data_t_dir, 'rb'))
    data_t_k = data_t_k_p[0]
    data_t = data_t_p[0]
    tk = data_t_k[0].split('\n')[0]
    t = data_t[0].split('\n')[0]
    print len(tk.split(t))
    for i in tk.split(t):
        print "----" + i
    # W_p = cPickle.load(open(W_dir, 'rb'))
    # word_ix_p = cPickle.load(open(word_ix_dir, 'rb'))
    # ixtoword = word_ix_p[1]
    # wordtoix = word_ix_p[0]
    # W = W_p[0]
    # print wordtoix['Mobile']
    # s = "新 Win10   Mobile PC 预览版 本周 晚些时候 推送"
    # for i in s.split(" "):
    #     print i
    #     print i == ''
    # print len(ixtoword)
    # print ixtoword[0]
    # print np.zeros(11)
    # print len(W)
    # print W[517991]
    # W = tf.Variable(W_p[0], name='w2v_dict')
    # input_x = [[0, 2, 4, 6, 3, 2, 6, 8, 6], [0, 2, 4, 6, 3, 2, 6, 8, 6]]
    # word_embedding = tf.nn.embedding_lookup(W, input_x)
    # embedded_chars_expanded = tf.expand_dims(word_embedding, -1)
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # print sess.run([embedded_chars_expanded])[0].shape
