# coding=utf8
import sys
import cPickle
import numpy as np
import tensorflow as tf

reload(sys)
sys.setdefaultencoding("utf-8")

def load_w2v_dict():
    predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    W_dir = predir + "word_vec_dict_true.p"
    W_p = cPickle.load(open(W_dir, 'rb'))
    W_original = W_p[0]
    W = np.zeros((len(W_original)+1, 100))
    W[:len(W_original)] = W_original
    W2v_dict = tf.Variable(W, dtype=tf.float32, name='W2v_dict')
    return W2v_dict

if __name__ == "__main__":
    # predir = "/home/CNN_zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    # W_dir = predir + "word_vec_dict_true.p"
    # W_p = cPickle.load(open(W_dir, 'rb'))
    # W = tf.Variable(W_p[0], name='w2v_dict')
    # input_x = [0, 2, 4, 6, 3, 2, 6, 8, 6]
    # word_embedding = tf.nn.embedding_lookup(W, input_x)
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # print sess.run([word_embedding])[0].shape
    load_w2v_dict()