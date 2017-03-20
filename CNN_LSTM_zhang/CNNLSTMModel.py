# coding=utf8
import os
import time
import datetime
import tensorflow as tf
import CNN_zhang.util as util
from CNN_zhang.load_data import *

reload(sys)
sys.setdefaultencoding("utf-8")


class TextCNNLSTMModel(object):
    def __init__(self, title_length, num_keywords, params, l2_reg_lamda=0.0,):
        """
        :param sequence_length: 句子长度
        :param num_classes: 类别个数
        :param word_vector_size: 词向量维度
        :param filter_sizes: 卷积核尺寸
        :param num_filters: 特征图数量
        :param lstm_size: lstm units 个数
        :param batch_size: 每批数据个数
        :param l2_reg_lamda:
        """
        # 定义模型输入
        self.input_title = tf.placeholder(tf.int32, [None, title_length], name='input_x')
        self.input_keywords = tf.placeholder(tf.int32, [None, num_keywords], name='input_x')
        self.input_label = tf.placeholder(tf.float32, [None, params['num_classes']], name='input_y')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # word emmbedding Layer
        W2v_dict = util.load_w2v_dict()
        self.embedded_title = tf.nn.embedding_lookup(W2v_dict, self.input_title)
        self.embedded_keywords = tf.nn.embedding_lookup(W2v_dict, self.input_keywords)
        self.embedded_keywords_expanded = tf.expand_dims(self.embedded_keywords, -1)

        # 建立卷积层和最大池化层
        pooled_outputs = []
        # 卷积层
        for filter_size in params['filter_sizes']:
            filter_shape = [filter_size, params['word_vector_size'], 1, params['num_filters']]
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5), name='W_conv')
            b_conv = tf.Variable(tf.constant(0.1, shape=[params['num_filters']], name='b_conv_' + str(filter_size)))
            conv = tf.nn.conv2d(self.embedded_keywords_expanded, W_conv, strides=[1, 1, 1, 1], padding='VALID',
                                name='conv')
            # 使用激活函数进行非线性化
            h = tf.nn.tanh(tf.nn.bias_add(conv, b_conv), name='tanh')
            # 最大池化层
            pooled = tf.nn.max_pool(h, ksize=[1, num_keywords - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name='pool')
            pooled_outputs.append(pooled)

        # 合并所有池化层输出
        num_filters_total = params['num_filters'] * len(params['filter_sizes'])
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 建立LSTM层
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(params['lstm_size'])
        # 初始化 LSTM memory 的状态.
        initial_state = state = lstm_cell.zero_state(params['batch_size'], tf.float32)
        with tf.variable_scope("myrnn") as scope:
            for time_step in range(title_length):
                if time_step > 0:
                    scope.reuse_variables()
                cell_output, state = lstm_cell(self.embedded_title[:, time_step, :], state)
                self.final_output = cell_output

        # 合并CNN提取特征和LSTM提取特征
        self.all_features = tf.concat([self.h_pool_flat, self.final_output], 1)

        # dropout层
        self.h_dropout = tf.nn.dropout(self.all_features, self.dropout)

        # Final (unnormalized) scores and predictions
        all_features_size = params['num_filters'] * len(params['filter_sizes']) + params['lstm_size']
        W_s = tf.Variable(tf.truncated_normal([all_features_size, params['num_classes']], stddev=0.1), name='W_s')
        b_s = tf.Variable(tf.constant(0.1, shape=[params['num_classes']]), name='b_s')
        l2_loss += tf.nn.l2_loss(W_s)
        l2_loss += tf.nn.l2_loss(b_s)
        self.scores = tf.nn.xw_plus_b(self.h_dropout, W_s, b_s, name="scores")
        self.prediction = tf.argmax(self.scores, 1, name='prediction')

        # 计算交叉熵
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.scores)
        self.loss = tf.reduce_mean(losses) + l2_reg_lamda * l2_loss

        # 准确率
        correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
