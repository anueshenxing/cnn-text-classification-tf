# coding=utf8
import sys
import tensorflow as tf
import util

reload(sys)
sys.setdefaultencoding("utf-8")


class TextCNNModel(object):
    """
    短文本分类卷积神经网络模型
    """

    def __init__(self, sequence_length, num_classes, vocab_size, word_vector_size, filter_sizes, num_filters,
                 l2_reg_lamda=0.0):
        """
        :param sequence_length: 文本中包含词语的个数，一般指最大值
        :param num_classes: 文本类别个数
        :param vocab_size: 词典中词语总数
        :param word_vector_size: 词向量维度
        :param filter_size: 卷积核尺寸
        :param num_filters: 特征图个数
        :param l2_reg_lamda:
        """
        # 定义卷积神经网络的输入
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 词语标号转换为词向量
        W2v_dict = util.load_w2v_dict()
        self.embedded_chars = tf.nn.embedding_lookup(W2v_dict, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 建立卷积层和最大池化层
        pooled_outputs = []
        # 卷积层
        for filter_size in filter_sizes:
            filter_shape = [filter_size, word_vector_size, 1, num_filters]
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5), name='W_conv_' + str(filter_size))
            b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b_conv_' + str(filter_size)))
            conv = tf.nn.conv2d(self.embedded_chars_expanded, W_conv, strides=[1, 1, 1, 1], padding='VALID',
                                name='conv')
            # 使用激活函数进行非线性化
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv), name='relu')
            # 最大池化层
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name='pool')
            pooled_outputs.append(pooled)

        # 合并所有池化层输出
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout层
        self.h_dropout = tf.nn.dropout(self.h_pool_flat, [-1, num_filters_total])

        # Final (unnormalized) scores and predictions
        W_s = tf.get_variable('W_s', shape=[num_filters_total, num_classes],
                              initializer=tf.contrib.xavier_initializer())
        b_s = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b_s')
        l2_loss += tf.nn.l2_loss(W_s)
        l2_loss += tf.nn.l2_loss(b_s)
        self.scores = tf.nn.xw_plus_b(self.h_dropout, W_s, b_s)
        self.prediction = tf.argmax(self.scores, 1, name='prediction')

        # 计算交叉熵
        losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lamda * l2_loss

        # 准确率
        correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
