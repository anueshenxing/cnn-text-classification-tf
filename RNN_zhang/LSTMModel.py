# coding=utf8
import datetime
import os
import time

import tensorflow as tf

import util.util as util
from util.load_data import *

reload(sys)
sys.setdefaultencoding("utf-8")


class TextLSTMModel(object):
    def __init__(self, lstm_size, sequence_length, num_classes, wordvec_size, batch_zize):
        # 定义LSTM层的输入
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        # 词语标号转换为词向量
        W2v_dict = util.load_w2v_dict()
        # shape(None, sequence_length, word_vector_size)
        self.embedded_chars = tf.nn.embedding_lookup(W2v_dict, self.input_x)


        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Initial state of the LSTM memory.
        initial_state = state = lstm_cell.zero_state(40, tf.float32)
        with tf.variable_scope("myrnn") as scope:
            for time_step in range(sequence_length):
                if time_step > 0:
                    scope.reuse_variables()
                cell_output, state = lstm_cell(self.embedded_chars[:, time_step, :], state)
                self.final_output = cell_output

        # dropout层
        self.h_dropout = tf.nn.dropout(self.final_output, self.dropout)

        # Final (unnormalized) scores and predictions
        W_s = tf.Variable(tf.truncated_normal([wordvec_size, num_classes], stddev=0.1), name='W_s')
        b_s = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b_s')

        self.scores = tf.nn.xw_plus_b(self.h_dropout, W_s, b_s, name="scores")
        self.prediction = tf.argmax(self.scores, 1, name='prediction')

        # 计算交叉熵
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
        self.loss = tf.reduce_mean(losses)

        # 准确率
        correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


if __name__ == "__main__":

    predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/data_file/"
    train_model_dir = "word2vec_100_withoutKeyword_RNN/"
    log_dir = predir + train_model_dir + "log.txt"
    log_file = open(log_dir, 'a')
    instruction = "word2vec训练的词向量，词向量维度为100，训练集为85000×0.8条新闻数据，无关键词，模型为LSTM"
    log_file.write(instruction + "\n\n\n")

    print "加载数据......" + time.asctime(time.localtime(time.time()))
    data_by_id, data_label = load_data()
    data_train, train_label, data_test, test_label = shuffled_data(data_by_id, data_label)
    print "加载数据完成" + time.asctime(time.localtime(time.time()))
    print "构建LSTM模型" + time.asctime(time.localtime(time.time()))
    LSTM = TextLSTMModel(100, 39, 11, 100, 40)
    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(LSTM.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "LSTM模型构建完成" + time.asctime(time.localtime(time.time()))

    sess = tf.Session()

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = log_dir + "checkpoint"
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    sess.run(tf.initialize_all_variables())


    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            LSTM.input_x: x_batch,
            LSTM.input_y: y_batch,
            LSTM.dropout: 0.5
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, LSTM.loss, LSTM.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        log_file.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy) + "\n")


    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        num_b = float(len(x_batch)/40)
        loss_sum = 0
        accuracy_sum = 0
        valid_batches = batch_iter(
            list(zip(x_batch, y_batch)), 40, 1)
        for dev_batch in valid_batches:
            x_test, y_test = zip(*dev_batch)
            feed_dict = {
                LSTM.input_x: x_test,
                LSTM.input_y: y_test,
                LSTM.dropout: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, LSTM.loss, LSTM.accuracy],
                feed_dict)
            loss_sum += loss
            accuracy_sum += accuracy
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}".format(time_str, loss_sum/num_b, accuracy_sum/num_b))
        log_file.write("{}: loss {:g}, acc {:g}".format(time_str, loss_sum/num_b, accuracy_sum/num_b) + '\n')


    # Generate batches
    batches = batch_iter(list(zip(data_train, train_label)), 40, 10)

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % 100 == 0:
            print("\nEvaluation:\n")
            log_file.write("\nEvaluation:\n")
            dev_step(data_test[:16960], test_label[:16960])
            print("\n")
            log_file.write('\n')
        if current_step % 500 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            # print("Saved model checkpoint to {}\n".format(path))
            log_file.write("Saved model checkpoint to {}\n".format(path))
            log_file.write("\n")
