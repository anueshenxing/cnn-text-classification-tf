# coding=utf8
import datetime
import os
import time
import tensorflow as tf
from CNNModel import TextCNNModel as TCNN
from util.load_data import *
from util.util import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    params = defaultdict()
    params['num_classes'] = 11
    params['word_vector_size'] = 100
    params['filter_sizes'] = [1, 2, 3]
    params['num_filters'] = 100
    params['batch_size'] = 100
    params['num_epochs'] = 10
    params['valid_freq'] = 100
    params['learning_rate'] = 0.0007
    params['data_dir'] = "news_title_category_with_keywords.p"

    step_of_train = []  # 训练步数
    train_loss = []  # 训练loss数据
    train_accuracy = []  # 训练accuracy数据
    step_of_valid = []  # 训练次数
    valid_loss = []  # 确认集loss数据
    valid_accuracy = []  # 确认集accuracy数据


    print time.asctime(time.localtime(time.time())) + " 加载数据......"
    data_predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    all_sentence, all_label, sentence_max_len = load_data(data_predir+params['data_dir'])
    sentence_train, label_train, sentence_test, label_test = shuffled_data(all_sentence, all_label)
    print time.asctime(time.localtime(time.time())) + " 数据加载完成......"

    print "构建CNN模型" + time.asctime(time.localtime(time.time()))
    cnn = TCNN(sentence_max_len, params)
    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "CNN_LSTM模型构建完成" + time.asctime(time.localtime(time.time()))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    def train_step(sentence_batch, label_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_sentence: sentence_batch,
            cnn.input_label: label_batch,
            cnn.dropout: 0.5
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        step_of_train.append(step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    def dev_step(sentence_test, label_test, writer=None):
        """
        Evaluates model on a dev set
        """
        num_b = float(len(sentence_test) / params['batch_size'])
        loss_sum = 0
        accuracy_sum = 0
        valid_batches = batch_iter(
            list(zip(sentence_test, label_test)), params['batch_size'], 1)
        for dev_batch in valid_batches:
            sentence_batch, label_batch = zip(*dev_batch)
            feed_dict = {
                cnn.input_sentence: sentence_batch,
                cnn.input_label: label_batch,
                cnn.dropout: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            loss_sum += loss
            accuracy_sum += accuracy
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}".format(time_str, loss_sum / num_b, accuracy_sum / num_b))
        valid_accuracy.append(accuracy_sum / num_b)
        valid_loss.append(loss_sum / num_b)

    # Generate batches
    batches = batch_iter(list(zip(sentence_train, label_train)), params['batch_size'], params['num_epochs'])

    for batch in batches:
        title_train_batch, label_train_batch = zip(*batch)
        train_step(title_train_batch, label_train_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % params['valid_freq'] == 0:
            step_of_valid.append(current_step / params['valid_freq'])
            print("\nEvaluation:\n")
            dev_step(sentence_test, label_test)
            print("\n")
    train__ = [step_of_train, train_loss, train_accuracy]
    valid__ = [step_of_valid, valid_loss, valid_accuracy]
    save__ = [train__, valid__]
    name__ = "CNN_" + "LSTM_" + params['data_dir']
    save_data(save__, name__)

def CNN(CNN_data_dir):
    params = defaultdict()
    params['num_classes'] = 11
    params['word_vector_size'] = 100
    params['filter_sizes'] = [1, 2, 3]
    params['num_filters'] = 100
    params['batch_size'] = 100
    params['num_epochs'] = 10
    params['valid_freq'] = 100
    params['learning_rate'] = 0.0007
    params['data_dir'] = CNN_data_dir

    step_of_train = []  # 训练步数
    train_loss = []  # 训练loss数据
    train_accuracy = []  # 训练accuracy数据
    step_of_valid = []  # 训练次数
    valid_loss = []  # 确认集loss数据
    valid_accuracy = []  # 确认集accuracy数据


    print time.asctime(time.localtime(time.time())) + " 加载数据......"
    data_predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    all_sentence, all_label, sentence_max_len = load_data(data_predir+params['data_dir'])
    sentence_train, label_train, sentence_test, label_test = shuffled_data(all_sentence, all_label)
    print time.asctime(time.localtime(time.time())) + " 数据加载完成......"

    print "构建CNN模型" + time.asctime(time.localtime(time.time()))
    cnn = TCNN(sentence_max_len, params)
    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "CNN模型构建完成" + time.asctime(time.localtime(time.time()))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    def train_step(sentence_batch, label_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_sentence: sentence_batch,
            cnn.input_label: label_batch,
            cnn.dropout: 0.5
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        step_of_train.append(step)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    def dev_step(sentence_test, label_test, writer=None):
        """
        Evaluates model on a dev set
        """
        num_b = float(len(sentence_test) / params['batch_size'])
        loss_sum = 0
        accuracy_sum = 0
        valid_batches = batch_iter(
            list(zip(sentence_test, label_test)), params['batch_size'], 1)
        for dev_batch in valid_batches:
            sentence_batch, label_batch = zip(*dev_batch)
            feed_dict = {
                cnn.input_sentence: sentence_batch,
                cnn.input_label: label_batch,
                cnn.dropout: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            loss_sum += loss
            accuracy_sum += accuracy
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}".format(time_str, loss_sum / num_b, accuracy_sum / num_b))
        valid_accuracy.append(accuracy_sum / num_b)
        valid_loss.append(loss_sum / num_b)

    # Generate batches
    batches = batch_iter(list(zip(sentence_train, label_train)), params['batch_size'], params['num_epochs'])

    for batch in batches:
        title_train_batch, label_train_batch = zip(*batch)
        train_step(title_train_batch, label_train_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % params['valid_freq'] == 0:
            step_of_valid.append(current_step / params['valid_freq'])
            print("\nCNN Evaluation:\n" + params['data_dir'])
            dev_step(sentence_test, label_test)
            print("\n")
    train__ = [step_of_train, train_loss, train_accuracy]
    valid__ = [step_of_valid, valid_loss, valid_accuracy]
    save__ = [train__, valid__]
    name__ = "CNN_" + "LSTM_" + params['data_dir']
    save_data(save__, name__)