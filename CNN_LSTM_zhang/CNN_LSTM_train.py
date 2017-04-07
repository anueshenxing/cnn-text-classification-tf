# coding=utf8
import datetime
import time
import tensorflow as tf
from CNNLSTMModel import TextCNNLSTMModel as CNNLSTM
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
    params['lstm_size'] = 100
    params['batch_size'] = 100
    params['num_epochs'] = 10
    params['valid_freq'] = 100
    params['learning_rate'] = 0.0007

    step_of_train = []  # 训练步数
    train_loss = []  # 训练loss数据
    train_accuracy = []  # 训练accuracy数据
    step_of_valid = []  # 训练次数
    valid_loss = []  # 确认集loss数据
    valid_accuracy = []  # 确认集accuracy数据

    predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/data_file/"
    train_model_dir = "word2vec_100_withKeyword_LSTMCNN/"
    log_dir = predir + train_model_dir + "log.txt"
    log_file = open(log_dir, 'a')
    instruction = "word2vec训练的词向量，词向量维度为100，训练集为85000×0.8条新闻数据，有键词，模型为LSTM+CNN"
    log_file.write(instruction + "\n\n\n")

    print "加载数据......" + time.asctime(time.localtime(time.time()))
    all_title, all_keywords, all_label, title_max_len, keywords_max_len = load_data_title_keywords()
    title_train, title_test, keywords_train, keywords_test, label_train, label_test = \
        shuffled_data_title_keyword_lable(all_title, all_keywords, all_label)
    print "加载数据完成" + time.asctime(time.localtime(time.time()))

    print "构建CNN_LSTM模型" + time.asctime(time.localtime(time.time()))
    # title_length, num_keywords, num_classes, word_vector_size, filter_sizes, num_filters,
    # lstm_size, batch_size,
    TCNNLSTM = CNNLSTM(title_max_len, keywords_max_len, params)
    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    grads_and_vars = optimizer.compute_gradients(TCNNLSTM.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "CNN_LSTM模型构建完成" + time.asctime(time.localtime(time.time()))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    def train_step(title_batch, keywords_batch, label_batch):
        """
        A single training step
        """
        feed_dict = {
            TCNNLSTM.input_title: title_batch,
            TCNNLSTM.input_keywords: keywords_batch,
            TCNNLSTM.input_label: label_batch,
            TCNNLSTM.dropout: 0.5
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, TCNNLSTM.loss, TCNNLSTM.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        step_of_train.append(step)
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


    def dev_step(title_test, keywords_test, label_test, writer=None):
        """
        Evaluates model on a dev set
        """
        num_b = float(len(title_test) / params['batch_size'])
        loss_sum = 0
        accuracy_sum = 0
        valid_batches = batch_iter(
            list(zip(title_test, keywords_test, label_test)), params['batch_size'], 1)
        for dev_batch in valid_batches:
            title_batch, keywords_batch, label_batch = zip(*dev_batch)
            feed_dict = {
                TCNNLSTM.input_title: title_batch,
                TCNNLSTM.input_keywords: keywords_batch,
                TCNNLSTM.input_label: label_batch,
                TCNNLSTM.dropout: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, TCNNLSTM.loss, TCNNLSTM.accuracy],
                feed_dict)
            loss_sum += loss
            accuracy_sum += accuracy
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}".format(time_str, loss_sum / num_b, accuracy_sum / num_b))
        valid_accuracy.append(accuracy_sum / num_b)
        valid_loss.append(loss_sum / num_b)

    # Generate batches
    batches = batch_iter(list(zip(title_train, keywords_train, label_train)), params['batch_size'],
                         params['num_epochs'])

    for batch in batches:
        title_train_batch, keywords_train_batch, label_train_batch = zip(*batch)
        train_step(title_train_batch, keywords_train_batch, label_train_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % params['valid_freq'] == 0:
            print("\nEvaluation:\n")
            dev_step(title_test, keywords_test, label_test)
            step_of_valid.append(current_step / params['valid_freq'])
            print("\n")

    train__ = [step_of_train, train_loss, train_accuracy]
    valid__ = [step_of_valid, valid_loss, valid_accuracy]
    save__ = [train__, valid__]
    name__ = "CNN_LSTM_Model_result.p"
    save_data(save__, name__)


def CNN_LSTM():
    params = defaultdict()
    params['num_classes'] = 11
    params['word_vector_size'] = 100
    params['filter_sizes'] = [1, 2, 3]
    params['num_filters'] = 100
    params['lstm_size'] = 100
    params['batch_size'] = 100
    params['num_epochs'] = 30
    params['valid_freq'] = 100
    params['learning_rate'] = 0.01

    step_of_train = []  # 训练步数
    train_loss = []  # 训练loss数据
    train_accuracy = []  # 训练accuracy数据
    step_of_valid = []  # 训练次数
    valid_loss = []  # 确认集loss数据
    valid_accuracy = []  # 确认集accuracy数据

    print "加载数据......" + time.asctime(time.localtime(time.time()))
    all_title, all_keywords, all_label, title_max_len, keywords_max_len = load_data_title_keywords()
    title_train, title_test, keywords_train, keywords_test, label_train, label_test = \
        shuffled_data_title_keyword_lable(all_title, all_keywords, all_label)
    print "加载数据完成" + time.asctime(time.localtime(time.time()))

    print "构建CNN_LSTM模型" + time.asctime(time.localtime(time.time()))
    TCNNLSTM = CNNLSTM(title_max_len, keywords_max_len, params)
    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    grads_and_vars = optimizer.compute_gradients(TCNNLSTM.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "CNN_LSTM模型构建完成" + time.asctime(time.localtime(time.time()))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    def train_step(title_batch, keywords_batch, label_batch):
        """
        A single training step
        """
        feed_dict = {
            TCNNLSTM.input_title: title_batch,
            TCNNLSTM.input_keywords: keywords_batch,
            TCNNLSTM.input_label: label_batch,
            TCNNLSTM.dropout: 0.5
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, TCNNLSTM.loss, TCNNLSTM.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        step_of_train.append(step)
        if step % 700 == 0:
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


    def dev_step(title_test, keywords_test, label_test, writer=None):
        """
        Evaluates model on a dev set
        """
        num_b = float(len(title_test) / params['batch_size'])
        loss_sum = 0
        accuracy_sum = 0
        valid_batches = batch_iter(
            list(zip(title_test, keywords_test, label_test)), params['batch_size'], 1)
        for dev_batch in valid_batches:
            title_batch, keywords_batch, label_batch = zip(*dev_batch)
            feed_dict = {
                TCNNLSTM.input_title: title_batch,
                TCNNLSTM.input_keywords: keywords_batch,
                TCNNLSTM.input_label: label_batch,
                TCNNLSTM.dropout: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, TCNNLSTM.loss, TCNNLSTM.accuracy],
                feed_dict)
            loss_sum += loss
            accuracy_sum += accuracy
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}".format(time_str, loss_sum / num_b, accuracy_sum / num_b))
        valid_accuracy.append(accuracy_sum / num_b)
        valid_loss.append(loss_sum / num_b)

    # Generate batches
    batches = batch_iter(list(zip(title_train, keywords_train, label_train)), params['batch_size'],
                         params['num_epochs'])

    for batch in batches:
        title_train_batch, keywords_train_batch, label_train_batch = zip(*batch)
        train_step(title_train_batch, keywords_train_batch, label_train_batch)
        current_step = tf.train.global_step(sess, global_step)

        # if current_step % params['valid_freq'] == 0:
        #     print("\nCNN_LSTM Evaluation:\n")
        #     dev_step(title_test, keywords_test, label_test)
        #     step_of_valid.append(current_step / params['valid_freq'])
        #     print("\n")

    train__ = [step_of_train, train_loss, train_accuracy]
    valid__ = [step_of_valid, valid_loss, valid_accuracy]
    save__ = [train__, valid__]
    name__ = "dropout_CNN_LSTM_Model_result_0.1.p"
    save_data(save__, name__)
