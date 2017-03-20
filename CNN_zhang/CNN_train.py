# coding=utf8
import datetime
import os
import time
import tensorflow as tf
from CNNModel import TextCNNModel as TCNN
from util.load_data import *

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
    params['data_dir'] = "news_title_category_with_keywords.p"

    print time.asctime(time.localtime(time.time())) + " 加载数据......"
    data_predir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    all_sentence, all_label, sentence_max_len = load_data(data_predir+params['data_dir'])
    sentence_train, label_train, sentence_test, label_test = shuffled_data(all_sentence, all_label)
    print time.asctime(time.localtime(time.time())) + " 数据加载完成......"

    predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/data_file/"
    train_model_dir = "word2vec_100_withoutKeyword_CNN_123/"
    log_dir = predir + train_model_dir + "log.txt"
    log_file = open(log_dir, 'a')
    instruction = "word2vec训练的词向量，词向量维度为100，训练集为85000×0.8条新闻数据，有键词，模型为LSTM+CNN"
    log_file.write(instruction + "\n")

    print "构建CNN模型" + time.asctime(time.localtime(time.time()))
    cnn = TCNN(sentence_max_len, params)
    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "CNN_LSTM模型构建完成" + time.asctime(time.localtime(time.time()))

    sess = tf.Session()

    # # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    # checkpoint_dir = log_dir + "checkpoint"
    # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # saver = tf.train.Saver(tf.all_variables())

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
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        # log_file.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy) + "\n")

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
        log_file.write("{}: loss {:g}, acc {:g}".format(time_str, loss_sum / num_b, accuracy_sum / num_b) + '\n')

    # Generate batches
    batches = batch_iter(list(zip(sentence_train, label_train)), params['batch_size'], params['num_epochs'])

    for batch in batches:
        title_train_batch, label_train_batch = zip(*batch)
        train_step(title_train_batch, label_train_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % params['valid_freq'] == 0:
            print("\nEvaluation:\n")
            # log_file.write("\nEvaluation:\n")
            dev_step(sentence_test, label_test)
            print("\n")
            # log_file.write('\n')
        # if current_step % 500 == 0:
        #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        #     # print("Saved model checkpoint to {}\n".format(path))
            # log_file.write("Saved model checkpoint to {}\n".format(path))
            # log_file.write("\n")
