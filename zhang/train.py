# coding=utf8
import sys
import time
import os
import datetime
import load_data
from CNNModel import TextCNNModel as TextCNN
import tensorflow as tf
reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    # 保存训练结果
    predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/data_file/"
    train_model_dir = "word2vec_100_withoutKeyword_CNN_123/"
    log_dir = predir + train_model_dir + "log.txt"
    log_file = open(log_dir, 'a')
    # 训练设置说明
    instruction = "word2vec训练的词向量，词向量维度为100，训练集为85000×0.8条新闻数据，无关键词"
    log_file.write(instruction + "\n\n\n")
    # Parameters
    # ==================================================
    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 500, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Evaluate model on dev set after this many steps (default: 100)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    log_file.write("Parameters:" + '\n')
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        log_file.write("{}={}".format(attr.upper(), value) + "\n")
        print("{}={}".format(attr.upper(), value))
    print("")
    log_file.write('\n')



    print "加载数据......" + time.asctime(time.localtime(time.time()))
    data_by_id, data_label = load_data.load_data()
    data_train, train_label, data_test, test_label = load_data.shuffled_data(data_by_id, data_label)
    print "加载数据完成" + time.asctime(time.localtime(time.time()))

    # 加载CNN模型
    cnn = TextCNN(39, 11,
                  FLAGS.embedding_dim,
                  list(map(int, FLAGS.filter_sizes.split(","))),
                  FLAGS.num_filters)

    # 定义训练过程
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess = tf.Session()

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = log_dir + "checkpoint"
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    # Initialize all variables
    sess.run(tf.initialize_all_variables())

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout: 0.5
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        log_file.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy) + "\n")


    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout: 1.0
        }
        step, loss, accuracy = sess.run(
            [global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        log_file.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy) + "\n")

    # Generate batches
    batches = load_data.batch_iter(
            list(zip(data_train, train_label)), FLAGS.batch_size, FLAGS.num_epochs)

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            # print("\nEvaluation:\n")
            log_file.write("\nEvaluation:\n")
            dev_step(data_test, test_label)
            # print("\n")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            # print("Saved model checkpoint to {}\n".format(path))
            log_file.write("Saved model checkpoint to {}\n".format(path))
            log_file.write("\n")
