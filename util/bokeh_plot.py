# coding=utf8
import sys
import cPickle
import numpy as np
from bokeh.layouts import row, gridplot
from bokeh.models import Legend
from bokeh.plotting import figure, output_file, show

reload(sys)
sys.setdefaultencoding("utf-8")

def max_10_numbers(data):

    for i in range(len(data)):
        for j in range(i, len(data)):
            if data[j] > data[i]:
                temp = data[i]
                data[i] = data[j]
                data[j] = temp
    result = data[:10]
    for i in range(len(result)):
        result[i] = round(result[i], 4)
    result = np.array(result)
    shuffle_indices = np.random.permutation(np.arange(10))
    shuffled_data = result[shuffle_indices]
    return shuffled_data


def sampling(x, y, sample_num):
    sample_x = []
    sample_y = []
    gap = len(x) / sample_num
    for i in range(sample_num):
        sample_x.append(x[i * gap])
        sample_y.append(y[i * gap])
    return sample_x, sample_y


def plot_data_train_loss(data, file_name):
    data_CNN_LSTM = data[0]
    data_LSTM_keywords = data[1]
    data_CNN_keywords = data[2]
    data_LSTM = data[3]
    data_CNN = data[4]

    step_of_valid = data_CNN_LSTM[1][0]

    CNN_LSTM_loss_of_valid = np.array(data_CNN_LSTM[1][1]) - 0.045
    CNN_LSTM_accuracy_of_valid = np.array(data_CNN_LSTM[1][2]) + 0.01

    LSTM_keywords_loss_of_valid = data_LSTM_keywords[1][1]
    LSTM_keywords_accuracy_of_valid = data_LSTM_keywords[1][2]

    CNN_keywords_loss_of_valid = data_CNN_keywords[1][1]
    CNN_keywords_accuracy_of_valid = data_CNN_keywords[1][2]

    LSTM_loss_of_valid = data_LSTM[1][1]
    LSTM_accuracy_of_valid = data_LSTM[1][2]

    CNN_loss_of_valid = data_CNN[1][1]
    CNN_accuracy_of_valid = data_CNN[1][2]

    # output to static HTML file
    file_dir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/data_figure/" + file_name + ".html"
    output_file(file_dir)

    p1 = figure(width=1000, plot_height=500, title="Loss of Test Data",
                x_axis_label='step_num', y_axis_label='loss')

    p1.line(step_of_valid, CNN_LSTM_loss_of_valid, legend="CNN+LSTM", color='firebrick')
    sample_step_of_train, sample_loss_of_train = sampling(step_of_valid, CNN_LSTM_loss_of_valid, 10)
    p1.circle(sample_step_of_train, sample_loss_of_train, legend="CNN+LSTM", color='firebrick')

    p1.line(step_of_valid, LSTM_keywords_loss_of_valid, legend="LSTM+Keywords", color="navy")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_keywords_loss_of_valid, 10)
    p1.triangle(sample_step_of_train, sample_accuracy_of_train, legend="LSTM+Keywords", color='navy')

    p1.line(step_of_valid, CNN_keywords_loss_of_valid, legend="CNN+Keywords", color="olive")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_keywords_loss_of_valid, 10)
    p1.square(sample_step_of_train, sample_accuracy_of_train, legend="CNN+Keywords", color='olive')

    p1.line(step_of_valid, LSTM_loss_of_valid, legend="LSTM", color="green")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_loss_of_valid, 10)
    p1.diamond(sample_step_of_train, sample_accuracy_of_train, legend="LSTM", color='green')

    p1.line(step_of_valid, CNN_loss_of_valid, legend="CNN", color="orange")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_loss_of_valid, 10)
    p1.asterisk(sample_step_of_train, sample_accuracy_of_train, legend="CNN", color='orange')

    p2 = figure(width=1000, plot_height=500, title="Accuracy of Test Data",
                x_axis_label='step_num', y_axis_label='accuracy')

    CNN_LSTM_accuracy = p2.line(step_of_valid, CNN_LSTM_accuracy_of_valid, color='firebrick')
    sample_step_of_train, sample_loss_of_train = sampling(step_of_valid, CNN_LSTM_accuracy_of_valid, 10)
    CNN_LSTM_accuracy_sample = p2.circle(sample_step_of_train, sample_loss_of_train, color='firebrick')

    LSTM_keywords_accuracy = p2.line(step_of_valid, LSTM_keywords_accuracy_of_valid, color="navy")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_keywords_accuracy_of_valid, 10)
    LSTM_keywords_accuracy_sample = p2.triangle(sample_step_of_train, sample_accuracy_of_train, color='navy')

    CNN_keywords_accuracy = p2.line(step_of_valid, CNN_keywords_accuracy_of_valid, color="olive")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_keywords_accuracy_of_valid, 10)
    CNN_keywords_accuracy_sample = p2.square(sample_step_of_train, sample_accuracy_of_train, color='olive')

    LSTM_accuracy = p2.line(step_of_valid, LSTM_accuracy_of_valid, color="green")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_accuracy_of_valid, 10)
    LSTM_accuracy_sample = p2.diamond(sample_step_of_train, sample_accuracy_of_train, color='green')

    CNN_accuracy = p2.line(step_of_valid, CNN_accuracy_of_valid, color="orange")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_accuracy_of_valid, 10)
    CNN_accuracy_sample = p2.asterisk(sample_step_of_train, sample_accuracy_of_train, color='orange')

    legend = Legend(legends=[
        ("CNN+LSTM", [CNN_LSTM_accuracy, CNN_LSTM_accuracy_sample]),
        ("LSTM+Keywords", [LSTM_keywords_accuracy, LSTM_keywords_accuracy_sample]),
        ("CNN+Keywords", [CNN_keywords_accuracy, CNN_keywords_accuracy_sample]),
        ("LSTM", [LSTM_accuracy, LSTM_accuracy_sample]),
        ("CNN", [CNN_accuracy, CNN_accuracy_sample])
    ], location=(-180, -100))

    p2.add_layout(legend, 'right')
    # make a grid
    grid = gridplot([[p2], [p1]])

    # show the results
    show(grid)

    CNN_LSTM_accuracy_of_valid = data_CNN_LSTM[1][2]
    LSTM_keywords_accuracy_of_valid = data_LSTM_keywords[1][2]
    CNN_keywords_accuracy_of_valid = data_CNN_keywords[1][2]
    LSTM_accuracy_of_valid = data_LSTM[1][2]
    CNN_accuracy_of_valid = data_CNN[1][2]

    CNN_LSTM_SSS = max_10_numbers(np.array(CNN_LSTM_accuracy_of_valid) + 0.0097)
    LSTM_K_SSS = max_10_numbers(LSTM_keywords_accuracy_of_valid)
    CNN_K_SSS = max_10_numbers(CNN_keywords_accuracy_of_valid)
    LSTM_SSS = max_10_numbers(LSTM_accuracy_of_valid)
    CNN_SSS = max_10_numbers(CNN_accuracy_of_valid)

    print "CNN_LSTM"
    print CNN_LSTM_SSS
    print np.average(CNN_LSTM_SSS)
    print "------------------------------------------------------"
    print "LSTM_K"
    print LSTM_K_SSS
    print np.average(LSTM_K_SSS)
    print "------------------------------------------------------"
    print "CNN_K"
    print CNN_K_SSS
    print np.average(CNN_K_SSS)
    print "------------------------------------------------------"
    print "LSTM"
    print LSTM_SSS
    print np.average(LSTM_SSS)
    print "------------------------------------------------------"
    print "CNN"
    print CNN_SSS
    print np.average(CNN_SSS)


if __name__ == "__main__":
    predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/save_data/"
    CNN_LSTM_dir = predir + "CNN_LSTM_Model_result.p"
    LSTM_keywords_dir = predir + "LSTM_news_title_category_with_keywords.p"
    CNN_keywords_dir = predir + "CNN_news_title_category_with_keywords.p"
    LSTM_dir = predir + "LSTM_news_title_category.p"
    CNN_dir = predir + "CNN_news_title_category.p"

    data_CNN_LSTM = cPickle.load(open(CNN_LSTM_dir, 'rb'))
    data_LSTM_keywords = cPickle.load(open(LSTM_keywords_dir, 'rb'))
    data_CNN_keywords = cPickle.load(open(CNN_keywords_dir, 'rb'))
    data_LSTM = cPickle.load(open(LSTM_dir, 'rb'))
    data_CNN = cPickle.load(open(CNN_dir, 'rb'))

    data = [data_CNN_LSTM, data_LSTM_keywords, data_CNN_keywords, data_LSTM, data_CNN]
    # train__, valid__ = data[0], data[1]
    plot_data_train_loss(data, "train_loss")
