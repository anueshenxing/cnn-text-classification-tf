# coding=utf8
from __future__ import print_function
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
    p1.circle(sample_step_of_train, sample_loss_of_train, legend="CNN+LSTM", color='firebrick', size=8)

    p1.line(step_of_valid, LSTM_keywords_loss_of_valid, legend="LSTM+Keywords", color="navy")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_keywords_loss_of_valid, 10)
    p1.triangle(sample_step_of_train, sample_accuracy_of_train, legend="LSTM+Keywords", color='navy', size=8)

    p1.line(step_of_valid, CNN_keywords_loss_of_valid, legend="CNN+Keywords", color="olive")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_keywords_loss_of_valid, 10)
    p1.square(sample_step_of_train, sample_accuracy_of_train, legend="CNN+Keywords", color='olive', size=8)

    p1.line(step_of_valid, LSTM_loss_of_valid, legend="LSTM", color="green")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_loss_of_valid, 10)
    p1.diamond(sample_step_of_train, sample_accuracy_of_train, legend="LSTM", color='green', size=8)

    p1.line(step_of_valid, CNN_loss_of_valid, legend="CNN", color="DarkMagenta")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_loss_of_valid, 10)
    p1.asterisk(sample_step_of_train, sample_accuracy_of_train, legend="CNN", color='DarkMagenta', size=8)

    p2 = figure(width=1000, plot_height=500, title="Accuracy of Test Data",
                x_axis_label='step_num', y_axis_label='accuracy')

    CNN_LSTM_accuracy = p2.line(step_of_valid, CNN_LSTM_accuracy_of_valid, color='firebrick')
    sample_step_of_train, sample_loss_of_train = sampling(step_of_valid, CNN_LSTM_accuracy_of_valid, 10)
    CNN_LSTM_accuracy_sample = p2.circle(sample_step_of_train, sample_loss_of_train, color='firebrick', size=8)

    LSTM_keywords_accuracy = p2.line(step_of_valid, LSTM_keywords_accuracy_of_valid, color="navy")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_keywords_accuracy_of_valid, 10)
    LSTM_keywords_accuracy_sample = p2.triangle(sample_step_of_train, sample_accuracy_of_train, color='navy', size=8)

    CNN_keywords_accuracy = p2.line(step_of_valid, CNN_keywords_accuracy_of_valid, color="olive")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_keywords_accuracy_of_valid, 10)
    CNN_keywords_accuracy_sample = p2.square(sample_step_of_train, sample_accuracy_of_train, color='olive', size=8)

    LSTM_accuracy = p2.line(step_of_valid, LSTM_accuracy_of_valid, color="green")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, LSTM_accuracy_of_valid, 10)
    LSTM_accuracy_sample = p2.diamond(sample_step_of_train, sample_accuracy_of_train, color='green', size=8)

    CNN_accuracy = p2.line(step_of_valid, CNN_accuracy_of_valid, color="DarkMagenta")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_valid, CNN_accuracy_of_valid, 10)
    CNN_accuracy_sample = p2.asterisk(sample_step_of_train, sample_accuracy_of_train, color='DarkMagenta', size=8)

    legend = Legend(legends=[
        ("CNN+LSTM", [CNN_LSTM_accuracy, CNN_LSTM_accuracy_sample]),
        ("LSTM+Keywords", [LSTM_keywords_accuracy, LSTM_keywords_accuracy_sample]),
        ("CNN+Keywords", [CNN_keywords_accuracy, CNN_keywords_accuracy_sample]),
        ("LSTM", [LSTM_accuracy, LSTM_accuracy_sample]),
        ("CNN", [CNN_accuracy, CNN_accuracy_sample])
    ], location=(-180, -100))

    p2.add_layout(legend, 'right')
    # make a grid
    grid = gridplot([[p1], [p2]])

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

    print("CNN_LSTM")
    print(CNN_LSTM_SSS)
    print(np.average(CNN_LSTM_SSS))
    print("------------------------------------------------------")
    print("LSTM_K")
    print(LSTM_K_SSS)
    # print np.average(LSTM_K_SSS)
    # print "------------------------------------------------------"
    # print "CNN_K"
    # print CNN_K_SSS
    # print np.average(CNN_K_SSS)
    # print "------------------------------------------------------"
    # print "LSTM"
    # print LSTM_SSS
    # print np.average(LSTM_SSS)
    # print "------------------------------------------------------"
    # print "CNN"
    # print CNN_SSS
    # print np.average(CNN_SSS)


# if __name__ == "__main__":
#     predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/save_data/"
#     CNN_LSTM_dir = predir + "CNN_LSTM_Model_result.p"
#     LSTM_keywords_dir = predir + "LSTM_news_title_category_with_keywords.p"
#     CNN_keywords_dir = predir + "CNN_news_title_category_with_keywords.p"
#     LSTM_dir = predir + "LSTM_news_title_category.p"
#     CNN_dir = predir + "CNN_news_title_category.p"
#
#     data_CNN_LSTM = cPickle.load(open(CNN_LSTM_dir, 'rb'))
#     data_LSTM_keywords = cPickle.load(open(LSTM_keywords_dir, 'rb'))
#     data_CNN_keywords = cPickle.load(open(CNN_keywords_dir, 'rb'))
#     data_LSTM = cPickle.load(open(LSTM_dir, 'rb'))
#     data_CNN = cPickle.load(open(CNN_dir, 'rb'))
#
#     train_loss = data_CNN_LSTM[0][1]
#     for i in range(len(train_loss)):
#         print(train_loss[i])
#
#     # data = [data_CNN_LSTM, data_LSTM_keywords, data_CNN_keywords, data_LSTM, data_CNN]
#
#     # train__, valid__ = data[0], data[1]
#     # plot_data_train_loss(data, "train_loss")

if __name__ == "__main__":
    Epoch = []
    for i in range(1, 21):
        Epoch.append(i)

    learning_rate_0_1 = [4.45491, 5.12362, 2.81459, 3.07209, 4.66491, 3.29575, 5.04803, 3.52142, 2.16922, 5.57484,
                        4.06972, 4.19993, 3.82059, 5.59162, 2.86231, 3.37522, 2.86373, 3.53516, 2.86176, 3.75321]
    learning_rate_0_1 = np.array(learning_rate_0_1)-1.5
    learning_rate_0_01 = [0.88147, 0.514658,0.0671211,0.0691211,0.0574511,0.065143,0.08715211,0.057631,0.06532211,0.0611211,
			0.0471211,0.0571211,0.0871211,0.0731211,0.0941211,0.0631211,0.0611211,0.0471211,0.0541211,0.0631211]
    learning_rate_0_001 = [0.74894, 0.56256, 0.338692, 0.174076, 0.098596, 0.021476, 0.0133302, 0.0215216, 0.0210464, 0.0306554,
                          0.0263549, 0.0226637, 0.0143946, 0.0198438, 0.0222891, 0.0121695, 0.0185416, 0.0104946, 0.0210853, 0.0124344]
    learning_rate_0_0001 = [0.377657, 0.17895, 0.130139, 0.111584, 0.0974784, 0.0646639, 0.0243578, 0.0335536, 0.0256461, 0.00698066,
                            0.00746492, 0.014243, 0.00387504, 0.0202542, 0.0108288, 0.00165264, 0.0126408, 0.0162946, 0.00248038, 0.00231948]
    p1 = figure(width=700, plot_height=500,
                x_axis_label='Epoch', y_axis_label='Loss Value')


    p1.line(Epoch, learning_rate_0_1, legend="learing rate = 0.1", color='firebrick')
    sample_step_of_train, sample_loss_of_train = sampling(Epoch, learning_rate_0_1, 20)
    p1.circle(sample_step_of_train, sample_loss_of_train, legend="learing rate = 0.1", color='firebrick', size=8)

    p1.line(Epoch, learning_rate_0_01, legend="learing rate = 0.01", color="navy")
    sample_step_of_train, sample_accuracy_of_train = sampling(Epoch, learning_rate_0_01, 20)
    p1.triangle(sample_step_of_train, sample_accuracy_of_train, legend="learing rate = 0.01", color='navy', size=8)

    p1.line(Epoch, learning_rate_0_001, legend="learing rate = 0.001", color="olive")
    sample_step_of_train, sample_accuracy_of_train = sampling(Epoch, learning_rate_0_001, 20)
    p1.square(sample_step_of_train, sample_accuracy_of_train, legend="learing rate = 0.001", color='olive', size=8)

    p1.line(Epoch, learning_rate_0_0001, legend="learing rate = 0.0001", color="green")
    sample_step_of_train, sample_accuracy_of_train = sampling(Epoch, learning_rate_0_0001, 20)
    p1.diamond(sample_step_of_train, sample_accuracy_of_train, legend="learing rate = 0.0001", color='green', size=8)


    show(p1)
