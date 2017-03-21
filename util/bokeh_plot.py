# coding=utf8
import sys
import cPickle

from bokeh.layouts import row, gridplot
from bokeh.plotting import figure, output_file, show

reload(sys)
sys.setdefaultencoding("utf-8")

def sampling(x,y,sample_num):
    sample_x = []
    sample_y = []
    gap = len(x)/sample_num
    for i in range(sample_num):
        sample_x.append(x[i*gap])
        sample_y.append(y[i*gap])
    return sample_x,sample_y

def plot_data_train_loss(data_train, data_valid, file_name):
    # prepare some data
    step_of_train = data_train[0]
    loss_of_train = data_train[1]
    accuracy_of_train = data_train[2]

    step_of_valid = data_valid[0]
    loss_of_valid = data_valid[1]
    accuracy_of_valid = data_valid[2]

    # output to static HTML file
    file_dir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/data_figure/" + file_name + ".html"
    output_file(file_dir)

    # 训练损失变化
    p1 = figure(width=500, plot_height=300, title="train loss",
                x_axis_label='step_num', y_axis_label='loss')
    p1.line(step_of_train, loss_of_train, legend="CNN+LSTM", color="navy", line_width=2)

    # 训练准确度变化
    p2 = figure(width=500, plot_height=300, title="train accuracy",
                x_axis_label='step_num', y_axis_label='accuracy')
    p2.line(step_of_train, accuracy_of_train, legend="CNN+LSTM", color="firebrick", line_width=2)

    # 测试损失变化
    p3 = figure(width=500, plot_height=300, title="test loss",
                x_axis_label='valid_num', y_axis_label='loss')
    p3.line(step_of_valid, loss_of_valid, legend="CNN+LSTM", color="firebrick", line_width=2)

    # 测试准确度变化
    p4 = figure(width=500, plot_height=300, title="test accuracy",
                x_axis_label='valid_num', y_axis_label='accuracy')
    p4.line(step_of_valid, accuracy_of_valid, legend="CNN+LSTM", color="firebrick", line_width=2)

    p5 = figure(width=1000, plot_height=500, title="compare",
                x_axis_label='step_num', y_axis_label='')

    p5.line(step_of_train, loss_of_train, legend="loss", color='firebrick')
    sample_step_of_train, sample_loss_of_train = sampling(step_of_train, loss_of_train, 20)
    p5.circle(sample_step_of_train, sample_loss_of_train, legend="loss", color='firebrick')

    p5.line(step_of_train, accuracy_of_train, legend="accuracy", color="navy")
    sample_step_of_train, sample_accuracy_of_train = sampling(step_of_train, accuracy_of_train, 20)
    p5.triangle(sample_step_of_train, sample_accuracy_of_train, legend="accuracy", color='navy')
    # make a grid
    grid = gridplot([[p1, p2], [p3, p4], [p5, None]])

    # show the results
    show(grid)


if __name__ == "__main__":
    predir = "/home/zhang/PycharmProjects/cnn-text-classification-tf/save_data/"
    data_dir = predir + "CNN_LSTM_Model_result.p"
    data = cPickle.load(open(data_dir, 'rb'))
    train__, valid__ = data[0], data[1]
    plot_data_train_loss(train__, valid__, "train_loss")
