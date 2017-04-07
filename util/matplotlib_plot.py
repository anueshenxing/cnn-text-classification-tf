# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def show_figure(accuracy, num_keywords):
    x_b = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    y_50 = accuracy[0]
    y_100 = accuracy[1]
    y_150 = accuracy[2]
    y_200 = accuracy[3]
    l1, l2, l3, l4 = plt.plot(x_b, y_50, 'o-', x_b, y_100, 's-', x_b, y_150, '*-', x_b, y_200, '^-')
    plt.legend((l1, l2, l3, l4), ('50', '100', '150', '200'), loc='upper right', shadow=True)
    plt.title('number of keywords:' + str(num_keywords))
    plt.xlabel('threshold value')
    plt.ylabel('accuracy(%)')
    plt.xlim(0.35, 0.7)
    plt.ylim(33, 43)
    plt.show()


if __name__ == "__main__":
    accuracy_6 = np.array([[32.5, 35.1, 36.7, 36.2, 35.7, 32.9],
                           [33.3, 34.7, 37.5, 36.9, 36.3, 33.7],
                           [32.9, 35.2, 37.3, 37.0, 36.5, 33.4],
                           [32.6, 35.0, 37.2, 36.4, 36.0, 34.1]])

    accuracy_9 = np.array([[39.7, 41.9, 43.0, 43.2, 42.1, 40.3],
                           [39.2, 42.4, 43.1, 43.0, 41.9, 39.8],
                           [39.5, 42.5, 43.2, 43.0, 41.5, 39.3],
                           [38.9, 42.0, 43.0, 42.5, 41.4, 39.1]])-3

    accuracy_12 = np.array([[43.69, 46.77, 48.45, 48.10, 46.3, 45.2],
                            [43.73, 46.9, 48.6, 48.45, 47.31, 45.54],
                            [43.35, 46.53, 48.43, 48.10, 46.92, 45.13],
                            [43.10, 46.21, 48.39, 48.01, 46.5, 44.98]])-4

    accuracy_15 = np.array([[51.0, 53.5, 54.7, 53.9, 52.0, 51.1],
                            [51.4, 53.9, 55.0, 53.6, 52.3, 51.5],
                            [51.12, 53.0, 54.9, 53.7, 52.1, 51.4],
                            [50.91, 52.9, 54.8, 53.3, 51.8, 51.0]])-4.8

    show_figure(accuracy_6, 6)
    # show_figure(accuracy_9, 9)
    # show_figure(accuracy_12, 12)
    # show_figure(accuracy_15, 15)
