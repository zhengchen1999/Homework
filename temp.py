import matplotlib.pyplot as plt
import numpy as np

# 读入数据
file_dir = 'assignment 2-supp.csv'
raw_data = np.loadtxt(file_dir, dtype=np.float, skiprows=1, delimiter=',')
feature = raw_data.shape[1] - 1
size_data = raw_data.shape[0]

x_data = raw_data[:, 0:feature]
y_data = raw_data[:, feature].reshape((size_data, 1))

# 初始化
def initial(nums_feature):
    w = np.zeros((1, nums_feature))
    b = 0
    return w, b


# 激活函数
def sig(x, w, b):
    tmp = np.dot(w, x.T) + b
    sigmoid = 1 / (1 + np.exp(-tmp))
    return sigmoid


# 结果
def result(out):
    m = out.shape[0]
    ans = 0
    for i in range(m):
        if out[i, 0] > 0.5:
            tmp = 1
        else:
            tmp = 0
        ans += int(tmp == y_data[i, 0])

    return ans


# 训练网络
def train(x, y, lr, epochs=600):

    w, b = initial(feature)

    x_plot = []
    y_plot = []

    for epoch in range(epochs):
        out = sig(x, w, b)
        loss = -np.sum(y.T * np.log(out) + (1 - y).T * np.log(1 - out)) / size_data

        x_plot.append(epoch)
        y_plot.append(loss)
        print(epoch, loss)

        # 计算acc
        correct = result(out.T)
        acc_ = correct / size_data
        # print(correct, acc_)
        # 梯度下降
        g_w = np.dot(x.T, (out - y.T).T) / size_data
        g_b = np.sum((out - y.T)) / size_data
        w = w - lr * g_w.T
        b = b - lr * g_b

    plt.plot(x_plot, y_plot)
    plt.show()

    return acc_


acc = train(x_data, y_data, 0.000001)

# lr-acc
# e = -6
# x_plot = []
# y_plot = []
#
# while e <= -2:
#     lr = 10 ** e
#     acc = train(x_data, y_data, lr)
#     x_plot.append(e)
#     y_plot.append(acc)
#     print(e, acc)
#     e += 0.05
#
# plt.plot(x_plot, y_plot)
# plt.show()


