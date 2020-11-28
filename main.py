import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# 读入数据
file_dir = 'assignment 2-supp.csv'
raw_data = np.loadtxt(file_dir, dtype=np.float, skiprows=1, delimiter=',')
feature = raw_data.shape[1] - 1

x = raw_data[:, 0:feature]
y = rawdata[:, feature]
x_data = torch.from_numpy(x).float()
y_data = torch.from_numpy(y).view(-1, 1).float()


# 逻辑回归
class Logistic_Regression(nn.Module):
    def __init__(self, n_feature):
        super(Logistic_Regression, self).__init__()
        self.lr = nn.Linear(n_feature, 1)
        self.sm = nn.Sigmoid()
        nn.init.zeros_(self.lr.weight)
        nn.init.zeros_(self.lr.bias)

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


# 训练网络
def train(x_data, y_data, lr, epochs=600):
    x_data_net = Variable(x_data)
    y_data_net = Variable(y_data)

    logistic_model = Logistic_Regression(feature)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=lr)
    print(logistic_model)
    # x_plot = []
    # y_plot = []

    for epoch in range(epochs):
        out = logistic_model(x_data_net)
        loss = criterion(out, y_data_net)

        # 计算accuracy
        mask = out.ge(0.5).float()
        correct = (mask == y_data_net).sum()
        acc = correct.item() / x_data_net.size(0)
        # x_plot.append(epoch)
        # y_plot.append(loss.data.item())
        # print(epoch, loss.data.item())
        # 梯度下降
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plt.plot(x_plot, y_plot)
    # plt.show()
    torch.save(logistic_model, "my_model.pth")

    return acc

acc = train(x_data, y_data, 0.000001)
print(acc)

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
