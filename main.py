import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd

# 临时使用
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 128
LR = 0.01


# one hots: (73, 398)
def load_data(filefolder):
    data = np.load(filefolder + '/names_onehots.npy', allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(filefolder + '/names_labels.txt', sep=',')
    label = label['Label'].values
    return data, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input size: [1, 73, 398]
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(  # input size: [16, 36, 199]
            nn.Conv2d(32, 32, 5, (1, 2), 2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.out = nn.Linear(32 * 18 * 50, 2)  # input size: [32, 18, 50]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        output_with_sm = F.softmax(output, dim=1)
        return output, output_with_sm


if __name__ == '__main__':
    # 读入数据
    train_x, train_y = load_data('train')
    validation_x, validation_y = load_data('validation')  # 241个0
    # 处理数据
    train_x = torch.from_numpy(train_x).float()
    train_x = torch.unsqueeze(train_x, dim=1)  # ([8169, 1, 73, 398])
    train_y = torch.from_numpy(train_y)  # ([8169])
    torch_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 63

    validation_x = torch.from_numpy(validation_x).float()
    validation_x = torch.unsqueeze(validation_x, dim=1)  # ([272, 1, 73, 398])
    validation_y = torch.from_numpy(validation_y)  # ([272])

    # 训练网络
    cnn = CNN()
    # test = CNN()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    best_auc = 0
    best_acc = 0
    epoch_auc = 0

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            cnn.train()
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(step)

            if step % 20 == 0:
                cnn.eval()
                validation_out, validation_out_with_sm = cnn(validation_x)
                validation_loss = loss_func(validation_out, validation_y)
                pred_y = torch.max(validation_out_with_sm, 1)[1].data.numpy()
                accuracy = float((pred_y == validation_y.data.numpy()).sum()) / float(validation_y.size(0))
                prob_y = validation_out_with_sm[:, 1].data.numpy()
                auc = roc_auc_score(validation_y.data.numpy(), prob_y)

                print('Epoch: ', epoch, '| AUC: %.2f' % auc,
                      '| validation loss: %.4f' % validation_loss.item(),
                      '| validation accuracy: %.2f' % accuracy)
                if auc > best_auc:
                    best_auc = auc
                    torch.save(cnn, 'new_net/cnn_auc.pth')
                    f = open('new_net/num_auc.txt', 'w')
                    f.write('Epoch: ' + str(epoch) + '| AUC: %.2f' % auc +
                            '| validation loss: %.4f' % validation_loss.item() +
                            '| validation accuracy: %.2f' % accuracy)
                    f.close()
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(cnn, 'new_net/cnn_acc.pth')
                    f = open('new_net/num_acc.txt', 'w')
                    f.write('Epoch: ' + str(epoch) + '| AUC: %.2f' % auc +
                            '| validation loss: %.4f' % validation_loss.item() +
                            '| validation accuracy: %.2f' % accuracy)
                    f.close()
                cnn.train()

        # 测验
        cnn.eval()
        validation_out, validation_out_with_sm = cnn(validation_x)
        validation_loss = loss_func(validation_out, validation_y)
        pred_y = torch.max(validation_out_with_sm, 1)[1].data.numpy()
        accuracy = float((pred_y == validation_y.data.numpy()).sum()) / float(validation_y.size(0))
        prob_y = validation_out_with_sm[:, 1].data.numpy()
        auc = roc_auc_score(validation_y.data.numpy(), prob_y)
        print(pred_y)
        print('Epoch: ', epoch, '| AUC: %.2f' % auc,
              '| validation loss: %.4f' % validation_loss.item(),
              '| validation accuracy: %.2f' % accuracy)
        if auc > epoch_auc:
            epoch_auc = auc
            torch.save(cnn, 'new_net/cnn_epo.pth')
            f = open('new_net/num_epo.txt', 'w')
            f.write('Epoch: ' + str(epoch) + '| AUC: %.2f' % auc +
                    '| validation loss: %.4f' % validation_loss.item() +
                    '| validation accuracy: %.2f' % accuracy)
            f.close()
        cnn.train()
