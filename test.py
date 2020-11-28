import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd


# one hots: (73, 398)
def load_data(filefolder):
    data = np.load(filefolder + '/names_onehots.npy', allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name


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
    test_data, test_name = load_data("test")
    test_data = torch.from_numpy(test_data).float()
    test_data = torch.unsqueeze(test_data, dim=1)  # ([272, 1, 73, 398])

    model = CNN()
    model.load_state_dict(torch.load('cnn_epo.pth'))
    print(model)

    model.eval()
    output, output_with_sm = model(test_data)
    prob_y = output_with_sm[:, 1].data.numpy()

    f = open('output_518030910055.txt', 'w')
    f.write('Chemical,Label\n')
    for i, v in enumerate(prob_y):
        f.write(test_name[i] + ',%f\n' % v)
    f.close()

