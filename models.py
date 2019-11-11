import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import torch.utils.data as data


class FightDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index, :], self.y[index]]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(157, 6)
        self.fc1_BN = nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.fc1_BN(self.fc1(x)))
        x = sigmoid(self.fc2(x))
        return x
