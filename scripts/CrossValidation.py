'''
Wrote a basic training loop for a simple MLP
'''

import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from models import DeepNet, SimpleNet, FightDataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score


def main(args):
    torch.manual_seed(args.seed)
    data = pd.read_csv("../data/datafinal.csv")
    label = data["Winner"].copy()
    data = data.drop(columns=['Winner'])
    data_np = data.to_numpy(dtype=np.float32)
    label_np = label.to_numpy(dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(data_np, label_np,
                                                        test_size=0.1)
    model = DeepNet()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=0.001)
    print(data_np.shape, label_np.shape)
    logistic = NeuralNetClassifier(model, max_epochs=20, lr=1e-3)
    scores = cross_val_score(logistic, data_np, label_np, cv=3,
                             scoring="accuracy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    main(args)
