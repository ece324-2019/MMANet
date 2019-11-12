'''
Wrote a basic training loop for a simple MLP
Only did train-valid split, no testing
'''

import pandas as pd
import torch
from time import time
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models import Net, FightDataset


def main(args):
    torch.manual_seed(420)
    data = pd.read_csv("../data/datafinal.csv")
    label = data["Winner"].copy()
    data = data.drop(columns=['Winner'])
    data_np = data.to_numpy()
    label_np = label.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data_np, label_np,
                                                        test_size=0.1)

    train_data = FightDataset(x_train, y_train)
    test_data = FightDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=True)

    model = Net()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_accuracystore = []
    v_accuracystore = []
    t_lossstore = []
    v_lossstore = []
    t_acc = 0
    v_acc = 0
    t = time()
    for i in range(args.epochs):
        t_acc = 0

        for j, d in enumerate(train_loader, 0):

            inputs, label = d
            optimizer.zero_grad()
            predict = model(inputs.float())
            t_loss = loss_function(input=predict.squeeze(),
                                   target=label.float())
            t_loss.backward()
            optimizer.step()

            # Evaluating training accuracy
            for k in range(len(label)):
                if round(predict[k].item()) == label[k]:
                    t_acc += 1

        v_acc = 0
        # Evaluating validation accuracy
        for j, d in enumerate(val_loader, 0):
            inputs, label = d
            predict = model(inputs.float())
            v_loss = loss_function(input=predict.squeeze(),
                                   target=label.float())
            for k in range(len(label)):
                if round(predict[k].item()) == label[k]:
                    v_acc += 1
        t_accuracystore.append(t_acc / len(train_data))
        v_accuracystore.append(v_acc / len(test_data))
        t_lossstore.append(t_loss)
        v_lossstore.append(v_loss)
        print(v_acc / len(test_data))

    elapsed = time() - t
    print(elapsed)

    # Plotting accuracies for training and validation
    epoch_store = range(len(t_accuracystore))
    plt.plot(epoch_store, t_accuracystore, label='Train')
    plt.plot(epoch_store, v_accuracystore, label='Validation')
    plt.title("Accuracy over Batches")
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Batch #')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()

    main(args)
