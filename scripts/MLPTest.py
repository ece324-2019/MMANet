'''
Wrote a basic training loop for a simple MLP
Only did train-valid split, no testing
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from time import time
from models import *
import argparse
from torch.utils.data import DataLoader


def main(args):
    torch.manual_seed(420)
    data = pd.read_csv("datafinal.csv")
    label = data["Winner"].copy()
    data = data.drop(columns=['Winner'])
    datanp = data.to_numpy()
    labelnp = label.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(datanp, labelnp, test_size=0.1)
    
    traindata = FightDataset(x_train, y_train)
    testdata = FightDataset(x_test, y_test)
    
    train_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(testdata, batch_size=len(x_test), shuffle=True)
    
    model = Net()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    taccuracystore = []
    vaccuracystore = []
    tlossstore = []
    vlossstore = []
    tacc = 0
    vacc = 0
    t = time()
    for i in range(args.epochs):
        tacc = 0

        for j, d in enumerate(train_loader, 0):

            inputs, label = d
            optimizer.zero_grad()
            predict = model(inputs.float())
            tloss = loss_function(input=predict.squeeze(), target=label.float())
            tloss.backward()
            optimizer.step()
            
            # Evaluating training accuracy
            for k in range(len(label)):
                if round(predict[k].item()) == label[k]:
                    tacc += 1

        vacc = 0
        # Evaluating validation accuracy
        for j, d in enumerate(val_loader, 0):
            inputs, label = d
            predict = model(inputs.float())
            vloss = loss_function(input=predict.squeeze(), target=label.float())
            for k in range(len(label)):
                if round(predict[k].item()) == label[k]:
                    vacc += 1
        taccuracystore.append(tacc/len(traindata))
        vaccuracystore.append(vacc/len(testdata))
        tlossstore.append(tloss)
        vlossstore.append(vloss)
        print(vacc/len(testdata))
    
    elapsed = time() - t
    print(elapsed)
    
    # Plotting accuracies for training and validation
    epochstore = range(len(taccuracystore))
    plt.plot(epochstore, taccuracystore, label='Train')
    plt.plot(epochstore, vaccuracystore, label='Validation')
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
