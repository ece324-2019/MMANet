import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import torch
import random

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)

#File names:
train_file = './train.csv'
valid_file = './valid.csv'
test_file = './test.csv'

train_data = pd.read_csv(train_file,index_col = None)
valid_data = pd.read_csv(valid_file, index_col = None)
test_data = pd.read_csv(test_file, index_col = None)
#print(data.shape)
#print(data.columns)


if __name__ == '__main__':
    train_x = train_data.to_numpy()[:,1:]
    train_y = train_data.to_numpy()[:,0]

    valid_x = valid_data.to_numpy()[:,1:]
    valid_y = valid_data.to_numpy()[:,0]

    test_x = test_data.to_numpy()[:,1:]
    test_y = test_data.to_numpy()[:,0]

    svm = SVC(gamma = 'auto')
    logit = LogisticRegression(solver = 'lbfgs')
    forest = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_split = 2) 
    ada = AdaBoostClassifier(n_estimators = 50, learning_rate = 1)

    svm.fit(train_x,train_y)
    logit.fit(train_x,train_y)
    forest.fit(train_x,train_y)
    ada.fit(train_x, train_y)
    print('svm done')
    y_pred = svm.predict(valid_x)
    y_l_pred = logit.predict(valid_x)
    forest_pred = forest.predict(valid_x)
    ada_pred = ada.predict(valid_x)

    v_accuracy = accuracy_score(valid_y,y_pred)
    logit_accuracy = accuracy_score(valid_y,y_l_pred)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    ada_accuracy = accuracy_score(valid_y, ada_pred)

    print('Validation Accuracy SVM:',v_accuracy)
    print('Validation Accuracy Logit',logit_accuracy)
    print('Validation Accuracy Forest:', forest_accuracy)
    print('Boosted Validation Accuract:', ada_accuracy)

    y_pred = svm.predict(test_x)
    y_l_pred = logit.predict(test_x)
    forest_pred = forest.predict(test_x)
    ada_pred = ada.predict(test_x)

    v_accuracy = accuracy_score(test_y,y_pred)
    logit_accuracy = accuracy_score(test_y,y_l_pred)
    forest_accuracy = accuracy_score(test_y, forest_pred)
    ada_accuracy = accuracy_score(test_y, ada_pred)

    print('Test Accuracy SVM:',v_accuracy)
    print('Test Accuracy Logit',logit_accuracy)
    print('Test Accuracy Forest:', forest_accuracy)
    print('Boosted Test Accuracy:', ada_accuracy)

    
