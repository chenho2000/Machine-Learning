import math
import numpy as np
import numpy.random as rnd
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
from sklearn import datasets
from collections import Counter

infinity = float(-2 ** 31)

with open('dataA1Q4v2.pickle', 'rb') as f:
    x_train, t_train, x_test, t_test = pickle.load(f)


def gd_logreg(lrate):
    np.random.seed(3)
    theta = rnd.randn(x_train.shape[1] + 1) / 1000
    a = np.full((x_train.shape[0], 1), 1)
    x = np.column_stack((a, x_train))
    count = 1
    train_entropy = []
    test_entropy = []
    train_accuracy = []
    test_accuracy = []
    while True:
        z = np.matmul(x, theta)
        y = 1.0 / (1.0 + np.exp(-z))
        theta = theta - lrate * (x.T @ (y - t_train)) / t_train.shape[0]
        test_entropy.append(
            t_test @ np.logaddexp(0, -z) + (1 - t_test) @ np.logaddexp(0, z))
        train_entropy.append(
            t_train @ np.logaddexp(0, -z) + (1 - t_train) @ np.logaddexp(0, z))
        if count != 1:
            if abs(train_entropy[-1] - train_entropy[-2]) < 10**-10:
                break
        count += 1
    return theta


if __name__ == "__main__":

    print(gd_logreg(0.1))
    print(gd_logreg(0.3))
    print(gd_logreg(1))
    print(gd_logreg(3))
    print(gd_logreg(10))
