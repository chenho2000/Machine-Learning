import numpy as np
import numpy.random as rnd
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import bonnerlib3 as bl3d

with open('mnistTVT.pickle', 'rb') as f:
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)

if __name__ == "__main__":
    # print(Xtrain.shape, Ttrain.shape, Xval.shape, Tval.shape, Xtest.shape,
    #       Ttest.shape)
    T_test_check = np.logical_or((Ttest == 5), (Ttest == 6))
    T_test = Ttest[T_test_check]
    X_test = Xtest[T_test_check]
    T_train_check = np.logical_or((Ttrain == 5), (Ttrain == 6))
    T_train = Ttrain[T_train_check]
    np.reshape(Xtrain, Xtrain.shape[0] * Xtrain.shape[1])
    # np.split(Xtrain, Xtrain.shape[0] // 784)
    print(type(Xtrain))
    print(len(T_train_check))
    # X_train = X_train[T_train_check]
    # T_val_check = np.logical_or((Tval == 5), (Tval == 6))
    # T_val = Tval[T_val_check]
    # X_val = Xval[T_val_check]
    # sT_train = T_train[:2000]
    # sX_train = X_train[:2000]
    # fig_b = plt.figure(figsize=(4, 4))
    # print(T_train.shape)
    # for i in range(16):
    #     current = T_train[i:i + 784].reshape(28, 28)
    #     fig_b.add_subplot(4, i + 1 % 4, i + 1 % 4)
    #     plt.imshow(current, cmap='gray')
    # plt.show()
