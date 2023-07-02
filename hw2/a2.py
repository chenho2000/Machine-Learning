import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import bonnerlib2D as bl2d
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy.random as rnd
from sklearn.utils import shuffle
import time

# import matplotlib
# matplotlib.use('Qt5Agg')
print("\n")
print("Question 1.")
print("-------------")

# Question 1
with open('dataA2Q1.pickle', 'rb') as f:
    q1_dataTrain, q1_dataTest = pickle.load(f)


def q1_error(w, x, t, K):
    length = x.shape[0]
    trig = np.ones((length, 1))
    k = np.arange(1, K + 1)
    if K > 0:
        curr = np.array([x, ] * K).transpose() * k
        trig = np.column_stack((trig, np.sin(curr)))
        trig = np.column_stack((trig, np.cos(curr)))
    y = trig @ w
    mse = np.square(np.subtract(y, t)).mean()
    return mse


def fit_plot(dataTrain, dataTest, K):
    length = dataTrain.shape[1]
    trig = np.ones((length, 1))
    k = np.arange(1, K + 1)
    if K > 0:
        curr = np.array([dataTrain[0], ] * K).transpose() * k
        trig = np.column_stack((trig, np.sin(curr)))
        trig = np.column_stack((trig, np.cos(curr)))
    w = np.linalg.lstsq(trig, dataTrain[1], rcond=None)[0]
    training_error = q1_error(w, dataTrain[0], dataTrain[1], K)
    test_error = q1_error(w, dataTest[0], dataTest[1], K)
    return w, training_error, test_error


def q1_show(w, dataTrain, dataTest, K, k, training_error, test_error, q):
    plt.scatter(dataTrain[0], dataTrain[1], s=20)
    x = np.linspace(np.min(dataTrain[0]), np.max(dataTrain[0]), 1000)
    trig = np.ones((1000, 1))
    curr = np.array([x, ] * K).transpose() * k
    trig = np.column_stack((trig, np.sin(curr)))
    trig = np.column_stack((trig, np.cos(curr)))
    y = trig @ w
    plt.plot(x, y, '-r')
    plt.ylim(np.min(dataTrain[1]) - 5, np.max(dataTrain[1]) + 5)
    plt.title("Question 1({}): the fitted function (K={})".format(q, K))
    if q == "f":
        plt.title("Question 1(f): the best fitting function")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    print("K value:", K, "\nweight vector:", w, "\nTraining error:",
          training_error, "\nTest error:", test_error)


# print("\n")
# print("Question 1(a).")
# print("-------------")
# q1a = fit_plot(q1_dataTrain,q1_dataTest,4)
# q1_show(q1a[0], q1_dataTrain,q1_dataTest,4 ,np.arange(1,4+1),q1a[1],q1a[2],"a" )

print("\n")
print("Question 1(b).")
print("-------------")
q1b = fit_plot(q1_dataTrain, q1_dataTest, 3)
q1_show(q1b[0], q1_dataTrain, q1_dataTest, 3, np.arange(1, 3 + 1), q1b[1],
        q1b[2], "b")

print("\n")
print("Question 1(c).")
print("-------------")
q1c = fit_plot(q1_dataTrain, q1_dataTest, 9)
q1_show(q1c[0], q1_dataTrain, q1_dataTest, 9, np.arange(1, 9 + 1), q1c[1],
        q1c[2], "c")

print("\n")
print("Question 1(d).")
print("-------------")
q1d = fit_plot(q1_dataTrain, q1_dataTest, 12)
q1_show(q1d[0], q1_dataTrain, q1_dataTest, 12, np.arange(1, 12 + 1), q1d[1],
        q1d[2], "d")

# print("\n")
# print("Question 1(e).")
# print("-------------")
plt.figure(figsize=(18, 16))
for i in range(1, 13):
    q1e = fit_plot(q1_dataTrain, q1_dataTest, i)
    k = np.arange(1, i + 1)
    w = q1e[0]
    plt.subplot(4, 3, i)
    plt.scatter(q1_dataTrain[0], q1_dataTrain[1], s=20)
    x = np.linspace(np.min(q1_dataTrain[0]), np.max(q1_dataTrain[0]), 1000)
    trig = np.ones((1000, 1))
    curr = np.array([x, ] * i).transpose() * k
    trig = np.column_stack((trig, np.sin(curr)))
    trig = np.column_stack((trig, np.cos(curr)))
    y = trig @ w
    plt.plot(x, y, '-r')
    plt.ylim(np.min(q1_dataTrain[1]) - 5, np.max(q1_dataTrain[1]) + 5)
plt.suptitle("Question 1(e):fitted functions for many values of K.")
plt.show()

print("\n")
print("Question 1(f).")
print("-------------")
times = len(np.split(q1_dataTrain.T, 5))
K = np.arange(0, 13)
training_errors_means = []
validation_errors_means = []
for k in range(0, 13):
    training_errors = []
    validation_errors = []
    for i in range(times):
        w, training_error, test_error = fit_plot(
            np.delete(q1_dataTrain, [np.arange(i * 5, i * 5 + 5)], 1),
            q1_dataTest, k)
        validation_error = q1_error(w, q1_dataTrain[0][i * 5:i * 5 + 5],
                                    q1_dataTrain[1][i * 5:i * 5 + 5], k)
        training_errors.append(training_error)
        validation_errors.append(validation_error)
    training_errors_means.append(np.mean(training_errors))
    validation_errors_means.append(np.mean(validation_errors))
plt.figure()
plt.semilogy(K, training_errors_means, c="b")
plt.semilogy(K, validation_errors_means, c="r")
plt.title("Question 1(f ): mean training and validation error")
plt.ylabel('Mean Error')
plt.xlabel('K')
plt.show()
smallest = np.argmin(validation_errors_means)
q1f = fit_plot(q1_dataTrain, q1_dataTest, smallest)
q1_show(q1f[0], q1_dataTrain, q1_dataTest, smallest, np.arange(1, smallest + 1),
        q1f[1], q1f[2], "f")

print("\n")
print("Question 2.")
print("-------------")

# Question 2
with open('dataA2Q2.pickle', 'rb') as file:
    q2_dataTrain, q2_dataTest = pickle.load(file)
Xtrain, Ttrain = q2_dataTrain
Xtest, Ttest = q2_dataTest


# 2(a)
def plot_data(X, T):
    color = np.where(T == 0, 'r', T)
    color = np.where(color == '1', 'b', color)
    color = np.where(color == '2', 'g', color)
    plt.xlim(np.min(X.T[0]) - 0.1, np.max(X.T[0]) + 0.1)
    plt.ylim(np.min(X.T[1]) - 0.1, np.max(X.T[1]) + 0.1)
    plt.scatter(X.T[0], X.T[1], c=color, s=2)


# print("\n")
# print("Question 2(a).")
# print("-------------")
# plot_data(Xtrain,Ttrain)
# plt.show()

# This function will be used in the rest of the assignment
def softmax(v):
    return (np.exp(v)) / (
        np.reshape(np.sum(np.exp(v), axis=1), (np.exp(v).shape[0], 1)))


# 2(b)
def accuracyLR(clf, X, T):
    w, w0 = clf.coef_, clf.intercept_
    v = X @ w.T + w0
    p = softmax(v)
    predict = np.argmax(p, axis=1)
    t = (predict == T)
    return np.mean(t)


print("\n")
print("Question 2(b).")
print("-------------")
clf = lin.LogisticRegression(multi_class='multinomial',
                             solver='lbfgs')  # create a classification object, clf
clf.fit(Xtrain, Ttrain)  # learn a logistic-regression classifier
accuracy1 = clf.score(Xtest, Ttest)
accuracy2 = accuracyLR(clf, Xtest, Ttest)
print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy2 - accuracy1:", accuracy2 - accuracy1)
plot_data(Xtrain, Ttrain)
bl2d.boundaries(clf)
plt.title("Question 2(b): decision boundaries for logistic regression")
plt.show()


# 2(c)
def accuracyQDA(clf, X, T):
    mean, cov, p = clf.means_, clf.covariance_, clf.priors_
    pxc = []
    for i in range(T.max() + 1):
        pxc.append(multivariate_normal.pdf(X, mean[i], cov[i]))
    pxc = (np.array(pxc)).T
    probability = pxc * p
    d = np.reshape((np.sum(probability, axis=1)), (X.shape[0], 1))
    predict = np.argmax(probability / d, axis=1)
    t = (predict == T)
    return np.mean(t)


print("\n")
print("Question 2(c).")
print("-------------")
clf = QuadraticDiscriminantAnalysis(store_covariance=True)
clf.fit(Xtrain, Ttrain)
accuracy1 = clf.score(Xtest, Ttest)
accuracy2 = accuracyQDA(clf, Xtest, Ttest)
print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy2 - accuracy1:", accuracy2 - accuracy1)
plot_data(Xtrain, Ttrain)
bl2d.boundaries(clf)
plt.title(
    "Question 2(c): decision boundaries for quadratic discriminant analysis")
plt.show()


# 2(d)
def accuracyNB(clf, X, T):
    mean = clf.theta_
    var = clf.sigma_
    exponent = np.exp(-np.square(
        X.reshape((X.shape[0], 1, mean.shape[1])) - mean.reshape(
            (1, mean.shape[0], mean.shape[1]))) / (
                              2 * var))
    probability = (exponent / np.sqrt(2 * np.pi * var))
    mult_prob = np.prod(probability, axis=2)
    p = clf.class_prior_
    predict = np.argmax(mult_prob * p, axis=1)
    t = (predict == T)
    return np.mean(t)


print("\n")
print("Question 2(d).")
print("-------------")
clf = GaussianNB()
clf.fit(Xtrain, Ttrain)
accuracy1 = clf.score(Xtest, Ttest)
accuracy2 = accuracyNB(clf, Xtest, Ttest)
print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy2 - accuracy1:", accuracy2 - accuracy1)
plot_data(Xtrain, Ttrain)
bl2d.boundaries(clf)
plt.title("Question 2(d): decision boundaries for Gaussian naive Bayes.")
plt.show()

print("\n")
print("Question 3.")
print("-------------")
# Question 3
with open('dataA2Q2.pickle', 'rb') as file:
    q3_dataTrain, q3_dataTest = pickle.load(file)
Xtrain, Ttrain = q3_dataTrain
Xtest, Ttest = q3_dataTest


# Question 3(b)
def q3b(size, epoch, seed):
    clf = MLPClassifier(hidden_layer_sizes=size, activation='logistic',
                        solver='sgd', learning_rate_init=0.01, tol=10 ** -6,
                        max_iter=epoch)
    np.random.seed(seed)
    clf.fit(Xtrain, Ttrain)
    accuracy1 = clf.score(Xtest, Ttest)
    print("test accuracy:", accuracy1)
    plot_data(Xtrain, Ttrain)
    bl2d.boundaries(clf)


print("\n")
print("Question 3(b).")
print("-------------")
q3b(1, 1000, 0)
plt.title("Question 3(b): Neural net with 1 hidden unit")
plt.show()

print("\n")
print("Question 3(c).")
print("-------------")
q3b(2, 1000, 0)
plt.title("Question 3(c): Neural net with 2 hidden unit")
plt.show()

print("\n")
print("Question 3(d).")
print("-------------")
q3b(9, 1000, 0)
plt.title("Question 3(d): Neural net with 9 hidden unit")
plt.show()

print("\n")
print("Question 3(e).")
print("-------------")
plt.figure(figsize=(18, 16))
for i in range(2, 11):
    plt.subplot(3, 3, i - 1)
    q3b(7, 2 ** i, 0)
plt.suptitle("Question 3(e): different numbers of epochs.")
plt.show()

print("\n")
print("Question 3(f).")
print("-------------")
plt.figure(figsize=(18, 16))
for i in range(1, 10):
    plt.subplot(3, 3, i)
    q3b(5, 1000, i)
plt.suptitle("Question 3(f): different initial weights")
plt.show()


# Question 3(g)
def accuracyNN(clf, X, T):
    w1, b1 = clf.coefs_[0], clf.intercepts_[0]
    v1 = 1 / (1 + np.exp(-(X @ w1 + b1)))
    w2, b2 = clf.coefs_[1], clf.intercepts_[1]
    v = v1 @ w2 + b2
    p = softmax(v)
    predict = np.argmax(p, axis=1)
    t = (predict == T)
    return np.mean(t)


print("\n")
print("Question 3(g).")
print("-------------")
np.random.seed(0)
clf = MLPClassifier(hidden_layer_sizes=9, activation='logistic', solver='sgd',
                    learning_rate_init=0.01, tol=10 ** -6, max_iter=1000)
clf.fit(Xtrain, Ttrain)
accuracy1 = clf.score(Xtest, Ttest)
accuracy2 = accuracyNN(clf, Xtest, Ttest)
print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy2 - accuracy1:", accuracy2 - accuracy1)


def ceNN(clf, X, T):
    t = np.zeros((T.size, T.max() + 1))
    t[np.arange(T.size), T] = 1
    w1, b1 = clf.coefs_[0], clf.intercepts_[0]
    v1 = 1 / (1 + np.exp(-(X @ w1 + b1)))
    w2, b2 = clf.coefs_[1], clf.intercepts_[1]
    v = v1 @ w2 + b2
    y = np.log(softmax(v))
    CE1 = np.sum((clf.predict_log_proba(X)) * (-t)) / t.shape[0]
    CE2 = np.sum((y) * (-t)) / t.shape[0]
    print("cross entropy1: ", CE1)
    print("cross entropy2: ", CE2)
    print("cross entropy2 - cross entropy1: ", CE2 - CE1)


print("\n")
print("Question 3(h).")
print("-------------")
np.random.seed(0)
clf = MLPClassifier(hidden_layer_sizes=9, activation='logistic', solver='sgd',
                    learning_rate_init=0.01, tol=10 ** -6, max_iter=1000)
clf.fit(Xtrain, Ttrain)
ceNN(clf, Xtest, Ttest)

# Question 5
print("\n")
print("Question 5.")
print("-------------")
with open('mnistTVT.pickle', 'rb') as f:
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)


# Question 5a)
def reduce(d1, d2):
    global Xtrain2, Ttrain2, Xtest2, Ttest2, XtrainSmall, TtrainSmall
    idx = (Ttrain == d1) | (Ttrain == d2)
    Xtrain2 = Xtrain[idx]
    Ttrain2 = Ttrain[idx]
    Ttrain2[Ttrain2 == d1] = 1
    Ttrain2[Ttrain2 == d2] = 0
    idx = (Ttest == d1) | (Ttest == d2)
    Xtest2 = Xtest[idx]
    Ttest2 = Ttest[idx]
    Ttest2[Ttest2 == d1] = 1
    Ttest2[Ttest2 == d2] = 0
    N = 2000
    XtrainSmall = Xtrain2[:N]
    TtrainSmall = Ttrain2[:N]


reduce(5, 6)


# Question 5b)
def evaluateNN(clf, X, T):
    accuracy1 = clf.score(X, T)
    w1, b1 = clf.coefs_[0], clf.intercepts_[0]
    v1 = np.tanh(X @ w1 + b1)
    w2, b2 = clf.coefs_[1], clf.intercepts_[1]
    v2 = np.tanh(v1 @ w2 + b2)
    w3, b3 = clf.coefs_[2], clf.intercepts_[2]
    v = 1 / (1 + np.exp(-(v2 @ w3 + b3)))
    p = np.column_stack((((np.ones((v.shape[0], v.shape[1]))) - v), v))
    predict = np.argmax(p, axis=1)
    t = (predict == T)
    accuracy2 = np.mean(t)
    t = np.zeros((T.size, T.max() + 1))
    t[np.arange(T.size), T] = 1
    y = np.log(p)
    CE1 = -np.mean(t * np.log(clf.predict_proba(X)) + (1 - t) * np.log(
        1 - clf.predict_proba(X)))
    CE2 = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
    return accuracy1, accuracy2, CE1, CE2


np.random.seed(0)
clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation="tanh",
                    solver='sgd', tol=10 ** -6, learning_rate_init=0.01,
                    batch_size=100, max_iter=100)
clf.fit(Xtrain2, Ttrain2)

print("\n")
print("Question 5(c).")
print("-------------")
accuracy1, accuracy2, CE1, CE2 = evaluateNN(clf, Xtest2, Ttest2)
print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy2 - accuracy1:", accuracy2 - accuracy1)
print("cross entropy1: ", CE1)
print("cross entropy2: ", CE2)
print("cross entropy2 - cross entropy1: ", CE2 - CE1)

# print("\n")
# print("Question 5(d).")
# print("-------------")
accuracy2_ = []
CE2_ = []
for k in range(14):
    #   start = time.time()
    np.random.seed(0)
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation="tanh",
                        solver='sgd', tol=10 ** -6, batch_size=2 ** k,
                        max_iter=1, learning_rate_init=0.001)
    clf.fit(Xtrain2, Ttrain2)
    accuracy1, accuracy2, CE1, CE2 = evaluateNN(clf, Xtest2, Ttest2)
    accuracy2_.append(accuracy2)
    CE2_.append(CE2)
    # end = time.time()
    # print("batch size (2**{}) runtime:".format(k),end-start)
plt.semilogx(2 ** (np.arange(0, 14)), accuracy2_)
plt.xlabel("batch size")
plt.ylabel("accuracy")
plt.title("Question 5(d): Accuracy v.s. batch size")
plt.show()
plt.semilogx(2 ** (np.arange(0, 14)), CE2_)
plt.xlabel("batch size")
plt.ylabel("cross entropy")
plt.title("Question 5(d): Cross entropy v.s. batch size")
plt.show()

print("\n")
print("Question 5(f).")
print("-------------")


def forward_pass(X, W, w0, V, v0, U, u0):
    XT = X @ W + w0
    H = np.tanh(XT)
    HT = H @ V + v0
    G = np.tanh(HT)
    GT = G @ U + u0
    O = 1 / (1 + np.exp(-GT))
    return O, G, H


def accuracy(X, T, W, w0, V, v0, U, u0):
    O, G, H = forward_pass(X, W, w0, V, v0, U, u0)
    p = (np.ones((O.shape[0], O.shape[1]))) - O
    p = np.column_stack((((np.ones((O.shape[0], O.shape[1]))) - O), O))
    predict = np.argmax(p, axis=1)
    t = (predict == T)
    return np.mean(t)


def coss_entropy(X, T, W, w0, V, v0, U, u0):
    t = np.zeros((T.size, T.max() + 1))
    t[np.arange(T.size), T] = 1
    O, G, H = forward_pass(X, W, w0, V, v0, U, u0)
    p = (np.ones((O.shape[0], O.shape[1]))) - O
    p = np.column_stack((((np.ones((O.shape[0], O.shape[1]))) - O), O))
    return np.mean(-t * np.log(p) - (1 - t) * np.log(1 - p))


def bgd(Xt, Tt, X, T, i, lrate):
    num = np.shape(Xt)[0]
    lrate = lrate / num
    input_num = np.shape(Xt)[1]
    output_num = 1
    np.random.seed(0)
    W = rnd.randn(input_num, 100)
    V = rnd.randn(100, 100)
    U = rnd.randn(100, output_num)
    w0 = np.zeros([1, 100])
    v0 = np.zeros([1, 100])
    u0 = np.zeros([1, output_num])
    for n in range(i):
        print('Epoch', n, "Test accuracy=", accuracy(X, T, W, w0, V, v0, U, u0))
        # forward pass
        O, G, H = forward_pass(Xt, W, w0, V, v0, U, u0)
        # backward pass
        gGT = O - Tt.reshape(Tt.shape[0], 1)
        gU = G.T @ gGT
        gu0 = np.sum(gGT, axis=0)
        gG = gGT @ U.T
        gHT = (1 - G ** 2) * gG
        gV = H.T @ gHT
        gv0 = np.sum(gHT, axis=0)
        gH = gHT @ V.T
        gXT = (1 - H ** 2) * gH
        gW = Xt.T @ gXT
        gw0 = np.sum(gXT, axis=0)
        # update weights and biases
        U -= lrate * gU
        u0 -= lrate * gu0
        V -= lrate * gV
        v0 -= lrate * gv0
        W -= lrate * gW
        w0 -= lrate * gw0
    print("Test accuracy=", accuracy(X, T, W, w0, V, v0, U, u0))
    print("cross entropy=", coss_entropy(X, T, W, w0, V, v0, U, u0))


bgd(Xtrain2, Ttrain2, Xtest2, Ttest2, 10, 0.1)

print("\n")
print("Question 5(g).")
print("-------------")


def sgd(Xt, Tt, Xtt, Ttt, i, lrate, size):
    np.random.seed(0)
    lrate = lrate / size
    input_num = np.shape(Xt)[1]
    output_num = 1
    W = rnd.randn(input_num, 100)
    V = rnd.randn(100, 100)
    U = rnd.randn(100, output_num)
    w0 = np.zeros([1, 100])
    v0 = np.zeros([1, 100])
    u0 = np.zeros([1, output_num])
    for n in range(i):
        print('Iteration', n, "Test accuracy=",
              accuracy(Xtt, Ttt, W, w0, V, v0, U, u0))
        Xt, Tt = shuffle(Xt, Tt)
        p1 = 0
        # epoch begin
        while p1 < np.shape(Xt)[0]:
            p2 = np.min((p1 + size, np.shape(Xt)[0]))
            # mini-batches are created
            X = Xt[p1:p2]
            T = Tt[p1:p2]
            p1 = p2
            # forward pass
            O, G, H = forward_pass(X, W, w0, V, v0, U, u0)
            # backward pass
            gGT = O - T.reshape(T.shape[0], 1)
            gU = G.T @ gGT
            gu0 = np.sum(gGT, axis=0)
            gG = gGT @ U.T
            gHT = (1 - G ** 2) * gG
            gV = H.T @ gHT
            gv0 = np.sum(gHT, axis=0)
            gH = gHT @ V.T
            gXT = (1 - H ** 2) * gH
            gW = X.T @ gXT
            gw0 = np.sum(gXT, axis=0)
            # update weights and biases
            U -= lrate * gU
            u0 -= lrate * gu0
            V -= lrate * gV
            v0 -= lrate * gv0
            W -= lrate * gW
            w0 -= lrate * gw0
    print("Test accuracy=",
          accuracy(Xtt, Ttt, W, w0, V, v0, U, u0))
    print("cross entropy=",
          coss_entropy(Xtt, Ttt, W, w0, V, v0, U, u0))


sgd(Xtrain2, Ttrain2, Xtest2, Ttest2, 10, 0.1, 10)

if __name__ == "__main__":
    print("---E.N.D---")
