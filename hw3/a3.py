import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis as da
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
from sklearn.utils.testing import ignore_warnings

# """Question 1"""
#
# print("\n")
# print("Question 1.")
# print("-------------")
#
# with open('./mnistTVT.pickle', 'rb') as f:
#     Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)
# Xtrain = Xtrain.astype(np.float64)
# Xval = Xval.astype(np.float64)
# Xtest = Xtest.astype(np.float64)
#
#
# # print("\n")
# # print("Question 1(a).")
# # print("-------------")
#
#
# def data_project(dim, q, Xtrain, Xtest):
#     pca = PCA(n_components=dim)
#     pca.fit(Xtrain)
#     reduced_data = pca.transform(Xtest)
#     projected_data = pca.inverse_transform(reduced_data)
#     plt.figure()
#     X = np.reshape(projected_data, [-1, 28, 28])
#     for d in range(25):
#         plt.subplot(5, 5, d + 1)
#         plt.imshow(X[d], cmap='Greys')
#         plt.axis('off')
#     plt.suptitle('Question 1({}): MNIST test data projected onto {} dimensions'.format(q, dim))
#     plt.show()
#
#
# data_project(30, 'a', Xtrain, Xtest)
#
# # print("\n")
# # print("Question 1(b).")
# # print("-------------")
# data_project(3, 'b', Xtrain, Xtest)
#
# # print("\n")
# # print("Question 1(c).")
# # print("-------------")
# data_project(300, 'c', Xtrain, Xtest)
#
#
# # print("\n")
# # print("Question 1(d).")
# # print("-------------")
#
# def myPCA(X, K):
#     mean = np.mean(X, axis=0)
#     cov = np.dot((X - mean).T, X - mean) / (X.shape[0] - 1)
#     eig_val, eig_vec = np.linalg.eigh(cov)
#     u = np.flip(eig_vec, axis=1)[:, :K]
#     z = np.dot(X - mean, u)
#     return z @ u.T + mean
#
#
# print("\n")
# print("Question 1(f).")
# print("-------------")
#
#
# def RMS(diff):
#     return np.sqrt(np.sum(np.square(diff) / diff.shape[0]))
#
#
# myXtrainP = myPCA(Xtrain[0:5000], 100)
# plt.figure()
# X = np.reshape(myXtrainP, [-1, 28, 28])
# for d in range(25):
#     plt.subplot(5, 5, d + 1)
#     plt.imshow(X[d], cmap='Greys')
#     plt.axis('off')
# plt.suptitle('Question 1(f): MNIST data projected onto 100 dimensions (mine)')
# plt.show()
#
# pca = PCA(n_components=100, svd_solver='full')
# pca.fit(Xtrain[0:5000])
# reduced_data = pca.transform(Xtrain[0:5000])
# XtrainP = pca.inverse_transform(reduced_data)
# plt.figure()
# X = np.reshape(XtrainP, [-1, 28, 28])
# for d in range(25):
#     plt.subplot(5, 5, d + 1)
#     plt.imshow(X[d], cmap='Greys')
#     plt.axis('off')
# plt.suptitle('Question 1(f): MNIST data projected onto 100 dimensions (sklearn)')
# plt.show()
# print('RMS:', RMS(XtrainP - myXtrainP))
#
# """Question 2"""
#
# print("\n")
# print("Question 2.")
# print("-------------")
# with open('./mnistTVT.pickle', 'rb') as f:
#     Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)
# Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = Xtrain.astype(np.float64), Ttrain.astype(np.float64), Xval.astype(
#     np.float64), Tval.astype(np.float64), Xtest.astype(np.float64), Ttest.astype(np.float64)
# DXtrain = Xtrain[0:300, :]
# DTtrain = Ttrain[0:300]
# Xtrain = Xtrain[0:200, :]
# Ttrain = Ttrain[0:200]
#
# print("\n")
# print("Question 2(a).")
# print("-------------")
#
# clf = da.QuadraticDiscriminantAnalysis(store_covariance=True)
# ignore_warnings(clf.fit)(Xtrain, Ttrain)
# accuracy1 = clf.score(Xtrain, Ttrain)
# accuracy2 = clf.score(Xtest, Ttest)
# print('Accuracy of the classifier on the small training set =', accuracy1)
# print('Accuracy of the classifier on the (full) test set =', accuracy2)
#
# print("\n")
# print("Question 2(b).")
# print("-------------")
#
# training_accuracy = []
# validation_accuracy = []
# for n in range(20):
#     clf = da.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=2 ** (-n))
#     ignore_warnings(clf.fit)(Xtrain, Ttrain)
#     accuracy1 = clf.score(Xtrain, Ttrain)
#     accuracy2 = clf.score(Xval, Tval)
#     training_accuracy.append(accuracy1)
#     validation_accuracy.append(accuracy2)
# m = np.argmax(validation_accuracy)
# print("maximum validation accuracy =", validation_accuracy[m])
# print("training accuracy =", training_accuracy[m])
# print("regularization parameter =", 2.0 ** (-m))
# plt.figure()
# plt.semilogx(2.0 ** (-np.arange(0, 20)), training_accuracy, 'b')
# plt.semilogx(2.0 ** (-np.arange(0, 20)), validation_accuracy, 'r')
# plt.suptitle('Question 2(b): Training and Validation Accuracy for Regularized QDA')
# plt.xlabel('Regularization parameter')
# plt.ylabel('Accuracy')
# plt.show()
#
# print("\n")
# print("Question 2(d).")
# print("-------------")
#
#
# def train2d(K, X, T):
#     pca = PCA(n_components=K, svd_solver="full")
#     pca.fit(X)
#     reduced_data = pca.transform(X)
#     clf = da.QuadraticDiscriminantAnalysis(store_covariance=True)
#     ignore_warnings(clf.fit)(reduced_data, T)
#     accuracy1 = clf.score(reduced_data, T)
#     return pca, clf, accuracy1
#
#
# def test2d(pca, qda, X, T):
#     reduced_data = pca.transform(X)
#     accuracy2 = qda.score(reduced_data, T)
#     return accuracy2
#
#
# training_accuracy = []
# validation_accuracy = []
# for K in range(1, 51):
#     pca, clf, accuracy1 = train2d(K, Xtrain, Ttrain)
#     accuracy2 = test2d(pca, clf, Xval, Tval)
#     training_accuracy.append(accuracy1)
#     validation_accuracy.append(accuracy2)
# m = np.argmax(validation_accuracy)
# print("maximum validation accuracy =", validation_accuracy[m])
# print("training accuracy =", training_accuracy[m])
# print("K =", m + 1)
# plt.figure()
# plt.plot(np.arange(1, 51), training_accuracy, 'b')
# plt.plot(np.arange(1, 51), validation_accuracy, 'r')
# plt.suptitle('Question 2(d): Training and Validation Accuracy for PCA + QDA')
# plt.xlabel('Reduced dimension')
# plt.ylabel('Accuracy')
# plt.show()
#
# print("\n")
# print("Question 2(f).")
# print("-------------")
#
# accMax, reg, k = 0, 0, 0
# accMaxK = []
# curr_K = 0
# max_train = 0
# for K in range(1, 51):
#     curr_K = 0
#     for n in range(20):
#         pca = PCA(n_components=K, svd_solver="full")
#         pca.fit(Xtrain)
#         reduced_Xtrain = pca.transform(Xtrain)
#         reduced_Xval = pca.transform(Xval)
#         clf = da.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=2.0 ** (-n))
#         ignore_warnings(clf.fit)(reduced_Xtrain, Ttrain)
#         accuracy2 = clf.score(reduced_Xval, Tval)
#         if accuracy2 > accMax:
#             max_train = clf.score(reduced_Xtrain, Ttrain)
#             accMax = accuracy2
#             reg = 2.0 ** (-n)
#             k = K
#         if accuracy2 > curr_K:
#             curr_K = accuracy2
#     accMaxK.append(curr_K)
# print("maximum validation accuracy =", accMax)
# print("Training accuracy =", max_train)
# print("regularization parameter =", reg)
# print("K =", k)
# plt.figure()
# plt.plot(np.arange(1, 51), accMaxK)
# plt.suptitle('Question 2(f): Maximum validation accuracy for QDA')
# plt.xlabel('Reduced dimension')
# plt.ylabel('Accuracy')
# plt.show()
#
# """Question 3"""
#
# print("\n")
# print("Question 3.")
# print("-------------")
# with open('./mnistTVT.pickle', 'rb') as f:
#     Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)
# Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = Xtrain.astype(np.float64), Ttrain.astype(np.float64), Xval.astype(
#     np.float64), Tval.astype(np.float64), Xtest.astype(np.float64), Ttest.astype(np.float64)
# DXtrain = Xtrain[0:300, :]
# DTtrain = Ttrain[0:300]
# Xtrain = Xtrain[0:200, :]
# Ttrain = Ttrain[0:200]
#
#
# # print("\n")
# # print("Question 3(a).")
# # print("-------------")
#
# def myBootstrap(X, T):
#     Xsample, Tsample = resample(X, T)
#     u, indices = np.unique(Tsample, return_index=True)
#     while len(u) != 10 and np.min(indices) < 3:
#         Xsample, Tsample = resample(X, T)
#         u, indices = np.unique(Tsample, return_index=True)
#     return Xsample, Tsample
#
#
# print("\n")
# print("Question 3(b).")
# print("-------------")
#
# clf1 = da.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.004)
# ignore_warnings(clf1.fit)(Xtrain, Ttrain)
# accuracy1 = clf1.score(Xval, Tval)
# print("validation accuracy of the base classifier :", accuracy1)
# prob_mat = None
# for i in range(50):
#     Xsample, Tsample = myBootstrap(Xtrain, Ttrain)
#     clf2 = da.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.004)
#     ignore_warnings(clf2.fit)(Xsample, Tsample)
#     if i == 0:
#         prob_mat = clf2.predict_proba(Xval)
#     else:
#         prob_mat += clf2.predict_proba(Xval)
# prob_mat = prob_mat / 50
# predict = np.argmax(prob_mat, axis=1)
# t = (predict == Tval)
# accuracy2 = np.mean(t)
# print("validation accuracy of the bagged classifier :", accuracy2)
#
# # print("\n")
# # print("Question 3(c).")
# # print("-------------")
#
# prob_mat = None
# val_acc = []
# for i in range(500):
#     Xsample, Tsample = myBootstrap(Xtrain, Ttrain)
#     clf2 = da.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=0.004)
#     ignore_warnings(clf2.fit)(Xsample, Tsample)
#     accuracy1 = clf1.score(Xval, Tval)
#     if i == 0:
#         prob_mat = clf2.predict_proba(Xval)
#     else:
#         prob_mat += clf2.predict_proba(Xval)
#     predict = np.argmax(prob_mat / (i + 1), axis=1)
#     t = (predict == Tval)
#     val_acc.append(np.mean(t))
# plt.figure()
# plt.plot(np.arange(1, 501), val_acc)
# plt.suptitle('Question 3(c): Validation accuracy')
# plt.xlabel('Number of bootstrap samples')
# plt.ylabel('Accuracy')
# plt.show()
# plt.figure()
# plt.semilogx(np.arange(1, 501), val_acc)
# plt.suptitle('Question 3(c): Validation accuracy (log scale)')
# plt.xlabel('Number of bootstrap samples')
# plt.ylabel('Accuracy')
# plt.show()
#
#
# # print("\n")
# # print("Question 3(d).")
# # print("-------------")
#
# def train3d(K, R, X, T):
#     pca = PCA(n_components=K, svd_solver="full")
#     pca.fit(X)
#     reduced_data = pca.transform(X)
#     clf = da.QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=R)
#     ignore_warnings(clf.fit)(reduced_data, T)
#     return pca, clf
#
#
# def proba3d(pca, qda, X):
#     reduced_data = pca.transform(X)
#     return qda.predict_proba(reduced_data)
#
#
# # print("\n")
# # print("Question 3(e).")
# # print("-------------")
#
# def myBag(K, R):
#     pca1, qda1 = train3d(K, R, Xtrain, Ttrain)
#     reduced_Xval = pca1.transform(Xval)
#     accuracy_base = qda1.score(reduced_Xval, Tval)
#     prob_mat = None
#     for i in range(200):
#         Xsample, Tsample = myBootstrap(Xtrain, Ttrain)
#         pca2, qda2 = train3d(K, R, Xsample, Tsample)
#         pred_prob = proba3d(pca2, qda2, Xval)
#         if i == 0:
#             prob_mat = pred_prob
#         else:
#             prob_mat += pred_prob
#     prob_mat = prob_mat / 200
#     predict = np.argmax(prob_mat, axis=1)
#     t = (predict == Tval)
#     accuracy_bagged = np.mean(t)
#     return accuracy_base, accuracy_bagged
#
#
# print("\n")
# print("Question 3(f).")
# print("-------------")
#
# accuracy1, accuracy2 = myBag(100, 0.01)
# print("validation accuracy of the base classifier :", accuracy1)
# print("validation accuracy of the bagged classifier :", accuracy2)
#
# print("\n")
# print("Question 3(g).")
# print("-------------")
#
# val_base = []
# val_bagged = []
# for i in range(50):
#     K = np.random.randint(1, 11)
#     R = np.random.uniform(0.2, 1)
#     accuracy_base, accuracy_bagged = myBag(K, R)
#     val_base.append(accuracy_base)
#     val_bagged.append(accuracy_bagged)
# plt.figure()
# plt.scatter(val_base, val_bagged, c='b')
# plt.suptitle('Question 3(g): Bagged v.s. base validation accuracy')
# plt.xlabel('Base validation accuracy')
# plt.ylabel('Bagged validation accuracy')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.plot((0, 1), (0, 1), 'r')
# plt.show()
#
# print("\n")
# print("Question 3(h).")
# print("-------------")
#
# val_base = []
# val_bagged = []
# for i in range(50):
#     K = np.random.randint(50, 201)
#     R = np.random.uniform(0, 0.05)
#     accuracy_base, accuracy_bagged = myBag(K, R)
#     val_base.append(accuracy_base)
#     val_bagged.append(accuracy_bagged)
# plt.close()
# plt.figure()
# plt.scatter(val_base, val_bagged, c='b')
# plt.suptitle('Question 3(h): Bagged v.s. base validation accuracy')
# plt.xlabel('Base validation accuracy')
# plt.ylabel('Bagged validation accuracy')
# plt.ylim(0, 1)
# max_idx = np.argmax(val_bagged)
# plt.axhline(y=val_bagged[max_idx], color='r', linestyle='-')
# plt.show()
# print("maximum bagged validation accuracy =", max(val_bagged))
#
# """Question 4"""
#
with open('./dataA2Q2.pickle', 'rb') as file:
    q2_dataTrain, q2_dataTest = pickle.load(file)
Xtrain, Ttrain = q2_dataTrain
Xtest, Ttest = q2_dataTest
#
#
# # print("\n")
# # print("Question 4(a).")
# # print("-------------")

def plot_clusters(X, R, Mu):
    order = np.argsort(np.sum(R, axis=0))
    R = R[:, order]
    plt.scatter(X[:, 0], X[:, 1], color=R, s=5)
    plt.scatter(Mu[:, 0], Mu[:, 1], color="black")

#
# print("\n")
# print("Question 4(b).")
# print("-------------")
#
# KM = KMeans(n_clusters=3).fit(Xtrain)
# Mu = KM.cluster_centers_
# d = KM.transform(Xtrain)
# R = np.zeros_like(d)
# R[np.arange(len(d)), d.argmin(1)] = 1
# plot_clusters(Xtrain, R, Mu)
# plt.suptitle('Question 4(b): K means')
# plt.show()
# KM_score1 = KM.score(Xtrain)
# KM_score2 = KM.score(Xtest)
# print("Training data score:", KM_score1)
# print("Test data score:", KM_score2)
#
# print("\n")
# print("Question 4(c).")
# print("-------------")
#
# GM = GaussianMixture(covariance_type='spherical', n_components=3).fit(Xtrain)
# Mu = GM.means_
# R = GM.predict_proba(Xtrain)
# plot_clusters(Xtrain, R, Mu)
# plt.suptitle('Question 4(c): Gaussian mixture model (spherical)')
# plt.show()
# s_score1 = GM.score(Xtrain)
# s_score2 = GM.score(Xtest)
# print("Training data score:", s_score1)
# print("Test data score:", s_score2)
#
# print("\n")
# print("Question 4(d).")
# print("-------------")
#
# GM = GaussianMixture(covariance_type='full', n_components=3).fit(Xtrain)
# Mu = GM.means_
# R = GM.predict_proba(Xtrain)
# plot_clusters(Xtrain, R, Mu)
# plt.suptitle('Question 4(d): Gaussian mixture model (full)')
# plt.show()
# f_score1 = GM.score(Xtrain)
# f_score2 = GM.score(Xtest)
# print("Training data score:", f_score1)
# print("Test data score:", f_score2)
# print("Q4d-Q4c test scores =", f_score2 - s_score2)
#
# print("\n")
# print("Question 4(e).")
# print("-------------")
#
#
# def assignment(X, center, j):
#     A = np.reshape(X, [X.shape[0], 1, X.shape[1]])
#     B = np.reshape(center, [1, center.shape[0], center.shape[1]])
#     C = np.sum(np.square(A - B), axis=2)
#     r = np.argmin(C, axis=1)
#     j.append(np.sum(np.min(C, axis=1)))
#     return r, j
#
#
# def myKmeans(X, K, I):
#     # randomly initialize cluster centres (basic K-means initialization)
#     n = X.shape[0]
#     index = (np.floor(np.random.random(size=K) * n)).astype(int)
#     center = X[index]
#     j = []
#     for i in range(I):
#         r, j = assignment(X, center, j)
#         split = np.full((len(r), len(X[0]), K), np.nan)
#         split[np.arange(len(r)), :, r] = 1
#         center = np.nanmean(X[:, :, None] * split, axis=0).T
#     r, j = assignment(X, center, j)
#     R = np.zeros((len(r), K))
#     R[np.arange(len(r)), r] = 1
#     return center, R, j
#
#
# center, R, j = myKmeans(Xtrain, 3, 100)
# plt.figure()
# plt.semilogx(np.arange(0, 20), j[:20])
# plt.suptitle('Question 4(e): score v.s. iteration (K means)')
# plt.xlabel('Iteration')
# plt.ylabel('Score')
# plt.show()
# plot_clusters(Xtrain, R, center)
# plt.suptitle('Question 4(e): Data clustered by K means')
# plt.show()
#
#
# def scoreKmeans(X, Mu):
#     A = np.reshape(X, [X.shape[0], 1, X.shape[1]])
#     B = np.reshape(center, [1, Mu.shape[0], Mu.shape[1]])
#     C = np.sum(np.square(A - B), axis=2)
#     return np.sum(np.min(C, axis=1))
#
#
# my_KM_score1 = scoreKmeans(Xtrain, center)
# my_KM_score2 = scoreKmeans(Xtest, center)
# print("Training data score:", my_KM_score1)
# print("Test data score:", my_KM_score2)
# print("Q4e+Q4b test scores =", KM_score2 + my_KM_score2)
#
# print("\n")
# print("Question 4(f).")
# print("-------------")


def E_step(X, Mu, Pi):
    A = np.reshape(X, [X.shape[0], 1, X.shape[1]])
    B = np.reshape(Mu, [1, Mu.shape[0], Mu.shape[1]])
    C = np.sum(np.square(A - B), axis=2)
    numerator = Pi * np.exp(-1 / 2 * C)
    r = numerator / (numerator.sum(axis=1)[:, np.newaxis])
    return r


def M_step(X, r):
    n_clusters = r.shape[1]
    r_sp = np.hsplit(r, n_clusters)
    mult = np.sum(X * r_sp, axis=1)
    Mu = mult / np.sum(r_sp, axis=1)
    Pi = r.sum(axis=0) / X.shape[0]
    return Mu, Pi


def scoreGMM(X, Mu, Pi):
    M = len(X[0])
    A = np.reshape(X, [X.shape[0], 1, X.shape[1]])
    B = np.reshape(Mu, [1, Mu.shape[0], Mu.shape[1]])
    C = np.sum(np.square(A - B), axis=2)
    numerator = (Pi * np.exp(-1 / 2 * C)) / ((2 * np.pi) ** (M / 2))
    return np.mean(np.log(numerator.sum(axis=1)))


def myGMM(X, K, I):
    # randomly initialize cluster centres
    n = X.shape[0]
    index = (np.floor(np.random.random(size=K) * n)).astype(int)
    Mu = X[index]
    # initial Pi with equally percentage to k clusters
    Pi = np.full(K, 1 / K)
    scores = []
    for i in range(I):
        scores.append(scoreGMM(X, Mu, Pi))
        r = E_step(X, Mu, Pi)
        Mu, Pi = M_step(X, r)
    return Mu, Pi, r, scores


Mu, Pi, r, scores = myGMM(Xtrain, 3, 100)
plt.figure()
plt.plot(np.arange(0, 20), scores[:20])
plt.suptitle('Question 4(f): a plot of score v.s. iteration number for the first 20 iterations')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.show()
plot_clusters(Xtrain, r, Mu)
plt.suptitle('Question 4(f): clustered training data')
plt.show()
scoref1 = scoreGMM(Xtrain, Mu, Pi)
scoref2 = scoreGMM(Xtest, Mu, Pi)
print("Training data score:", scoref1)
print("Test data score:", scoref2)
# print("Q4c-Q4f test scores =", s_score2 - scoref2)

if __name__ == "__main__":
    print("E.N.D")
