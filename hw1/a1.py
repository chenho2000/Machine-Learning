import numpy as np
import numpy.random as rnd
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
from sklearn import neighbors
import bonnerlib3 as bl3d

# Question 1
rnd.seed(3)
print("\n\nQuestion 1")
print("----------")
print("\nQuestion 1(a):")
B = rnd.random((4, 5))
print(B)
print("\nQuestion 1(b):")
y = rnd.random((4, 1))
print(y)
print("\nQuestion 1(c):")
C = B.reshape(2, 10)
print(C)
print("\nQuestion 1(d):")
D = B - y
print(D)
print("\nQuestion 1(e):")
z = np.reshape(y, 4)
print(z)
print("\nQuestion 1(f):")
B[:, 3] = z
print(B)
print("\nQuestion 1(g):")
D[:, 0] = B[:, 2] + z
print(D)
print("\nQuestion 1(h):")
print(B[:3, :])
print("\nQuestion 1(i):")
print(B[:, [1, 3]])
print("\nQuestion 1(j):")
print(np.log(B))
print("\nQuestion 1(k):")
print(B.sum())
print("\nQuestion 1(l):")
print(B.max(axis=0))
print("\nQuestion 1(m):")
print((B.sum(axis=1)).max())
print("\nQuestion 1(n):")
print(np.matmul(B.transpose(), D))
print("\nQuestion 1(o):")
print(np.matmul(np.matmul(np.matmul(y.transpose(), D), D.transpose()), y))


# Question 2
# Question 2(a)
def matrix_poly(A):
    n = A.shape[0]
    ans = np.zeros((n, n))
    f = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = A[i, j]
            for k in range(n):
                s += A[i, k] * A[k, j]
            f[i, j] = s
    for i in range(n):
        for j in range(n):
            a = A[i, j]
            for k in range(n):
                a += A[i, k] * f[k, j]
            ans[i, j] = a
    return ans


# Question 2(b)
def timing(n):
    print("timing ({})".format(n))
    a = rnd.random((n, n))
    start1 = time.time()
    b1 = matrix_poly(a)
    end1 = time.time()
    print("Execution time of matrix_poly is :", end1 - start1)
    start2 = time.time()
    b2 = a + np.matmul(a, (a + np.matmul(a, a)))
    end2 = time.time()
    print("Execution time of functions numpy.matmul and + is :", end2 - start2)
    magnitude = np.sqrt(np.sum(np.square(b1 - b2)))
    print("Time difference:", magnitude)


print("\n\nQuestion 2")
print("----------")
print("\nQuestion 2(c):")

timing(100)
timing(300)
timing(1000)


# Question 3
# Question 3(a)
def least_squares(x, t):
    n, = x.shape
    o = np.column_stack((np.ones(n), x))
    return np.dot(np.linalg.inv(np.dot(o.T, o)), np.dot(o.T, t))


# Question 3(b)
def plot_data(x, t):
    plt.scatter(x, t)
    b, a = least_squares(x, t)
    x = np.linspace(np.min(x), np.max(x), 100)
    y = a * x + b
    plt.plot(x, y, '-r')
    plt.suptitle("Question 3(b): the fitted line")
    plt.show()
    return


# Question 3(c)
def error(a, b, x, t):
    y = a * np.array(x) + b
    mse = np.square(np.subtract(t, y)).mean()
    return mse


print("\n\nQuestion 3")
print("----------")
print("\nQuestion 3(d):")
with open('dataA1Q3.pickle', 'rb') as f:
    dataTrain, dataTest = pickle.load(f)
plot_data(dataTrain[0], dataTrain[1])
b, a = least_squares(dataTrain[0], dataTrain[1])
print("a = ", a, "b = ", b)
print("Training error = ", error(a, b, dataTrain[0], dataTrain[1]))
print("Test error = ", error(a, b, dataTest[0], dataTest[1]))

# Question 4
print("\n\nQuestion 4")
print("----------")
print("\nQuestion 4(a):")
with open('dataA1Q4v2.pickle', 'rb') as f:
    x_train, t_train, x_test, t_test = pickle.load(f)
clf = lin.LogisticRegression()  # create a classification object, clf
clf.fit(x_train, t_train)  # learn a logistic-regression classifier
w = clf.coef_[0]  # weight vector
w0 = clf.intercept_[0]  # bias term
print("weight vector:", w)
print("bias term:", w0)
print("\nQuestion 4(b):")
accuracy1 = clf.score(x_test, t_test)
v = x_test @ w.T + w0
predict = (v > 0)
t = (predict == t_test)
accuracy2 = np.mean(t)
print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy2 - accuracy1:", accuracy2 - accuracy1)
bl3d.plot_db(x_train, t_train, w, w0, 30, 5)[0].set_title(
    'Question 4(c): Training data and decision boundary')
plt.show()
bl3d.plot_db(x_train, t_train, w, w0, 30, 20)[0].set_title(
    'Question 4(d): Training data and decision boundary')
plt.show()


# Question 5
def gd_logreg(lrate):
    np.random.seed(3)
    theta = rnd.randn(x_train.shape[1] + 1) / 1000
    a = np.full((x_train.shape[0], 1), 1)
    x = np.column_stack((a, x_train))
    t = np.column_stack((a, x_test))
    count = 0
    train_entropy = []
    test_entropy = []
    train_accuracy = []
    test_accuracy = []
    z = np.matmul(x, theta)
    v = np.matmul(t, theta)
    while True:
        y = 1.0 / (1.0 + np.exp(-z))
        theta = theta - lrate * (x.T @ (y - t_train)) / t_train.shape[0]
        z = np.matmul(x, theta)
        v = np.matmul(t, theta)
        test_entropy.append(
            (t_test @ np.logaddexp(0, -v) + (1 - t_test) @ np.logaddexp(0, v)) /
            t_train.shape[0])
        train_entropy.append(
            (t_train @ np.logaddexp(0, -z) + (1 - t_train) @ np.logaddexp(0, z)) /
            t_train.shape[0])
        predict1 = (z > 0)
        predict2 = (v > 0)
        train_accuracy.append(np.mean((predict1 == t_train)))
        test_accuracy.append(np.mean((predict2 == t_test)))
        if count != 0:
            if abs(train_entropy[-1] - train_entropy[-2]) < 10 ** -10:
                count += 1
                break
        count += 1
    print("\n\nQuestion 5")
    print("----------")
    print("\nQuestion 5(e):")
    print("final weight vector (including the bias term at index 0): ", theta)
    print("the number of iterations: ", count)
    print("learning rate: ", lrate)
    plt.suptitle("Question 5: Training and test loss v.s. iterations")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.plot(range(1, count + 1), train_entropy, c="blue")
    plt.plot(range(1, count + 1), test_entropy, c="red")
    plt.show()
    plt.suptitle(
        "Question 5: Training and test loss v.s. iterations (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.semilogx(range(1, count + 1), train_entropy, c="blue")
    plt.semilogx(range(1, count + 1), test_entropy, c="red")
    plt.show()
    plt.suptitle(
        "Question 5: Training and test accuracy v.s. iterations (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy")
    plt.semilogx(range(1, count + 1), train_accuracy, c="blue")
    plt.semilogx(range(1, count + 1), test_accuracy, c="red")
    plt.show()
    plt.suptitle("Question 5: last 100 training cross entropies")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.plot(range(count - 100, count), train_entropy[-100:], c="blue")
    plt.show()
    plt.suptitle("Question 5: test loss from iteration 50 on (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.semilogx(range(50, count), test_entropy[50:], c="blue")
    plt.show()
    bl3d.plot_db(x_train, t_train, theta[1:], theta[0], 30, 5)[0].set_title(
        'Question 5: Training data and decision boundary')
    return


gd_logreg(1)

# Question 6
with open('mnistTVT.pickle', 'rb') as f:
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)


def abc(d1, d2):
    T_test_check = np.logical_or((Ttest == d1), (Ttest == d2))
    T_test = Ttest[T_test_check]
    X_test = Xtest[T_test_check, :]
    T_train_check = np.logical_or((Ttrain == d1), (Ttrain == d2))
    T_train = Ttrain[T_train_check]
    X_train = Xtrain[T_train_check, :]
    T_val_check = np.logical_or((Tval == d1), (Tval == d2))
    T_val = Tval[T_val_check]
    X_val = Xval[T_val_check]
    sT_train = T_train[:2000]
    sX_train = X_train[:2000]
    # b)
    fig_b = plt.figure()
    for i in range(16):
        current = X_train[i].reshape(28, 28)
        fig_b.add_subplot(4, 4, i + 1)
        plt.imshow(current, cmap='Greys', interpolation='nearest')
        plt.axis("off")
    fig_b.suptitle('Question 6(b): 16 MNIST training images')
    plt.show()
    training_accuracies = []
    validation_accuracies = []
    for i in range(1, 20)[::2]:
        nb = neighbors.KNeighborsClassifier(n_neighbors=i)
        nb.fit(np.squeeze(X_train), np.squeeze(T_train))
        validation_accuracies.append(
            nb.score(np.squeeze(X_val), np.squeeze(T_val), sample_weight=None))
        training_accuracies.append(
            nb.score(np.squeeze(sX_train), np.squeeze(sT_train),
                     sample_weight=None))
    # print(training_accuracies)
    # print(validation_accuracies)
    plt.title(
        "Question 6(c): Training and Validation Accuracy for KNN, digits {} and {}".format(
            d1, d2))
    plt.xlabel("Number of Neighbours, K.")
    plt.ylabel("Error")
    plt.plot(range(1, 20)[::2], training_accuracies, c="blue")
    plt.plot(range(1, 20)[::2], validation_accuracies, c="red")
    plt.show()
    best = 2 * validation_accuracies.index(max(validation_accuracies)) + 1
    print("the best value of K:", best)
    print("validation accuracy", max(validation_accuracies))
    nbest = neighbors.KNeighborsClassifier(n_neighbors=best)
    nbest.fit(np.squeeze(X_train), np.squeeze(T_train))
    print("test accuracy", nbest.score(np.squeeze(X_test), np.squeeze(T_test),
                                       sample_weight=None))


print("\n\nQuestion 6")
print("----------")
print("\nQuestion 6(c):")
abc(5, 6)
print("\nQuestion 6(d):")
abc(4, 7)

if __name__ == "__main__":
    print("---A1 END---")
