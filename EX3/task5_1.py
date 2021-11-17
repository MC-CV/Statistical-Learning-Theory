import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import cv2

plt.style.use("seaborn")

# 更新W
def update_W(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return W


# 更新pi
def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


# 计算log似然函数
def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


# 更新Mu
def update_Mu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu


# 更新Var
def update_Var(X, Mu, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
    return Var


if __name__ == "__main__":
    # 载入数据
    data = []
    with open("brainimage.txt") as f:
        for line in f.readlines():
            temp = []
            s = line.strip().split("   ")
            for item in s:
                data.append(int(item))
    data = np.array(data)

    ss = MinMaxScaler()
    data = data.reshape(-1, 1)
    data = ss.fit_transform(data)

    var = data.var()
    # 初始化
    n_clusters = 3
    n_points = len(data)
    Mu = [[0.3], [0.5], [0.7]]
    Var = [[var], [var], [var]]
    Pi = [1 / n_clusters] * 3
    W = np.ones((n_points, n_clusters)) / n_clusters
    Pi = W.sum(axis=0) / W.sum()
    # 迭代
    loglh = []
    for i in range(20):
        loglh.append(logLH(data, Pi, Mu, Var))
        W = update_W(data, Mu, Var, Pi)
        Pi = update_Pi(W)
        Mu = update_Mu(data, W)
        print("log-likehood:%.3f" % loglh[-1])
        Var = update_Var(data, Mu, W)

    pred = W
    predicted_cls = []
    for i in pred:
        i = list(i)
        predicted_cls.append(i.index(max(i)))

    # 画图
    idx = 0
    res = []
    for i in range(151):
        temp = []
        for j in range(171):
            if predicted_cls[idx] == 0:
                temp.append(255)
            else:
                temp.append(0)
            idx += 1
        res.append(temp)
    res = np.array(res)

    cv2.imwrite("outside brain.jpg", res)

    idx = 0
    res = []
    for i in range(151):
        temp = []
        for j in range(171):
            if predicted_cls[idx] == 1:
                temp.append(255)
            else:
                temp.append(0)
            idx += 1
        res.append(temp)
    res = np.array(res)

    cv2.imwrite("gray matter.jpg", res)

    idx = 0
    res = []
    for i in range(151):
        temp = []
        for j in range(171):
            if predicted_cls[idx] == 2:
                temp.append(255)
            else:
                temp.append(0)
            idx += 1
        res.append(temp)
    res = np.array(res)

    cv2.imwrite("white matter.jpg", res)

    epochs = [i for i in range(1, 21)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, loglh, label="log-likehood", color="blue")
    ax.set_xlabel("epochs")
    ax.set_ylabel("log-likehood")
    ax.legend(loc=0)
    plt.show()
