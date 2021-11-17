import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import cv2

data = []
with open("brainimage.txt") as f:
    for line in f.readlines():
        temp = []
        s = line.strip().split("   ")
        for item in s:
            # temp.append(item)
            data.append(int(item))
data = np.array(data)

ss = MinMaxScaler()
data = data.reshape(-1, 1)
data = ss.fit_transform(data)

var = data.var()

gmm = GaussianMixture(
    n_components=3,
    weights_init=[1 / 3, 1 / 3, 1 / 3],
    means_init=[[0.3], [0.5], [0.7]],
    covariance_type="full",
)

gmm.fit(data)
predicted_cls = gmm.predict(data)

count_0 = 0
count_1 = 0
count_2 = 0

for i in predicted_cls:
    if i == 0:
        count_0 += 1
    elif i == 1:
        count_1 += 1
    elif i == 2:
        count_2 += 1
print(count_0, count_1, count_2)

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
