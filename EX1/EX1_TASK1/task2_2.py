from sklearn import datasets
from sklearn.model_selection import cross_val_predict,cross_val_score,cross_validate
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pandas as pd
from sklearn.linear_model import LinearRegression,SGDRegressor
import numpy as np


lr = linear_model.LinearRegression()
X, y = datasets.load_boston(return_X_y=True)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'b',lw=5)   
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

num = [i for i in range(2,10,1)]
score = []
fit_time = []
score_time = []
for i in num:
    score.append(np.max(cross_val_score(lr, X, y, cv=i)))
    fit_time.append(np.mean(cross_validate(lr, X, y, cv=i)['fit_time']))
    score_time.append(np.mean(cross_validate(lr, X, y, cv=i)['score_time']))       
print(np.max(cross_validate(lr, X, y, cv=3)))
fig, ax = plt.subplots()
plt.plot(num, score, color='r')
ax.scatter(num, score)
ax.set_xlabel('k')
ax.set_ylabel('score')
plt.show()
plt.plot(num,fit_time,label="fit_time", color='r')
plt.plot(num,score_time,label="score_time", color='b')
plt.xlabel("k")
plt.ylabel("time")
plt.legend()
plt.show()

