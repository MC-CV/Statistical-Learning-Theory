from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
boston = datasets.load_boston()
factor_list = list(boston.feature_names)
facvals = [ [boston.data[i][index] for i in range(len(boston.data))] for index in range(len(factor_list)) ]
factor_list.append('TARGET')
facvals.append(boston.target)
factor_dict = dict(zip(factor_list,facvals))
boston_df = pd.DataFrame(
			factor_dict, # columns = ['CRIM','CHAS','AGE','TARGET'],
			)
def plot(strfac):
	x_axis = list(boston_df[strfac])
	y_axis = list(boston_df['TARGET'])
	plt.title(strfac)
	plt.ylabel('Price')
	plt.scatter(x_axis,y_axis,s=3)

for i in range(len(factor_list)):
	index = i+1
	factor = factor_list[i]
	plt.subplot(3,5,index)
	plot(factor)

plt.show()
boston = datasets.load_boston()
df = pd.DataFrame(boston.data,columns=boston.feature_names)
df_corr = df.corr()
seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
plt.show()

model = LinearRegression()
duoyuan = PolynomialFeatures(degree=2)  
X2 = duoyuan.fit_transform(boston.data)
y = boston.target
model.fit(X2,y)
y_pred = model.predict(X2)
fig, ax = plt.subplots()
ax.set_xlabel("y_test")
ax.set_ylabel("y_pred")
plt.scatter(y_pred,y)
plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'b',lw=5)   
plt.show()
print('r2_score:',r2_score(y,y_pred))
print('mean_squared_error',mean_squared_error(y,y_pred))
print('mean_absolute_error',mean_absolute_error(y,y_pred))