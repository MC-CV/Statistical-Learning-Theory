import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression,SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures



if __name__ == '__main__':
    boston = datasets.load_boston()
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.2,random_state=2)
    model = LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    plt.plot(y_pred,'ro-',label="y_pred")
    plt.plot(y_test,'go--',label="y_test")
    plt.title("mean_absolute_error_curve")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.legend()
    plt.show()
    fig, ax = plt.subplots()
    ax.set_xlabel("y_test")
    ax.set_ylabel("y_pred")
    plt.scatter(y_pred,y_test)
    plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'b',lw=5)   
    plt.show()
    print('r2_score:',r2_score(y_test,y_pred))
    print('mean_squared_error',mean_squared_error(y_test,y_pred))
    print('mean_absolute_error',mean_absolute_error(y_test,y_pred))

    sg = SGDRegressor()
    sg.fit(x_train,y_train)
    y_pred_sg = sg.predict(x_test)
    print('sg_r2_score:',r2_score(y_test,y_pred_sg))
    print('sg_mean_squared_error',mean_squared_error(y_test,y_pred_sg))
    print('sg_mean_absolute_error',mean_absolute_error(y_test,y_pred_sg))


    
