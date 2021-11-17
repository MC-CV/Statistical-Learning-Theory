from sklearn.linear_model import Ridge,Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import seaborn

if __name__ == '__main__':
    boston = load_boston()
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target)
    ri = Ridge(alpha=0.5)
    ri.fit(x_train,y_train)
    y_pred = ri.predict(x_test)

    output = ri.predict(boston.data)
    factor_list = list(boston.feature_names)
    facvals = [ [boston.data[i][index] for i in range(len(boston.data))] for index in range(len(factor_list)) ]
    factor_list.append('PRED')
    facvals.append(output)
    factor_dict = dict(zip(factor_list,facvals))
    boston_df = pd.DataFrame(
                factor_dict
                )
    df_corr = boston_df.corr()
    seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
    plt.show()


    la = Lasso(alpha=0.5,fit_intercept=True)
    la.fit(x_train,y_train)
    y_pred_la = la.predict(x_test)

    plt.plot(y_pred,'ro-',label="y_pred")
    plt.plot(y_test,'go--',label="y_test")
    plt.title("Ridge")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.legend()
    plt.show()

    plt.plot(y_pred_la,'ro-',label="y_pred")
    plt.plot(y_test,'go--',label="y_test")
    plt.title("Lasso")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.legend()
    plt.show()

    print('Ridge_r2_score:',r2_score(y_test,y_pred))
    print('Ridge_mean_squared_error',mean_squared_error(y_test,y_pred))
    print('Ridge_mean_absolute_error',mean_absolute_error(y_test,y_pred))

    print('Lasso_r2_score:',r2_score(y_test,y_pred_la))
    print('Lasso_mean_squared_error',mean_squared_error(y_test,y_pred_la))
    print('Lasso_mean_absolute_error',mean_absolute_error(y_test,y_pred_la))


    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    duoyuan = PolynomialFeatures(degree=2)
    X2_train = duoyuan.fit_transform(x_train)
    X2_test = duoyuan.fit_transform(x_test)
    y = boston.target
    lr.fit(X2_train,y_train)
    y_pred_po = lr.predict(X2_test)
    
    print('Poly_r2_score:',r2_score(y_test,y_pred_po))
    print('Poly_mean_squared_error',mean_squared_error(y_test,y_pred_po))
    print('Poly_mean_absolute_error',mean_absolute_error(y_test,y_pred_po))
    alp = [i/10 for i in range(0,10)]
    coef = []
    for i in alp:
        la1 = Lasso(alpha=i,fit_intercept=True)
        la1.fit(x_train,y_train)
        coef.append(la1.coef_)
        # print(la1.coef_)
    for i in range(len(boston.feature_names)):
        temp = [coef[j][i] for j in range(len(coef))]
        plt.subplot(3,5,i+1)
        plt.plot(alp,temp,label="Lasso_coef", color='b')
        plt.xlabel("Lasso_Alpha")
        plt.ylabel("Lasso_Coefficient")
        plt.title(boston.feature_names[i])
        plt.legend()
    plt.show()

    coef_Ri = []
    for i in alp:
        ri1 = Ridge(alpha=i,fit_intercept=True)
        ri1.fit(x_train,y_train)
        coef_Ri.append(ri1.coef_)
    for i in range(len(boston.feature_names)):
        temp = [coef_Ri[j][i] for j in range(len(coef_Ri))]
        plt.subplot(3,5,i+1)
        plt.plot(alp,temp,label="Ridge_coef", color='b')
        plt.xlabel("Ridge_Alpha")
        plt.ylabel("Ridge_Coefficient")
        plt.title(boston.feature_names[i])
        plt.legend()
    plt.show()