from sklearn.tree import DecisionTreeClassifier
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data = pd.read_csv('Data.csv')
    X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,:-1],data['fetal_health'],test_size=0.1,random_state=666)
    feature_names = X_train.keys()
    class_names = ['1.0','2.0','3.0']
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    kernels = ["gini", "entropy"]
    Cs = [0,0.015,0.030,0.045]
    for i in kernels:
        for j in Cs:
            p1 = DecisionTreeClassifier(criterion = i,ccp_alpha = j)
            p1.fit(X_train,y_train)
            print(i,'C = ',j,'score = ',p1.score(X_test,y_test))


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    kf = KFold(n_splits=3)

    num = 0
    for j in kernels:
        for i in Cs:
            cls = DecisionTreeClassifier(criterion = j,ccp_alpha = i)
            t = 1
            for train_index, test_index in kf.split(X_train,y_train):
                
                X_train_1, X_test_1 = X_train[train_index], X_train[test_index]
                Y_train_1, Y_test_1 = y_train[train_index], y_train[test_index]
                cls.fit(X_train_1, Y_train_1)
                y_pred = cls.predict(X_test_1)
                Y_test = Y_test_1
                print('第',t,'次:',j,i)
                print(accuracy_score(Y_test, y_pred))
                # print ("test_Accuracy:", accuracy)
                # print ("test_error_rate:", 1-accuracy)
                t += 1

# 画图
# from sklearn.tree import export_graphviz

# model = DecisionTreeClassifier()
# model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
# export_graphviz(model,out_file='tree.dot',
#                feature_names = feature_names,
#                class_names = class_names,
#                rounded = True, proportion =False,
#                precision = 2, filled = True)

# import os
# os.environ["PATH"]+= os.pathsep + r'D:\Program Files\Graphviz\bin'
# # 将tree.dot文件转化为tree.png
# from subprocess import call
# call(['dot', '-Tpng','tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# # # 在jupyter notebook中查看决策树图像
# from IPython.display import Image
# Image(filename ='tree.png')