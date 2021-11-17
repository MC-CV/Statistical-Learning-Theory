import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error





class Perception(object):
    def __init__(self,X,Y,lr,num_arg):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.rate = lr
        self.final_w = 0
        self.final_b = 0
        self.num_arg = num_arg
    
    # def init_the_data(self):
    #     if 

    def train(self):
        num = [i for i in range(len(self.Y))]
        w = np.zeros(self.num_arg)
        b = 0
        while True:
            # 设置符号判断以保证所有的点都被正确分类才退出训练
            flag = True
            for i in num:
                inn = np.inner(w,self.X[i])
                if self.Y[i]*(inn+b) <= 0:
                    flag = False
                    w += self.rate*self.Y[i]*self.X[i]
                    b += self.rate*self.Y[i]
                else:
                    continue
            if flag:
                break
            else:
                continue
        self.final_w = w
        self.final_b = b
        print(self.final_w,self.final_b)
    def test(self,X,keys):
        self.pred = np.zeros(len(X))
        for i in range(len(X)):
            self.pred[i] = np.sign(np.inner(self.final_w,X.iloc[i])+self.final_b)
        return self.pred
    def eval(self,y_pred,y_true):
        y_true = np.array(y_true)
        num = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                num += 1
        print('score:',num/len(y_true))
        print('r2_score=',r2_score(y_pred,y_true))
        print('mean_absolute_error=',mean_absolute_error(y_pred,y_true))
        print('mean_squared_error=',mean_squared_error(y_pred,y_true))


if __name__ == '__main__':
    data = pd.read_csv('Data.csv')
    X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,:5],data['fetal_health'],test_size=0.2)
    p1 = Perception(X_train,y_train,1,len(X_train.keys()))
    p1.train()
    keys = data.keys()[:5]
    y_pred = p1.test(X_test,keys)
    p1.eval(y_pred,y_test)