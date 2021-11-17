import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def data_process(l1):
    for i in l1.keys():
        total_len_half = (max(l1[i])-min(l1[i]))/2
        avg = (max(l1[i])+min(l1[i]))/2
        for j in range(len(l1[i])):
            l1[i][j] = int((l1[i][j]-avg)/total_len_half)
    return l1

def get_class(l1):
    temp = [i for i in list(set(l1))]
    return temp
def get_one_hot(map):
    list1 = [0]*len(map)
    list2 = [1]*len(map)
    res = []
    res1 = []
    for i in range(len(map)):
        t1 = list1.copy()
        t2 = list2.copy()
        t1[i] = 1
        t2[i] = 0
        res.append(t1)
        res1.append(t2)
    temp = {}
    temp1 = {}
    for i in range(len(map)):
        temp[str(map[i])] = res[i]
        temp1[str(map[i])] = res1[i]
    return temp,temp1


class Perceptron(object):
    def __init__(self,X,Y,lr,num_arg,num_class,one_hot,one_hot_reverse):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.rate = lr
        self.final_w = 0
        self.final_b = 0
        self.num_arg = num_arg
        self.num_class = num_class
        self.epoch = 10
        self.alpha = 0
        self.beta = 0
        self.one_hot = one_hot
        self.one_hot_reverse = one_hot_reverse

    def train(self):
        # w = np.zeros([self.num_class,self.num_arg])
        # b = np.zeros([self.num_class,1])
        # 训练1500次后收敛的权重
        w = np.array([[0.01,51.963,31.873,23.863,41.382],[0.756,87.284,156.14,29.437,134.535],[0.756,102.381,166.863,38.494,158.087]])
        b = np.array([[-62.682],[-248.84],[-266.204]])
        for i in range(self.epoch):
            loss = 0
            self.alpha = np.zeros([self.num_class,self.num_arg]) #(3,5)
            self.beta = np.zeros([self.num_class,1])
            for xi, yi in zip(self.X, self.Y):
                ai = np.sum(np.multiply([xi] * self.num_class, w), axis=1) + b
                y_predicti = np.exp(ai) / sum(np.exp(ai)) 
                y_i = self.one_hot[str(yi)]
                for j in range(self.num_class):
                    self.alpha[j] += np.multiply(sum(np.multiply(self.one_hot_reverse[str(float(j+1))], y_i)), xi)
                    self.beta[j] += sum(np.multiply(self.one_hot_reverse[str(float(j+1))],y_i))
            w -= self.alpha*self.rate
            b -= self.beta*self.rate
        self.final_w = w
        self.final_b = b
    
    def val(self,x_val,y_val,map_cls):
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        predict = []
        recall = 0
        for xi, yi in zip(x_val, y_val):
            ai = np.sum(np.multiply([xi] * self.num_class, self.final_w), axis=1) + self.final_b
            y_predicti = np.exp(ai) / sum(np.exp(ai))
            y_predicti = [map_cls[idx] for idx, i in enumerate(y_predicti) if max(i) == max(y_predicti[:,0])]
            predict.append(y_predicti)
            recall += 1 if y_predicti == yi else 0
        # import pdb; pdb.set_trace()
        print('score:', recall/len(y_val), '预测正确数：', recall)
        print('mean_absolute_error=',mean_absolute_error(predict,y_val))
        print('mean_squared_error=',mean_squared_error(predict,y_val))

    def test(self,x_test,map_cls):
        with open('perception.csv', 'w') as fp:
            for xi in x_test:
                ai = np.sum(np.multiply([xi] * 3, self.final_w), axis=1) + self.final_b
                y_predicti = np.exp(ai) / sum(np.exp(ai))
                y_predicti = [map_cls[idx] for idx, i in enumerate(y_predicti) if i == max(y_predicti[:,0])]
                fp.write(str(y_predicti)+'\n')

if __name__ == '__main__':
    data = pd.read_csv('Data.csv')
    data.iloc[:,:5] = data_process(data.iloc[:,:5])
    X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,:5],data['fetal_health'],test_size=0.1,random_state=666)
    class_map = get_class(data['fetal_health']) # [1,2,3]
    one_hot,reversed_one_hot = get_one_hot(class_map) # {'1.0': [1, 0, 0], '2.0': [0, 1, 0], '3.0': [0, 0, 1]}
    p1 = Perceptron(X_train,y_train,0.00001,len(X_train.keys()),len(class_map),one_hot,reversed_one_hot)
    p1.train()
    p1.val(X_test,y_test,class_map)