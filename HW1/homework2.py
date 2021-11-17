from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
import random
import itertools 
# path = r'D:\Users\53263\courses\tongji'
corpos = pd.DataFrame(columns=['text','kind'])
def load_data():
    # csv.field_size_limit(500 * 1024 * 1024)
    temp_fake = []
    temp_true = []
    with open('clean_fake.txt','r',encoding='utf-8') as f_fake:
        
        temp = 0
        for i in f_fake.readlines():
            # temp_fake.append(i[4])
            temp += 1
            corpos.loc[len(corpos)] = [i,'fake']
            if temp == 1298: break
    with open('clean_real.txt','r',encoding='utf-8') as f_true:
        
        temp = 0
        for i in f_true.readlines():
            temp += 1
            # temp_true.append(i[1])
            corpos.loc[len(corpos)] = [i,'true']
            if temp == 1968: break
    # with open('fake.txt','w',encoding='utf-8') as r_fake:
    #     for i in temp_fake:
    #         r_fake.write(i+'\n')
    # with open('true.txt','w',encoding='utf-8') as r_true:
    #     for i in temp_true:
    #         r_true.write(i+'\n')
    

def main():
    load_data()
    # print(corpos)
    cv = CountVectorizer()
    countvector = cv.fit_transform(corpos.iloc[:,0]).toarray()

    # 转换数字
    kind = np.unique(corpos['kind'].values)
    nkind = np.zeros(len(countvector)) 
    for i in range(len(kind)):
        index = corpos[corpos['kind']==kind[i]].index
        nkind[index] = i+1
    pca = PCA(n_components=2)
    newvector = pca.fit_transform(countvector)
    plt.figure()
    for i,c,m in zip(range(len(kind)),['r','b'],['o','^']):
        index = corpos[corpos['kind']==kind[i]].index
        x = newvector[index,0]
        y = newvector[index,1]
        plt.scatter(x,y,c=c,marker=m,label=kind[i])
    plt.legend()
    plt.xlim(-1,3)
    plt.ylim(-2.5,5)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.show()
    # 划分数据集
    index = np.arange(0,len(countvector),1)
    random.shuffle(index)
    index_train,index_val,index_test = index[:int(len(countvector)*0.8)],index[int(len(countvector)*0.8):int(len(countvector)*0.9)],\
        index[int(len(countvector)*0.9):]
    X_train = countvector[index_train]
    y_train = corpos.iloc[index_train,1]

    X_val = countvector[index_val]
    y_val = corpos.iloc[index_val,1]

    X_test = countvector[index_test]
    y_test = corpos.iloc[index_test,1]

    # knn分类
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train,y_train)
    # 精度计算
    print('The accuracy of val is：',knn.score(X_val,y_val))
    y_pred = knn.predict(X_test)
    print('The accuracy of test is：',knn.score(X_test,y_test))
    acc = np.mean(y_pred == y_test)

    # 计算混淆矩阵
    knn_confusion = confusion_matrix(y_test,y_pred)
    plt.imshow(knn_confusion,interpolation='nearest',cmap=plt.cm.Oranges) 
    plt.xlabel('y_pred')
    plt.ylabel('y_True')
    tick_marks = np.arange(len(kind))
    plt.xticks(tick_marks,kind,rotation=90)
    plt.yticks(tick_marks,kind)
    plt.colorbar()
    plt.title('confusion_matrix')
    for i,j in itertools.product(range(len(knn_confusion)),range(len(knn_confusion))):
        plt.text(i,j,knn_confusion[j,i],
                horizontalalignment="center")
    plt.show()

    
if __name__ == '__main__':
    main()