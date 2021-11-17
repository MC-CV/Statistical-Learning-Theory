import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
score = []
for i in range(1,10):   
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    score.append(knn.score(X_test,y_test))
k = [i for i in range(1,10)]
plt.plot(k,score,label="score", color='b')
plt.xlabel("k")
plt.ylabel("score")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)
svm_pred = clf.predict(X_test)
print('KNN_r2_score:',r2_score(y_test,y_pred))
print('KNN_mean_squared_error',mean_squared_error(y_test,y_pred))
print('KNN_mean_absolute_error',mean_absolute_error(y_test,y_pred))

print('SVM_r2_score:',r2_score(y_test,svm_pred))
print('SVM_mean_squared_error',mean_squared_error(y_test,svm_pred))
print('SVM_mean_absolute_error',mean_absolute_error(y_test,svm_pred))