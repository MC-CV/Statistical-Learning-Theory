from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# 载入brest_cancer数据集，并划分为训练集和测试集
X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# 决策树分类
clf = DecisionTreeClassifier(random_state=0)
# 返回剪枝的过程中每一个ccp_alphas和impurities的值
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
# 遍历每个ccp_alpha并作图表
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# 尝试每一个ccp_alpha在训练和测试中的效果
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
temp = DecisionTreeClassifier(random_state=0, ccp_alpha=0)
temp.fit(X_train,y_train)
print('if ccp_alphas = 0:')
print('The precision of train dataset is:',temp.score(X_train,y_train))
print('The precision of test dataset is:',temp.score(X_test,y_test))
temp1 = []
alpha = []
for ccp_alpha in ccp_alphas:
    temp1 = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    temp1.fit(X_train, y_train)
    if max(test_scores)==temp1.score(X_test,y_test):
        alpha.append(ccp_alpha)
        num = temp1.score(X_train,y_train)
print('The max precision of test dataset is:',max(test_scores),'\n','and at this time the ccp_alphas is:',min(alpha))
print('and the precision of train dataset is:',num)

ax.legend()
plt.show()