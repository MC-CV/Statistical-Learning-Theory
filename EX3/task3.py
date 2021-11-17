import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    digits = load_digits()
    scaler = StandardScaler().fit(digits.data)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, train_size=0.8, random_state=888,
    )
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    estimators = [100, 200, 400]
    max_features = ["auto", "sqrt", "log2"]
    nodes = [10, 50, 100]

    for features in max_features:
        for es in estimators:
            for node in nodes:
                rf = RandomForestClassifier(
                    n_estimators=es,
                    max_features=features,
                    oob_score=True,
                    max_leaf_nodes=node,
                )
                rf.fit(X_train, y_train)
                pre_test = rf.predict(X_test)

                acc_score = accuracy_score(y_test, pre_test)
                pre_score = precision_score(y_test, pre_test, average="micro")

                print(
                    "n_estimators = %f,max_features = %s,max_leaf_nodes = %f"
                    % (es, features, node)
                )
                print("accuracy:%f,pre_score:%f" % (acc_score, pre_score))
                # print(rf.feature_importances_)

# rf = RandomForestClassifier(
#     n_estimators=400, max_features="log2", oob_score=True, max_leaf_nodes=100
# )
# rf.fit(X_train, y_train)
# pre_test = rf.predict(X_test)

# acc_score = accuracy_score(y_test, pre_test)
# pre_score = precision_score(y_test, pre_test)

# # print('n_estimators = %f,max_features = %s,max_leaf_nodes = %f'%(es,features,node))
# print("accuracy:%f,auc_score:%f,pre_score:%f" % (acc_score, pre_score))
# print(rf.feature_importances_)
# feature = X_train.keys()
# feature_importance = rf.feature_importances_
# feature_df = pd.DataFrame({"Features": feature, "Importance": feature_importance})
# feature_df.sort_values("Importance", inplace=True, ascending=False)
# print(feature_df)

