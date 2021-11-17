from operator import index
import numpy as np
import lightgbm as lgb
from numpy.lib.function_base import append

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.misc import derivative
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

losses = []
acces = []


def focal_loss_lgb(y_pred, dtrain, alpha, gamma, num_class):
    a, g = alpha, gamma
    y_true = dtrain.label
    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype("int")]
    y_pred = y_pred.reshape(-1, num_class, order="F")
    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return (
            -(a * t + (1 - a) * (1 - t))
            * ((1 - (t * p + (1 - t) * (1 - p))) ** g)
            * (t * np.log(p) + (1 - t) * np.log(1 - p))
        )

    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order

    return grad.flatten("F"), hess.flatten("F")


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma, num_class):
    a, g = alpha, gamma
    y_true = dtrain.label
    y_true = np.eye(num_class)[y_true.astype("int")]
    y_pred = y_pred.reshape(-1, num_class, order="F")
    p = 1 / (1 + np.exp(-y_pred))
    loss = (
        -(a * y_true + (1 - a) * (1 - y_true))
        * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g)
        * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    )
    # a variant can be np.sum(loss)/num_class

    res = []
    for i in y_pred:
        i = list(i)
        res.append(i.index(max(i)))

    tr = []
    for i in y_true:
        i = list(i)
        tr.append(i.index(max(i)))
    acces.append(accuracy_score(tr, res))
    # global losses

    losses.append(np.mean(loss))

    return "focal_loss", np.mean(loss), False


if __name__ == "__main__":
    # very inadequate dataset as is perfectly balanced, but just to illustrate
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    lgbtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
    lgbeval = lgb.Dataset(X_val, y_val)
    focal_loss = lambda x, y: focal_loss_lgb(x, y, 0.25, 2.0, 10)
    eval_error = lambda x, y: focal_loss_lgb_eval_error(x, y, 0.25, 2.0, 10)
    params = {
        "learning_rate": 0.1,
        "num_boost_round": 1000,
        "num_class": 10,
        "verbose": -1,
    }
    model = lgb.train(
        params, lgbtrain, valid_sets=[lgbeval], fobj=focal_loss, feval=eval_error
    )
    y_pred = list(model.predict(X_val))
    res = []
    for i in y_pred:
        i = list(i)
        res.append(i.index(max(i)))
    y_pred = np.array(res)
    print(accuracy_score(y_val, y_pred))
    epochs = [i for i in range(1, len(losses) + 1)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(epochs, losses, label="loss", color="red")
    ax2.plot(epochs, acces, label="acc", color="blue")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend(loc=0)
    ax2.set_ylabel("acc")
    ax2.legend(loc=2)
    plt.show()
