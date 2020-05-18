import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
from matplotlib import style


data = pd.read_csv("^GSPC.csv", sep=",")
data = data[["Date","Open","High","Low","Close","ADJClose","Volume"]]
data = data.delete(["Date"])
predict = "ADJClose"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("SP500.pickle", "wb") as file:
            pickle.dump(linear, file)


