from numpy import *
import matplotlib.pyplot as plt
X_data = loadtxt("data/X_train.txt")
print (X_data.shape)
X = X_data
y_data = loadtxt("data/y_train.txt", dtype = int)
print (y_data.shape)
y = y_data
from sklearn.utils import shuffle
X_new, y_new = shuffle(X, y)

X_train = X_new[:1000]
y_train = y_new[:1000]
X_test = X_new[1000:]
y_test = y_new[1000:]
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
y_pred_test = logreg.predict(X_test)
print ("Accuracy on test set:", logreg.score(X_test, y_test))
