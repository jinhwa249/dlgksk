from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

= dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values
X = train, X_test, y_train, y_test = train test _split(x, y, test_size=0.20)
s = StandardScaler()
Xtest - s.fit transform(X_test)

with open("pytorch/data/knn.picke', 'br') as f:

knn= pickle.load(f)

y_pred = knn.predict(X_test)

