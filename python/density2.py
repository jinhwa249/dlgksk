import numba as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

X = pd.read_csv('python/credit caredit.csv')

X = X.drop('CUST_ID', axis=1, inplace=True)
X.fillna(method(method='ffill', inplace=True))
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized - pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())

db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
labels1 = db.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels1]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[0])
g= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[1])
b= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[2])
c= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[3])
y= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[4])
m= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[5])
k= plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[6])

plt.figure(figsize=(9.9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r, g, b,c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4','Label 5', 'Label -1'),
            scatterpoints=1,
            loc='upper left',
            ncol=3,
            fontsize=8)
plt.show()