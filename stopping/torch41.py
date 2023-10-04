# 1 import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kmeans_pytorch import kmeans, kmeans_predict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2 load data
df = pd.read_csv(r"C:\Users\admin\stsble")
df.info()
print("-" * 30)
print(df)


# 3 word embedding
data = pd.get_dummies(df, columns=["Species"])
data["Species_Iris-setosa"] = data["Species_Iris-setosa"].astype("float32")
data["Species_Iris-versicolor"] = data["Species_Iris-versicolor"].astype("float32")
data["Species_Iris-virginica"] = data["Species_Iris-virginica"].astype("float32")
data.info()


# 4 split data
x, y = train_test_split(data, test_size=0.2, random_state=123)
print(type(x), "\n", x)
print(type(y), "\n", y)


# 5 scaling
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_scaled = ss.fit(data).transform(x)
y_scaled = ss.fit(data).transform(y)
print(type(x_scaled), "\n", x_scaled)

# 6 to tensor
x = torch.from_numpy(x_scaled).to(device)
y = torch.from_numpy(y_scaled).to(device)
print(type(x), "\n", x.size(), "\n", y.size(), x)

num_clusters = 3
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_cluster=num_clusters, distance='euclidean', device=device
)

print(cluster_ids_x)
print(cluster_centers)

cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)

print(cluster_ids_y)

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='viridis', marker='x')

plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c='white',
    alpha=0.6,
    edgecolors='biack',
    linewidths=2
)

plt.tight_layout()
plt.show()

