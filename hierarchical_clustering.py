import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings("ignore")

#### Data Preprocessing

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

print(X)

#### Plotting Dendrogram

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram for the Mall Customer Cluster")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

#### Training Using Hierarchical Clusturing

hc = AgglomerativeClustering(n_clusters=5, affinity= "euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

print(y_hc)


# Visualisig the clusters

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=75, c="red", label="Cluster-1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=75, c="blue", label="Cluster-2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=75, c="green", label="Cluster-3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=75, c="magenta", label="Cluster-4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=75, c="cyan", label="Cluster-5")

plt.title("Clusters of Customers of the mall")
plt.xlabel("Annual Income of the customers")
plt.ylabel("Spending Score of the customers")
plt.legend()
plt.show()