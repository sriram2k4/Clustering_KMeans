import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("Mall_Customers.csv")
dataset.drop("CustomerID", axis=1, inplace=True)
print(dataset)

#### Feature - Annual Income, Spending Score

X = dataset.iloc[:,[2,3]].values

# print(X)

#### Finding the optimal number of cluster by elbow method

wcss = [] ## Within cluster sum of squares

for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=46)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#### Visualizing plot between the wcss and the number of clusters ( 5 )

# print(wcss)

# plt.plot(range(1,12), wcss)
# plt.xlabel("Number of Clusters")
# plt.ylabel("WCSS")
# plt.title("The Elbow Method")
# plt.show()

#### Training the model

kmeans = KMeans(n_clusters=5, init="k-means++", random_state=46)
y_kmeans = kmeans.fit_predict(X)

# Visualisig the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=75, c="red", label="Cluster-1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=75, c="blue", label="Cluster-2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=75, c="green", label="Cluster-3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=75, c="magenta", label="Cluster-4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=75, c="cyan", label="Cluster-5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c="yellow", label="Centroid")
plt.title("Clusters of Customers of the mall")
plt.xlabel("Annual Income of the customers")
plt.ylabel("Spending Score of the customers")
plt.legend()
plt.show()
