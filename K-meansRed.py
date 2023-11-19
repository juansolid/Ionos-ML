# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 01:31:37 2023

@author: Juan Antonio Murillo
"""

# K-Means

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Cargamos los datos con pandas
df = pd.read_csv("ad_data6.csv")
#para trabajar con variables categoricas debemos hacer la transformacion objeto a categorias
df['copy']= df['copy'].astype('category')
df['audience']= df['audience'].astype('category')
df['keywords']= df['keywords'].astype('category')
df['store']= df['store'].astype('category')
df['copy'] = df['copy'].cat.codes
df['audience']= df['audience'].cat.codes
df['store']= df['store'].cat.codes
df['keywords'] = df['keywords'].cat.codes

X= df.iloc[:, [1,2]].values

# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()


# Aplicar el método de k-means para segmentar el data set
kmeans = KMeans(n_clusters = 3, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# Visualización de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = "red", label = "C1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = "blue", label = "C2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = "green", label = "C3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = "cyan", label = "C4")
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = "magenta", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = "yellow", label = "Baricentros")
plt.title("Cluster clima espacial")
plt.xlabel('Temperatura °k')
plt.ylabel('Velocidad km/s')
plt.legend()
plt.show()
