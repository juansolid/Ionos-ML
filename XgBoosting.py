# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:33:10 2023

@author: Juan Antonio Murillo
"""

import pandas as pd
import numpy as np

df = pd.read_csv ('ad_datas.csv')
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
X = df.drop("clicked", axis=1)
y = df["clicked"]

X['copy'] = X['copy'].astype("category")
X['audience']= X['audience'].astype("category")
X['keywords'] = X['keywords'].astype("category")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.info()
X_test.info()
model = XGBClassifier(tree_method="hist",enable_categorical=True)
model.fit(X_train, y_train)
model.feature_importances_
#graph = XGBClassifier.to_graphviz(model, num_trees=1)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
#reporte de clasificacion y matriz de confusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#pintas la mattriz de confusion

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot = True, fmt='g', cmap ='Blues')
plt.show()
#guarda el arbol de clasificación en archivo .dot
print(classification_report(y_test,y_pred))