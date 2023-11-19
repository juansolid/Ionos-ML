
# Clasificación con árboles de Decisión


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('trainV6.csv')
dataset['sismo'] = dataset['sismo'].replace([0, 1], [1,0])
X = dataset.iloc[:, 8:12]
y = dataset.iloc[:, 12]


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0) #40% de datos para prueba , 60% entrenamiento se realizó asi ya que el adataset es de gran dimension


# Ajustar el clasificador de Árbol de Decisión en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=6,criterion = "gini", random_state = 0,class_weight='balanced')#max depth = 6 es la profundidad de los datos y como escalara el algoritmo de regresion
classifier.fit(X_train, y_train)
datas = classifier.feature_importances_
print(datas)

y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#precicion del algoritmo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print(precision)
#obtener acuraccy (precision del arbol de decision)
accuracy = classifier.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
##libreria seaborn 
import seaborn as sns
#reporte de clasificacion y matriz de confusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#pintas la mattriz de confusion

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot = True, fmt='g', cmap ='Blues')
plt.show()
#guarda el arbol de clasificación en archivo .dot
print(classification_report(y_test,y_pred))
from sklearn.tree import export_graphviz
export_graphviz(classifier, out_file='arbolClasificacion.dot',filled = True, feature_names = list(X.columns))

from sklearn.tree import plot_tree
#pinta el árbol de decision
plot_tree(classifier, filled=True)
plt.title("Arbol de decision  efectos sismo magneticos")
plt.show()
##Curva de roc
from sklearn.metrics import roc_curve, roc_auc_score
y_scores = classifier.predict_proba(X_test)[:, 1]
# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calcular el área bajo la curva ROC (AUC-ROC)
roc_auc = roc_auc_score(y_test, y_scores)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC del Árbol de Decisión')
plt.legend(loc='lower right')
plt.show()



import pickle
#crear el modelo
entrenamiento = open("entrenamiento.pickle","wb")
#guardar entrenamiento
pickle.dump(classifier,entrenamiento)
entrenamiento.close()
#arreglo para realizar la prediccion se pueden crear multiples arreglor para guardar la prediccion
#datos = [[15,330,11.55,30]]
#datos2 =[[60,8000,1,10000,16,1.25]]
#carga = open("entrenamiento.pkl","rb")
#carga2= open("entrenamiento.pickle","rb")
#datas = pickle.load(carga)
#datas2 = pickle.load(carga2)
#prediccion = datas.predict(datos)
#prediccion2 = datas2.predict(datos)
#print(prediccion)
#print(prediccion2)




