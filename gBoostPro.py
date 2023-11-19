import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import export_graphviz
import subprocess

# Clasificación con árboles de Decisión

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importar el data set
dataset = pd.read_csv('trainV6.csv')
dataset['sismo'] = dataset['sismo'].replace([0, 1], [1,0])
X = dataset.iloc[:, 8:12]
y = dataset.iloc[:, 12]

# Divide los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )

# Crea el modelo de Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)

# Entrena el modelo
gb_model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = gb_model.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualiza la matriz de confusión
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues'  )

plt.title('Confusion Matrix G Boosting')
plt.show()

print(classification_report(y_test,y_pred))
tree_to_visualize = gb_model.estimators_[0, 0]
##Curva de roc
from sklearn.metrics import roc_curve, roc_auc_score
y_scores = gb_model.predict_proba(X_test)[:, 1]
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
plt.title('Curva ROC del Gradient Boosting')
plt.legend(loc='lower right')
plt.show()

# Exporta el árbol a formato .dot
dot_filename = 'tree.dot'
export_graphviz(tree_to_visualize, out_file=dot_filename, feature_names=X_train.columns, filled=True, rounded=True)

# Convierte el archivo .dot en una imagen (por ejemplo, PNG)
#image_filename = 'tree.png'
#subprocess.call(['dot', '-Tpng', dot_filename, '-o', image_filename])




