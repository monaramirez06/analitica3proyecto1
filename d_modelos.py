#lectura base modelos
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn import metrics
from sklearn.tree import plot_tree
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from a_funciones import cross_validation

ff = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/df_real.csv'
df_real = pd.read_csv((ff), sep= ',')

# Separacion de variables
y = df_real.attrition
X = df_real.drop(["attrition"], axis = 1)

#Separación de caracteristicas númericas y categóricas
numeric_columns=list(X.select_dtypes('float64').columns)

#Estandarización de variables númericas
pipeline=ColumnTransformer([('num',StandardScaler(),numeric_columns)], remainder='passthrough')
X1 = pipeline.fit_transform(X)
Xe = pd.DataFrame(X1, index = X.index, columns=X.columns)


############## MODELOS SIN SELECCION DE VARIABLES
#################################################


############regresion logistica
# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(Xe, y, test_size=0.2, random_state=12)

# creación del modelo
# Crea el modelo
regr_logbase0 = LogisticRegression(class_weight="balanced", random_state=12, max_iter=1000)

# Calibra el modelo
regr_logbase0.fit(X_train, y_train)

#Predicciones sobre conjunto de entrenamiento
y_pred = regr_logbase0.predict(X_train)
#Exactitud de modelo
print(f"Accuracy of the classifier is (train modelo 1): {accuracy_score(y_train, y_pred)}")

print("-------------------------------------------------------")

#Predicciones sobre el conjunto de test
y_hat = regr_logbase0.predict(X_test)
#Exactitud de modelo
print(f"Accuracy of the classifier is (test modelo 1): {accuracy_score(y_test, y_hat)}")

# Matriz de confusión  entrenamiento
fig = plt.figure(figsize=(20,10))
trainbase = confusion_matrix(y_train, y_pred, labels=regr_logbase0.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = trainbase, display_labels=regr_logbase0.classes_)
disp.plot(cmap='CMRmap_r')
plt.title('Matriz de confusión modelo base (Reg. logistica) (train)')
print(plt.show())

# Matriz de confusión test
fig = plt.figure(figsize=(20,10))
testbase = confusion_matrix(y_test, y_hat, labels=regr_logbase0.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = testbase, display_labels=regr_logbase0.classes_)
disp.plot(cmap='CMRmap_r')
plt.title('Matriz de confusión modelo base (Reg. logistica) (test)')
print(plt.show())

#MÉTRICASS DE TRAIN Y TEST
TP=trainbase[1,1]
FP=trainbase[0,1]
FN=trainbase[1,0]
TN=trainbase[0,0]
print(f"Accuracy of the classifier is (train) modelo base: {accuracy_score(y_train, y_pred)}")
print(f'Precisión (train): {TP/(TP+FP)}')
print(f'Recuperación (train): {TP/(TP+FN)}')
precision=TP/(TP+FP)
recall=TP/(TP+FN)
print(f'F1-score (train): {(2*precision*recall)/(precision+recall)}')
print(f'Especificidad (train): {TN/(FP+TN)}')
print("-----------------------------------")
TP=testbase[1,1]
FP=testbase[0,1]
FN=testbase[1,0]
TN=testbase[0,0]
print(f"Accuracy of the classifier is (test) modelo base: {accuracy_score(y_test, y_hat)}")
print(f'Precisión (test): {TP/(TP+FP)}')
print(f'Recuperación (test): {TP/(TP+FN)}')
precision=TP/(TP+FP)
recall=TP/(TP+FN)
print(f'F1-score (test): {(2*precision*recall)/(precision+recall)}')
print(f'Especificidad (test): {TN/(FP+TN)}')


#############Arbol de decision
# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(Xe, y, test_size=0.2, random_state=12)

# Creación del clasificador
clf0 = tree.DecisionTreeClassifier(
          criterion = 'gini',
          random_state=25,
          class_weight = 'balanced')

clf0.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, clf0.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, clf0.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, clf0.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, clf0.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusión
cm1= confusion_matrix(y_test, clf0.predict(X_test))
# Visualización de la matriz de confusión
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()




######bosque aleatorio
# Creación del clasificador
ranfor0 = RandomForestClassifier(
            criterion    = 'gini',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 25,
            class_weight = 'balanced'
         )
ranfor0.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, ranfor0.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, ranfor0.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, ranfor0.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, ranfor0.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusion
cm1= confusion_matrix(y_test, ranfor0.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

#######XGB
#Definición del modelo
XGBmodel0 = XGBClassifier(tree_method="hist",
                          enable_categorical=True
                        )
XGBmodel0.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, XGBmodel0.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, XGBmodel0.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, XGBmodel0.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, XGBmodel0.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusion
cm1= confusion_matrix(y_test, XGBmodel0.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

################ EVALUACION DE MODELOS
#########################################
# Evaluación regresión logística
log_model_base_result0 = cross_validation(regr_logbase0, Xe, y, 15)
print("Mean Training F1 Score Regresión logística: ", round(log_model_base_result0['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Regresión logística: ", round(log_model_base_result0['Mean Validation F1 Score'],4),
      "\nMean Training Precision Regresión logística: ", round(log_model_base_result0['Mean Training Precision'],4),
      "\nMean Validation Precision Regresión logística: ", round(log_model_base_result0['Mean Validation Precision'],4),
      "\nMean Training Recall Regresión logística: ", round(log_model_base_result0['Mean Training Recall'],4),
      "\nMean Validation Recall Regresión logística: ", round(log_model_base_result0['Mean Validation Recall'],4),
      "\nMean Training Accuracy Regresión logística: ", round(log_model_base_result0['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Regresión logística: ", round(log_model_base_result0['Mean Validation Accuracy'],4))
print("----------------------------------")
# Evaluación del árbol de decisión 1
model_clf0 = cross_validation(clf0, Xe, y, 15)
print("Mean Training F1 Score Árbol de decisión: ", round(model_clf0['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Árbol de decisión: ", round(model_clf0['Mean Validation F1 Score'],4),
      "\nMean Training Precision Árbol de decisión: ", round(model_clf0['Mean Training Precision'],4),
      "\nMean Validation Precision Árbol de decisión: ", round(model_clf0['Mean Validation Precision'],4),
      "\nMean Training Recall Árbol de decisión: ", round(model_clf0['Mean Training Recall'],4),
      "\nMean Validation Recall Árbol de decisión: ", round(model_clf0['Mean Validation Recall'],4),
      "\nMean Training Accuracy Árbol de decisión: ", round(model_clf0['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Árbol de decisión : ", round(model_clf0['Mean Validation Accuracy'],4))
print("----------------------------------")
# Evaluación del bosque aleatorio 1
model_ranfor0 = cross_validation(ranfor0, Xe, y, 15)
print("Mean Training F1 Score Bosque aleatorio: ", round(model_ranfor0['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Bosque aleatorio: ", round(model_ranfor0['Mean Validation F1 Score'],4),
      "\nMean Training Precision Bosque aleatorio: ", round(model_ranfor0['Mean Training Precision'],4),
      "\nMean Validation Precision Bosque aleatorio: ", round(model_ranfor0['Mean Validation Precision'],4),
      "\nMean Training Recall Bosque aleatorio: ", round(model_ranfor0['Mean Training Recall'],4),
      "\nMean Validation Recall Bosque aleatorio: ", round(model_ranfor0['Mean Validation Recall'],4),
      "\nMean Training Accuracy Bosque aleatorio: ", round(model_ranfor0['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Bosque aleatorio: ", round(model_ranfor0['Mean Validation Accuracy'],4))
print("----------------------------------")
## Evaluación del XGB 1
model_XGB0 = cross_validation(XGBmodel0, Xe, y, 15)
print("Mean Training F1 Score XGB: ", round(model_XGB0['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score XGB: ", round(model_XGB0['Mean Validation F1 Score'],4),
      "\nMean Training Precision XGB: ", round(model_XGB0['Mean Training Precision'],4),
      "\nMean Validation Precision XGB: ", round(model_XGB0['Mean Validation Precision'],4),
      "\nMean Training Recall XGB: ", round(model_XGB0['Mean Training Recall'],4),
      "\nMean Validation Recall XGB: ", round(model_XGB0['Mean Validation Recall'],4),
      "\nMean Training Accuracy XGB: ", round(model_XGB0['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy XGB: ", round(model_XGB0['Mean Validation Accuracy'],4))

metricas = pd.DataFrame()
metricas.insert(0,'Accuracy RL', log_model_base_result0['Validation Accuracy scores'])
metricas.insert(1, 'Precision RL', log_model_base_result0['Validation Precision scores'])
metricas.insert(2, 'Recall RL', log_model_base_result0['Validation Recall scores'])
metricas.insert(3, 'F1 RL', log_model_base_result0['Validation F1 scores'])
metricas.insert(4,'Accuracy AD', model_clf0['Validation Accuracy scores'])
metricas.insert(5, 'Precision AD', model_clf0['Validation Precision scores'])
metricas.insert(6, 'Recall AD', model_clf0['Validation Recall scores'])
metricas.insert(7, 'F1 AD', model_clf0['Validation F1 scores'])
metricas.insert(8,'Accuracy BA', model_ranfor0['Validation Accuracy scores'])
metricas.insert(9, 'Precision BA', model_ranfor0['Validation Precision scores'])
metricas.insert(10, 'Recall BA', model_ranfor0['Validation Recall scores'])
metricas.insert(11, 'F1 BA', model_ranfor0['Validation F1 scores'])
metricas.insert(12,'Accuracy XGB', model_XGB0['Validation Accuracy scores'])
metricas.insert(13, 'Precision XGB', model_XGB0['Validation Precision scores'])
metricas.insert(14, 'Recall XGB', model_XGB0['Validation Recall scores'])
metricas.insert(15, 'F1 XGB', model_XGB0['Validation F1 scores'])


plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
metricas.boxplot(column=["Accuracy RL", "Accuracy AD", "Accuracy BA", "Accuracy XGB"], grid=False)

plt.subplot(2,2,2)
metricas.boxplot(column=["Precision RL", "Precision AD", "Precision BA", "Precision XGB"], grid=False)

plt.subplot(2,2,3)
metricas.boxplot(column=["Recall RL", "Recall AD", "Recall BA", "Recall XGB"], grid=False)

plt.subplot(2,2,4)
metricas.boxplot(column=["F1 RL", "F1 AD", "F1 BA", "F1 XGB"], grid=False)
plt.show()