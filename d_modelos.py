# Importar librearias
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from a_funciones import cross_validation
from a_funciones import sel_variables
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import joblib

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

#### Modelos con selección de variables por medio de SelectFromModel
######################################################################



# Función para obtener las variables importantes de cada modelo
    # Definición de variables
modelos = list([regr_logbase0,clf0, ranfor0, XGBmodel0])
var_names = sel_variables(modelos,Xe,y,threshold="2.5*mean")
Xenew=Xe[var_names] ### Matriz con variables seleccionadas
Xenew.info()
Xenew.to_csv("Xenew.csv")
y.to_csv("y.csv")

###### Regresión logística  #########

# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(Xenew, y, test_size=0.2, random_state=12)

# creación del modelo
# Crea el modelo
regr_logbase1 = LogisticRegression(class_weight="balanced", random_state=12, max_iter=1000)

# Calibra el modelo
regr_logbase1.fit(X_train, y_train)

#Predicciones sobre conjunto de entrenamiento
y_pred = regr_logbase1.predict(X_train)
#Exactitud de modelo
print(f"Accuracy of the classifier is (train modelo 1): {accuracy_score(y_train, y_pred)}")

print("-------------------------------------------------------")

#Predicciones sobre el conjunto de test
y_hat = regr_logbase1.predict(X_test)
#Exactitud de modelo
print(f"Accuracy of the classifier is (test modelo 1): {accuracy_score(y_test, y_hat)}")

# Matriz de confusión datos de entrenamiento
fig = plt.figure(figsize=(20,10))
trainbase = confusion_matrix(y_train, y_pred, labels=regr_logbase1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = trainbase, display_labels=regr_logbase1.classes_)
disp.plot(cmap='CMRmap_r')
plt.title('Matriz de confusión modelo base(train)')
print(plt.show())

# Matriz de confusión test
fig = plt.figure(figsize=(20,10))
testbase = confusion_matrix(y_test, y_hat, labels=regr_logbase1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = testbase, display_labels=regr_logbase1.classes_)
disp.plot(cmap='CMRmap_r')
plt.title('Matriz de confusión modelo base(test)')
print(plt.show())

# MÉTRICASS DE TRAIN Y TEST
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

##################  Árbol de decisión   ###############
###################################################

# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(Xenew, y, test_size=0.2, random_state=12)

# Creación del clasificador
clf1 = tree.DecisionTreeClassifier(
          criterion = 'gini',
          random_state=25,
          class_weight = 'balanced')

clf1.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, clf1.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, clf1.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, clf1.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, clf1.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusión
cm1= confusion_matrix(y_test, clf1.predict(X_test))
# Visualización de la matriz de confusión
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

############################ Bosque Aleatorio ################
##############################################################

# Creación del clasificador
ranfor1 = RandomForestClassifier(
            criterion    = 'gini',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 25,
            class_weight = 'balanced'
         )
ranfor1.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, ranfor1.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, ranfor1.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, ranfor1.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, ranfor1.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusion
cm1= confusion_matrix(y_test, ranfor1.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

########################### XGB #######################
########################################################

#Definición del modelo
XGBmodel1 = XGBClassifier(tree_method="hist",
                          enable_categorical=True
                        )
XGBmodel1.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, XGBmodel1.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, XGBmodel1.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, XGBmodel1.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, XGBmodel1.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusion
cm1= confusion_matrix(y_test, XGBmodel1.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

######################## Cross Validation########################
##################################################################

##### Evaluación regresión logística
####################################
log_model_base_result1 = cross_validation(regr_logbase1, Xenew, y, 15)
print("Mean Training F1 Score Regresión logística: ", round(log_model_base_result1['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Regresión logística: ", round(log_model_base_result1['Mean Validation F1 Score'],4),
      "\nMean Training Precision Regresión logística: ", round(log_model_base_result1['Mean Training Precision'],4),
      "\nMean Validation Precision Regresión logística: ", round(log_model_base_result1['Mean Validation Precision'],4),
      "\nMean Training Recall Regresión logística: ", round(log_model_base_result1['Mean Training Recall'],4),
      "\nMean Validation Recall Regresión logística: ", round(log_model_base_result1['Mean Validation Recall'],4),
      "\nMean Training Accuracy Regresión logística: ", round(log_model_base_result1['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Regresión logística: ", round(log_model_base_result1['Mean Validation Accuracy'],4))
print("----------------------------------")


#### Evaluación del árbol de decisión 1
#######################################
model_clf1 = cross_validation(clf1, Xenew, y, 15)
print("Mean Training F1 Score Árbol de decisión 1: ", round(model_clf1['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Árbol de decisión 1: ", round(model_clf1['Mean Validation F1 Score'],4),
      "\nMean Training Precision Árbol de decisión 1: ", round(model_clf1['Mean Training Precision'],4),
      "\nMean Validation Precision Árbol de decisión 1: ", round(model_clf1['Mean Validation Precision'],4),
      "\nMean Training Recall Árbol de decisión 1: ", round(model_clf1['Mean Training Recall'],4),
      "\nMean Validation Recall Árbol de decisión 1: ", round(model_clf1['Mean Validation Recall'],4),
      "\nMean Training Accuracy Árbol de decisión 1: ", round(model_clf1['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Árbol de decisión 1: ", round(model_clf1['Mean Validation Accuracy'],4))
print("----------------------------------")


#### Evaluación del bosque aleatorio 1
######################################
model_ranfor1 = cross_validation(ranfor1, Xenew, y, 15)
print("Mean Training F1 Score Bosque aleatorio 1: ", round(model_ranfor1['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Bosque aleatorio 1: ", round(model_ranfor1['Mean Validation F1 Score'],4),
      "\nMean Training Precision Bosque aleatorio 1: ", round(model_ranfor1['Mean Training Precision'],4),
      "\nMean Validation Precision Bosque aleatorio 1: ", round(model_ranfor1['Mean Validation Precision'],4),
      "\nMean Training Recall Bosque aleatorio 1: ", round(model_ranfor1['Mean Training Recall'],4),
      "\nMean Validation Recall Bosque aleatorio 1: ", round(model_ranfor1['Mean Validation Recall'],4),
      "\nMean Training Accuracy Bosque aleatorio 1: ", round(model_ranfor1['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Bosque aleatorio 1: ", round(model_ranfor1['Mean Validation Accuracy'],4))
print("----------------------------------")


#### Evaluación del XGB 1
#########################
model_XGB1 = cross_validation(XGBmodel1, Xenew, y, 15)
print("Mean Training F1 Score XGB 1: ", round(model_XGB1['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score XGB 1: ", round(model_XGB1['Mean Validation F1 Score'],4),
      "\nMean Training Precision XGB 1: ", round(model_XGB1['Mean Training Precision'],4),
      "\nMean Validation Precision XGB 1: ", round(model_XGB1['Mean Validation Precision'],4),
      "\nMean Training Recall XGB 1: ", round(model_XGB1['Mean Training Recall'],4),
      "\nMean Validation Recall XGB 1: ", round(model_XGB1['Mean Validation Recall'],4),
      "\nMean Training Accuracy XGB 1: ", round(model_XGB1['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy XGB 1: ", round(model_XGB1['Mean Validation Accuracy'],4))


metricas = pd.DataFrame()
metricas.insert(0,'Accuracy RL', log_model_base_result1['Validation Accuracy scores'])
metricas.insert(1, 'Precision RL', log_model_base_result1['Validation Precision scores'])
metricas.insert(2, 'Recall RL', log_model_base_result1['Validation Recall scores'])
metricas.insert(3, 'F1 RL', log_model_base_result1['Validation F1 scores'])
metricas.insert(4,'Accuracy AD', model_clf1['Validation Accuracy scores'])
metricas.insert(5, 'Precision AD', model_clf1['Validation Precision scores'])
metricas.insert(6, 'Recall AD', model_clf1['Validation Recall scores'])
metricas.insert(7, 'F1 AD', model_clf1['Validation F1 scores'])
metricas.insert(8,'Accuracy BA', model_ranfor1['Validation Accuracy scores'])
metricas.insert(9, 'Precision BA', model_ranfor1['Validation Precision scores'])
metricas.insert(10, 'Recall BA', model_ranfor1['Validation Recall scores'])
metricas.insert(11, 'F1 BA', model_ranfor1['Validation F1 scores'])
metricas.insert(12,'Accuracy XGB', model_XGB1['Validation Accuracy scores'])
metricas.insert(13, 'Precision XGB', model_XGB1['Validation Precision scores'])
metricas.insert(14, 'Recall XGB', model_XGB1['Validation Recall scores'])
metricas.insert(15, 'F1 XGB', model_XGB1['Validation F1 scores'])

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

#######SELECION VARIABLE CON LASSO
#Selector de variables con Lasso
sel_lasso = SelectFromModel(Lasso(alpha=0.01), max_features=39)
sel_lasso.fit(Xe, y)

#Imprimir coeficientes del estimador
print(sel_lasso.estimator_.coef_)

#Obtener variables seleccionadas
X_new = sel_lasso.get_support()

#Filtrar X_train y Y_train para eliminar variables con coeficiente 0
Xenew2 = Xe.iloc[:,X_new]
Xenew2.info()

#####REGRESION LOGISTICA
# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(Xenew2, y, test_size=0.2, random_state=12)

# creación del modelo
# Crea el modelo
regr_logbase2 = LogisticRegression(class_weight="balanced", random_state=12, max_iter=1000)

# Calibra el modelo
regr_logbase2.fit(X_train, y_train)

#Predicciones sobre conjunto de entrenamiento
y_pred = regr_logbase2.predict(X_train)
#Exactitud de modelo
print(f"Accuracy of the classifier is (train modelo 1): {accuracy_score(y_train, y_pred)}")

print("-------------------------------------------------------")

#Predicciones sobre el conjunto de test
y_hat = regr_logbase2.predict(X_test)
#Exactitud de modelo
print(f"Accuracy of the classifier is (test modelo 1): {accuracy_score(y_test, y_hat)}")

# Matriz de confusión datos de entrenamiento
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
fig = plt.figure(figsize=(20,10))
trainbase = confusion_matrix(y_train, y_pred, labels=regr_logbase2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = trainbase, display_labels=regr_logbase2.classes_)
disp.plot(cmap='CMRmap_r')
plt.title('Matriz de confusión modelo base(train)')
print(plt.show())

# Matriz de confusión test
fig = plt.figure(figsize=(20,10))
testbase = confusion_matrix(y_test, y_hat, labels=regr_logbase2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = testbase, display_labels=regr_logbase2.classes_)
disp.plot(cmap='CMRmap_r')
plt.title('Matriz de confusión modelo base(test)')
print(plt.show())

#MÉTRICASS DE TRAIN Y TEST
TP=trainbase[1,1]
FP=trainbase[0,1]
FN=trainbase[1,0]
TN=trainbase[0,0]
print(f"Accuracy of the classifier is (train) modelo: {accuracy_score(y_train, y_pred)}")
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
print(f"Accuracy of the classifier is (test) modelo: {accuracy_score(y_test, y_hat)}")
print(f'Precisión (test): {TP/(TP+FP)}')
print(f'Recuperación (test): {TP/(TP+FN)}')
precision=TP/(TP+FP)
recall=TP/(TP+FN)
print(f'F1-score (test): {(2*precision*recall)/(precision+recall)}')
print(f'Especificidad (test): {TN/(FP+TN)}')


###ARBOL DE DECISION
# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(Xenew2, y, test_size=0.2, random_state=12)

# Creación del clasificador
clf2 = tree.DecisionTreeClassifier(
          criterion = 'gini',
          random_state=25,
          class_weight = 'balanced')

clf2.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, clf2.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, clf2.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, clf2.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, clf2.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusión
cm1= confusion_matrix(y_test, clf2.predict(X_test))
# Visualización de la matriz de confusión
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

###BOSQUE ALEATORIO
# Creación del clasificador
ranfor2 = RandomForestClassifier(
            criterion    = 'gini',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 25,
            class_weight = 'balanced'
         )
ranfor2.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, ranfor2.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, ranfor2.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, ranfor2.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, ranfor2.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusion
cm1= confusion_matrix(y_test, ranfor2.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()


####XGB
#Definición del modelo
XGBmodel2 = XGBClassifier(tree_method="hist",
                          enable_categorical=True
                        )
XGBmodel2.fit(X_train, y_train)

# Métricas de desempeño
print ("Train - Accuracy :", metrics.accuracy_score(y_train, XGBmodel2.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, XGBmodel2.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, XGBmodel2.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, XGBmodel2.predict(X_test)))
print('-----------------------------------------------------------------------')

# Matriz de confusion
cm1= confusion_matrix(y_test, XGBmodel2.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()


####VALIDACION CRUZADA CON EL METODO LASSO
# Evaluación regresión logística
log_model_base_result2 = cross_validation(regr_logbase2, Xenew2, y, 15)
print("Mean Training F1 Score Regresión logística: ", round(log_model_base_result2['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Regresión logística: ", round(log_model_base_result2['Mean Validation F1 Score'],4),
      "\nMean Training Precision Regresión logística: ", round(log_model_base_result2['Mean Training Precision'],4),
      "\nMean Validation Precision Regresión logística: ", round(log_model_base_result2['Mean Validation Precision'],4),
      "\nMean Training Recall Regresión logística: ", round(log_model_base_result2['Mean Training Recall'],4),
      "\nMean Validation Recall Regresión logística: ", round(log_model_base_result2['Mean Validation Recall'],4),
      "\nMean Training Accuracy Regresión logística: ", round(log_model_base_result2['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Regresión logística: ", round(log_model_base_result2['Mean Validation Accuracy'],4))
print("----------------------------------")
# Evaluación del árbol de decisión 2
model_clf2 = cross_validation(clf2, Xenew2, y, 15)
print("Mean Training F1 Score Árbol de decisión 2: ", round(model_clf2['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Árbol de decisión 2: ", round(model_clf2['Mean Validation F1 Score'],4),
      "\nMean Training Precision Árbol de decisión 2: ", round(model_clf2['Mean Training Precision'],4),
      "\nMean Validation Precision Árbol de decisión 2: ", round(model_clf2['Mean Validation Precision'],4),
      "\nMean Training Recall Árbol de decisión 2: ", round(model_clf2['Mean Training Recall'],4),
      "\nMean Validation Recall Árbol de decisión 2: ", round(model_clf2['Mean Validation Recall'],4),
      "\nMean Training Accuracy Árbol de decisión 2: ", round(model_clf2['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Árbol de decisión 2: ", round(model_clf2['Mean Validation Accuracy'],4))
print("----------------------------------")
# Evaluación del bosque aleatorio 2
model_ranfor2 = cross_validation(ranfor2, Xenew2, y, 15)
print("Mean Training F1 Score Bosque aleatorio 2: ", round(model_ranfor2['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Bosque aleatorio 2: ", round(model_ranfor2['Mean Validation F1 Score'],4),
      "\nMean Training Precision Bosque aleatorio 2: ", round(model_ranfor2['Mean Training Precision'],4),
      "\nMean Validation Precision Bosque aleatorio 2: ", round(model_ranfor2['Mean Validation Precision'],4),
      "\nMean Training Recall Bosque aleatorio 2: ", round(model_ranfor2['Mean Training Recall'],4),
      "\nMean Validation Recall Bosque aleatorio 2: ", round(model_ranfor2['Mean Validation Recall'],4),
      "\nMean Training Accuracy Bosque aleatorio 2: ", round(model_ranfor2['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Bosque aleatorio 2: ", round(model_ranfor2['Mean Validation Accuracy'],4))
print("----------------------------------")
## Evaluación del XGB 2
model_XGB2 = cross_validation(XGBmodel2, Xenew2, y, 15)
print("Mean Training F1 Score XGB 2: ", round(model_XGB2['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score XGB 2: ", round(model_XGB2['Mean Validation F1 Score'],4),
      "\nMean Training Precision XGB 2: ", round(model_XGB2['Mean Training Precision'],4),
      "\nMean Validation Precision XGB 2: ", round(model_XGB2['Mean Validation Precision'],4),
      "\nMean Training Recall XGB 2: ", round(model_XGB2['Mean Training Recall'],4),
      "\nMean Validation Recall XGB 2: ", round(model_XGB2['Mean Validation Recall'],4),
      "\nMean Training Accuracy XGB 2: ", round(model_XGB2['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy XGB 2: ", round(model_XGB2['Mean Validation Accuracy'],4))

metricas = pd.DataFrame()
metricas.insert(0,'Accuracy RL', log_model_base_result2['Validation Accuracy scores'])
metricas.insert(1, 'Precision RL', log_model_base_result2['Validation Precision scores'])
metricas.insert(2, 'Recall RL', log_model_base_result2['Validation Recall scores'])
metricas.insert(3, 'F1 RL', log_model_base_result2['Validation F1 scores'])
metricas.insert(4,'Accuracy AD', model_clf2['Validation Accuracy scores'])
metricas.insert(5, 'Precision AD', model_clf2['Validation Precision scores'])
metricas.insert(6, 'Recall AD', model_clf2['Validation Recall scores'])
metricas.insert(7, 'F1 AD', model_clf2['Validation F1 scores'])
metricas.insert(8,'Accuracy BA', model_ranfor2['Validation Accuracy scores'])
metricas.insert(9, 'Precision BA', model_ranfor2['Validation Precision scores'])
metricas.insert(10, 'Recall BA', model_ranfor2['Validation Recall scores'])
metricas.insert(11, 'F1 BA', model_ranfor2['Validation F1 scores'])
metricas.insert(12,'Accuracy XGB', model_XGB2['Validation Accuracy scores'])
metricas.insert(13, 'Precision XGB', model_XGB2['Validation Precision scores'])
metricas.insert(14, 'Recall XGB', model_XGB2['Validation Recall scores'])
metricas.insert(15, 'F1 XGB', model_XGB2['Validation F1 scores'])


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

### SELECCIÓN DEL MODELO Y OPTIMIZACIÓN DE HIPERPARÁMETROS####
#---------------------------------------------------------------#
metricas = pd.DataFrame()
metricas.insert(0,'Recall AD', model_clf1['Validation Recall scores'])
metricas.insert(0,'Recall BA', model_ranfor1['Validation Recall scores'])
metricas.insert(0,'Accuracy AD', model_clf1['Validation Accuracy scores'])
metricas.insert(0,'Accuracy BA', model_ranfor1['Validation Accuracy scores'])

metricas.boxplot(column=["Recall AD", "Recall BA", "Accuracy AD", "Accuracy BA"], grid=False)

# Optimización de hiperparámetros del modelo seleccionado : árbol de decisión

# Definición de cuadricula de hiperparámetros
parameters = {'max_depth': [16,18,20,22,24,26,28,30],
              'max_leaf_nodes': [150,200,250,300,350],
              'max_features':[35,'auto','sqrt','log2','none']}

X_train, X_test, y_train, y_test = train_test_split(Xenew, y, test_size=0.2, random_state=25)

rand_ad = RandomizedSearchCV(estimator=clf1, param_distributions=parameters, n_iter=20, scoring='recall', cv=20, verbose=False, random_state=1)

rand_ad.fit(X_train, y_train)

print('Best Params: ', rand_ad.best_params_)
print('Best Score: ', rand_ad.best_score_)
print('-----------------------------------------------------------------------')

# Aplicación de hiperparámetros
clff = tree.DecisionTreeClassifier(
          criterion = 'gini',
          random_state=25,
          max_depth = 30,
          max_leaf_nodes = 350,
          max_features = 'auto',
          class_weight = 'balanced')

clff.fit(X_train, y_train)

# Métricas del árbol
print ("Train - Accuracy :", metrics.accuracy_score(y_train, clff.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, clff.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, clff.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, clff.predict(X_test)))
print ("Train - Recall :", metrics.recall_score(y_train, clff.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, clff.predict(X_train)))
print ("Test - Recall :", metrics.recall_score(y_test, clff.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, clff.predict(X_test)))
print('-----------------------------------------------------------------------')

# Características del árbol
print(f"Profundidad del árbol: {clff.get_depth()}")
print(f"Número de nodos terminales: {clff.get_n_leaves()}")
print('-----------------------------------------------------------------------')

# Matriz de confusión
cm1= confusion_matrix(y_test, clff.predict(X_test))
# Visualización de la matriz de confusion
cm1_display = ConfusionMatrixDisplay(confusion_matrix = cm1)
cm1_display.plot()
plt.show()

# Visualización del árbol
fig, ax = plt.subplots(figsize=(22, 20))
plot = plot_tree(
            decision_tree = clff,
            feature_names = Xenew.columns,
            class_names   = ['False', 'True'],
            filled        = True,
            impurity      = False,
            fontsize      = 10,
            precision     = 2,
            ax            = ax
       )

# Evaluación del árbol de decisión
model_clf = cross_validation(clff, Xenew, y, 100)
print("Mean Training F1 Score Árbol de decisión 1: ", round(model_clf['Mean Training F1 Score'],4),
      "\nMean Validation F1 Score Árbol de decisión 1: ", round(model_clf['Mean Validation F1 Score'],4),
      "\nMean Training Precision Árbol de decisión 1: ", round(model_clf['Mean Training Precision'],4),
      "\nMean Validation Precision Árbol de decisión 1: ", round(model_clf['Mean Validation Precision'],4),
      "\nMean Training Recall Árbol de decisión 1: ", round(model_clf['Mean Training Recall'],4),
      "\nMean Validation Recall Árbol de decisión 1: ", round(model_clf['Mean Validation Recall'],4),
      "\nMean Training Accuracy Árbol de decisión 1: ", round(model_clf['Mean Training Accuracy'],4),
      "\nMean Validation Accuracy Árbol de decisión 1: ", round(model_clf['Mean Validation Accuracy'],4))

# Visualización de métricas
metricas = pd.DataFrame()
metricas.insert(0,'Accuracy train', model_clf['Training Accuracy scores'])
metricas.insert(1, 'Precision train', model_clf['Training Precision scores'])
metricas.insert(2, 'Recall train', model_clf['Training Recall scores'])
metricas.insert(3, 'F1 train', model_clf['Training F1 scores'])
metricas.insert(4,'Accuracy test', model_clf['Validation Accuracy scores'])
metricas.insert(5, 'Precision test', model_clf['Validation Precision scores'])
metricas.insert(6, 'Recall test', model_clf['Validation Recall scores'])
metricas.insert(7, 'F1 test', model_clf['Validation F1 scores'])

plt.figure(figsize=(8, 8))

plt.subplot(2,2,1)
metricas.boxplot(column=["Recall train", "Recall test"], grid=False)

plt.subplot(2,2,2)
metricas.boxplot(column=["F1 train", "F1 test"], grid=False)

plt.subplot(2,2,3)
metricas.boxplot(column=["Precision train", "Precision test"], grid=False)

plt.subplot(2,2,4)
metricas.boxplot(column=["Accuracy train", "Accuracy test"], grid=False)
plt.show()

# Guardar modelo
joblib.dump(clff, "clff_final.pkl")
clff_final = joblib.load("clff_final.pkl")