# Importar librearias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

### Cargar datos
y = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/y.csv'
y = pd.read_csv((y), sep= ',')
y =y.drop('Unnamed: 0', axis=1)

Xenew = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/Xenew.csv'
Xenew = pd.read_csv((Xenew), sep= ',')
Xenew =Xenew.drop('Unnamed: 0', axis=1)

## Cargar modelo y predecir
X_train, X_test, y_train, y_test = train_test_split(Xenew, y, test_size=0.2, random_state=25)
clff = tree.DecisionTreeClassifier(
          criterion = 'gini',
          random_state=25,
          max_depth = 30,
          max_leaf_nodes = 350,
          max_features = 'auto',
          class_weight = 'balanced')

clff.fit(X_train, y_train)
#clff = joblib.load("clff_final.pkl")

# Importancia de variables para plan de acción
pd.set_option('display.max_rows', 100)
importancia1=pd.DataFrame(clff.feature_names_in_)
importancia2=pd.DataFrame(clff.feature_importances_)
importancia=pd.concat([importancia1,importancia2],axis=1)
importancia.columns=["variable","peso"]

importancia.sort_values("peso", inplace=True, ascending=False)
importancia

#Visualización de variables categóricas ordinales
plt.figure(figsize=(6, 6))
plt.barh(importancia.variable, importancia.peso)
plt.show()

# Predicciones
Pred=pd.DataFrame(clff.predict(Xenew), columns=['attrition pred'])

ff1 = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/df1.csv'
df1 = pd.read_csv((ff1), sep= ',')
df1 =df1.drop('Unnamed: 0', axis=1)
df1.head()

    
df_final=pd.concat([df1,Pred],axis=1)

# Convertir variable objetivo a un tipo de variable que permita construir los indicadores
df_final['attrition'] = df_final['attrition'].astype('int64')
df_final.dtypes

# Indicadores de rotación por área - departamento con predicciones
areas = df_final.groupby(['department'])[['attrition pred']].sum()
areas1 = df_final.groupby(['department'])[['attrition pred']].count()
print(round(areas/areas1*100,2))

# Indicadores de rotación por área - departamento real
areas = df_final.groupby(['department'])[['attrition']].sum()
areas1 = df_final.groupby(['department'])[['attrition']].count()
print(round(areas/areas1*100,2))

# Indicadores de rotación por cargo - rol con predicciones
cargos = df_final.groupby(['jobrole'])[['attrition pred']].sum()
cargos1 = df_final.groupby(['jobrole'])[['attrition pred']].count()
print(round(cargos/cargos1*100,2))

# Indicadores de rotación por cargo - rol real
cargos = df_final.groupby(['jobrole'])[['attrition']].sum()
cargos1 = df_final.groupby(['jobrole'])[['attrition']].count()
print(round(cargos/cargos1*100,2))

# Indicadores de rotación por cargo - rol real
cargos = df_final.groupby(['department','jobrole'])[['attrition']].sum()
cargos1 = df_final.groupby(['department','jobrole'])[['attrition']].count()
print(round(cargos/cargos1*100,2))
    
# Indicadores de rotación por cargo - rol real
cargos = df_final.groupby(['department','jobrole'])[['attrition pred']].sum()
cargos1 = df_final.groupby(['department','jobrole'])[['attrition pred']].count()
print(round(cargos/cargos1*100,2))