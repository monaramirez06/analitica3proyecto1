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

### Cargar datos
y = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/y.csv'
y = pd.read_csv((y), sep= ',')

Xenew = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/y.csv'
Xenew = pd.read_csv((Xenew), sep= ',')

## Cargar modelo y predecir
clff = joblib.load("clff_final.pkl")

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
Pred=pd.DataFrame(clff.predict(Xenew), columns=['Predicciones retiros'])

ff1 = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/df1.csv'
df1 = pd.read_csv((ff1), sep= ',')

    
df_final=pd.concat([df1,Pred],axis=1)

# Convertir variable objetivo a un tipo de variable que permita construir los indicadores
df_final['attrition'] = df_final['attrition'].astype('int64')
df_final.dtypes

# Indicadores de rotación por área - departamento con predicciones
areas = df_final.groupby(['department'])[['attrition pred']].sum()
areas1 = df_final.groupby(['department'])[['attrition pred']].count()
round(areas/areas1*100,2)

# Indicadores de rotación por área - departamento real
areas = df_final.groupby(['department'])[['attrition']].sum()
areas1 = df_final.groupby(['department'])[['attrition']].count()
round(areas/areas1*100,2)

# Indicadores de rotación por cargo - rol con predicciones
cargos = df_final.groupby(['jobrole'])[['attrition pred']].sum()
cargos1 = df_final.groupby(['jobrole'])[['attrition pred']].count()
round(cargos/cargos1*100,2)

# Indicadores de rotación por cargo - rol real
cargos = df_final.groupby(['jobrole'])[['attrition']].sum()
cargos1 = df_final.groupby(['jobrole'])[['attrition']].count()
round(cargos/cargos1*100,2)

# Indicadores de rotación por cargo - rol real
cargos = df_final.groupby(['department','jobrole'])[['attrition']].sum()
cargos1 = df_final.groupby(['department','jobrole'])[['attrition']].count()
round(cargos/cargos1*100,2)
    
# Indicadores de rotación por cargo - rol real
cargos = df_final.groupby(['department','jobrole'])[['attrition pred']].sum()
cargos1 = df_final.groupby(['department','jobrole'])[['attrition pred']].count()
round(cargos/cargos1*100,2)