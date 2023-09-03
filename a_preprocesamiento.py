###importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso
from functools import reduce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
%matplotlib inline
from sklearn import tree
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Evitar salidas de Warnings
import warnings
warnings.filterwarnings("ignore")

####lectura de datos
employee = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/employee_survey_data.csv'
general = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/general_data.csv'
manager = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/manager_survey_data.csv'
retirement = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/retirement_info.csv'
intime = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/in_time.csv'
outime = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/out_time.csv'

dfe = pd.read_csv((employee), sep=',')
dfg =pd.read_csv((general), sep=';')
dfsd =pd.read_csv((manager), sep=',')
dfr =pd.read_csv((retirement), sep=';')
dfit= pd.read_csv((intime), sep=',')
dfot  = pd.read_csv((outime), sep=',')

#verificar que las bases se hayan cargado de forma correcta
dfe.head(5)
dfg.head(5) 
dfsd.head(5) 
dfr.head(5)
dfit.head(5)
dfot.head(5)

# Crear una copia de las bases de datos de horas antes de tratarlas
dfit1= dfit.copy()
dfot1= dfot.copy()

# Cambiar los tipos de datos a fechas y horas
dfit1.iloc[:,1:262] = dfit1.iloc[:,1:262] .astype('datetime64[ns]')
dfot1.iloc[:,1:262]  = dfot1.iloc[:,1:262] .astype('datetime64[ns]')

# Ordenar bases de datos por orden de ID
dfit1.sort_values('Unnamed: 0', inplace=True)
dfot1.sort_values('Unnamed: 0', inplace=True)

# Restar hora de salida y hora de entrada - Jornada laboral
dft1 = dfot1.iloc[:,1:262].sub(dfit1.iloc[:,1:262])
dft1.insert(0, 'Unnamed: 0', dfit1["Unnamed: 0"])

# Establecer el equivalente en horas de la jornada laboral
for i in range(0,4410):
  k=0
  for k in range(1,262):
    dft1.iloc[i,k] = dft1.iloc[i,k].total_seconds()/3600
    k=k+1
  i=i+1

# Sacar el promedio de horas que labora cada empleado
dft1["mean_time"]=''
for i in range(0,4410):
  dft1['mean_time'][i] = np.nanmean(dft1.iloc[i,1:262])
  i+1

# Crear base de datos con empleado y su respectivo promedio de jornada laboral
dftw = pd.DataFrame()
dftw.insert(0, 'employeeid', dft1["Unnamed: 0"])
dftw.insert(1, 'mean_time', dft1["mean_time"])
dftw