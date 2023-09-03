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

# Cambiar los nombres de las variables a letra minuscula
dfg.columns = dfg.columns.map(str.lower)
dfe.columns = dfe.columns.map(str.lower)
dfsd.columns = dfsd.columns.map(str.lower)
dftw.columns = dftw.columns.map(str.lower)
dfr.columns = dfr.columns.map(str.lower)

# Tratamiento de la base de datos de retiros
dfr1 = pd.DataFrame()
dfr1.insert(0, 'employeeid', value=[i for i in range(1,4411)])

data_frames0 =[dfr1, dfr.iloc[:,0:2]]

dfa = reduce(lambda  left,right: pd.merge(left,right,on=['employeeid'],
                                            how='outer'), data_frames0)

dfa.sort_values('employeeid', inplace=True)
dfa.attrition =dfa.attrition.fillna("No")
dfa

# Unión bases de datos
# Unir las bases de datos por medio de key:
data_frames =[dfa, dfg, dfe, dfsd, dftw]

df = reduce(lambda  left,right: pd.merge(left,right,on=['employeeid'],
                                            how='outer'), data_frames)

pd.DataFrame.to_csv(df, 'merged.txt', sep=',', na_rep='.', index=False)

df

#tamaño de la base de datos
df.shape

#categoria y cantidad de datos no nulos por varible de inf general
df.info()

# Conversión de variables categóricas que aparentemente son numéricas
df.education = df['education'].astype(str)
df.joblevel = df['joblevel'].astype(str)
df.stockoptionlevel = df['stockoptionlevel'].astype(str)
df.jobinvolvement = df['jobinvolvement'].astype(str)
df.performancerating = df['performancerating'].astype(str)

df.info()

# Variables cualitativas nominales
print("Attrition: ", df['attrition'].unique())
print("Businesstravel: ", df['businesstravel'].unique())
print("Department: ", df['department'].unique())
print("Educationfield: ", df['educationfield'].unique())
print("Gender: ", df['gender'].unique())
print("Jobrole: ", df['jobrole'].unique())
print("Maritalstatus: ", df['maritalstatus'].unique())
print("Over18: ", df['over18'].unique())

#Variables cualitativas ordinales/categóricas
print("Education: ", df['education'].unique())
print("EnvironmentSatisfaction: ", df['environmentsatisfaction'].unique())
print("JobInvolvement: ", df['jobinvolvement'].unique())
print("JobSatisfaction: ", df['jobsatisfaction'].unique())
print("JobLevel: ", df['joblevel'].unique())
print("PerformanceRating: ", df['performancerating'].unique())
print("WorkLifeBalance: ", df['worklifebalance'].unique())

#cantidad de nulos de inf general
df.isnull().sum()

# En la base de datos, existen 5 variables con datos nulos:

# Numcompaniesworked: 19 nulos
# Environmentsatisfaction: 25 nulos
# Jobsatisfaction: 20 nulos
# Worklifebalance: 38 nulos
# Para hacer la imputación de los datos nulos en las variables numéricas (numcompaniesworked, mean_time) se hace una exploración estadística para identificar la existencia de valores atípicos.

# Para la imputación de datos nulos en las variables cualitativas (environmentsatisfaction, jobsatisfaction, worklifebalance) se usa la moda, es decir, el valor que más se repite en la muestra, para completar los datos faltantes.
