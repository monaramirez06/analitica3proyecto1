# Importar librerias necesarias para el procesamiento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.impute import SimpleImputer

# Evitar salidas de Warnings
import warnings
warnings.filterwarnings("ignore")

# Lectura de las bases de datos
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

# Verificar que las bases se hayan cargado de forma correcta
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

# Restar hora de salida y hora de entrada -> Jornada laboral
dft1 = dfot1.iloc[:,1:262].sub(dfit1.iloc[:,1:262])
dft1.insert(0, 'Unnamed: 0', dfit1["Unnamed: 0"])

# Establecer el equivalente en horas de la jornada laboral
for i in range(0,4410):
  k=0
  for k in range(1,262):
    dft1.iloc[i,k] = dft1.iloc[i,k].total_seconds()/3600
    k=k+1
  i=i+1

# Establecer el promedio de horas que labora cada empleado
dft1["mean_time"]=''
for i in range(0,4410):
  dft1['mean_time'][i] = np.nanmean(dft1.iloc[i,1:262])
  i+1

# Crear base de datos con empleado y su respectivo promedio de jornada laboral
dftw = pd.DataFrame()
dftw.insert(0, 'employeeid', dft1["Unnamed: 0"])
dftw.insert(1, 'mean_time', dft1["mean_time"])
dftw

# Cambiar los nombres de las variables a letra minúscula
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

df.head()

# Tamaño de la base de datos
df.shape

# Categoria y cantidad de datos no nulos por varible de información general
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

# En la base de datos, existen cuatro variables con datos nulos:

# Numcompaniesworked: 19 nulos
# Environmentsatisfaction: 25 nulos
# Jobsatisfaction: 20 nulos
# Worklifebalance: 38 nulos

# Para hacer la imputación de los datos nulos en las variables numéricas (numcompaniesworked) se hace una exploración estadística para identificar la existencia de valores atípicos.
# Para la imputación de datos nulos en las variables cualitativas (environmentsatisfaction, jobsatisfaction, worklifebalance) se usa la moda, es decir, el valor que más se repite en la muestra, para completar los datos faltantes.

# Identificación de valores atípicos en variables numcompaniesworked
# Tamaño de las figuras
plt.figure(figsize=(10, 6))
df['numcompaniesworked'].plot(kind='box')
plt.title('Boxplot numcompaniesworked')
plt.show()

# En el diagrama de cajas y bigotes de las variables numéricas que contienen datos faltantes, se observa la existencia de datos por encima de 1.5 veces el rango intercuartílico (RIC), lo que indica la presencia de valores atípicos. Por tal motivo, se hace la imputación de datos a través de la mediana.
# Imputación de datos en valores ausentes con sklearn
df1 = df.copy()

# Definir imputer para variables numéricas
imp_num = SimpleImputer(strategy='median')
imp_num.fit(df1.iloc[:, [0,2,5,8,13,14,16,17,19,20,21,22,23,29]])
df1.iloc[:, [0,2,5,8,13,14,16,17,19,20,21,22,23,29]] = imp_num.transform(df1.iloc[:, [0,2,5,8,13,14,16,17,19,20,21,22,23,29]])

# Definir imputer para variables categóricas
imp_cat = SimpleImputer(strategy='most_frequent')
df1["environmentsatisfaction"] = imp_cat.fit_transform(df1["environmentsatisfaction"].values.reshape(-1, 1))
df1["jobsatisfaction"] = imp_cat.fit_transform(df1["jobsatisfaction"].values.reshape(-1, 1))
df1["worklifebalance"] = imp_cat.fit_transform(df1["worklifebalance"].values.reshape(-1, 1))
print(df1.isnull().sum())

df1.head(5)

# Exportar base de datos final
df1.to_csv("df1.csv")