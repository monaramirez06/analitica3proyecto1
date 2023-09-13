# Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Evitar salidas de Warnings
import warnings
warnings.filterwarnings("ignore")

#lectura de df1 
dfcompleto = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/df1.csv'
df1 = pd.read_csv((dfcompleto), sep= ',')
df1.head(5)
df1 =df1.drop('Unnamed: 0', axis=1)

# Conversión de variables categóricas que aparentemente son numéricas (En caso de no querer que se vuelvan dummies, comentar)
#df1.education = df1['education'].astype(str)
#df1.joblevel = df1['joblevel'].astype(str)
#df1.stockoptionlevel = df1['stockoptionlevel'].astype(str)
# df1.jobinvolvement = df1['jobinvolvement'].astype(str)
#df1.performancerating = df1['performancerating'].astype(str)
#df1.environmentsatisfaction = df1['environmentsatisfaction'].astype(str)
#df1.jobsatisfaction = df1['jobsatisfaction'].astype(str)
#df1.worklifebalance = df1['worklifebalance'].astype(str)

# Separación de variables numéricas y categóricas
dfnum = df1.iloc[:,[0,2,5,8,13,14,16,17,19,20,21,22,23,29]]
dfcat = df1.iloc[:,[1,3,4,6,7,9,10,11,12,15,18,24,25,26,27,28]]

#print(dfnum.head(5))
#print(dfcat.head(5))

# Descripción estadística de las variables cuantitativas
# Cantidad de datos, promedio, desviación estandar, mínimo, máximo y percentiles
dfnum.describe()

# En la tabla se observa que:

# Las variables employeecount y standardhours presentan una desviación estandar nula, lo que significa que para todo el conjunto de datos el valor de estas variables permanece constante y no tienen un nivel se significancia importante para determinar las razones por las cuales rota de manera frecuente el personal.
# Las demás variables poseen una desviación estandar significativa, sin embargo, la escala de los datos es muy diferente, lo que podría sesgar las predicciones solicitadas. Es necesario hacer una normalización de los datos para evitar sesgos.
# La edad de los trabajadores en la empresa se encuentra entre los 18 y 60 años de edad.
# El 50% de los empleados debe desplazarse 7 km de su domicilio a su trabajo.
# En promedio los empleados han trabajado en aproximadamente 3 compañías
# El 50% de los empleados tienen un aumento salarial del 14%
# El ingreso promedio de los empleados es de aproximadamanete $65.029,31

# Visualización variables numéricas
# Age
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.age,25)
plt.xlabel('Age')
plt.title('Histograma - age')

plt.subplot(1,3,2)
sns.kdeplot(x = "age", data = dfnum)
plt.xlabel('Age')
plt.title('Distribución - Age')

plt.subplot(1,3,3)
dfnum['age'].plot(kind='box')
plt.title('Boxplot - Age')
plt.show()

# Visualización variables numéricas
# Distancefromhome
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.distancefromhome,25)
plt.xlabel('distancefromhome')
plt.title('Histograma - distancefromhome')

plt.subplot(1,3,2)
sns.kdeplot(x = "distancefromhome", data = dfnum)
plt.xlabel('Distancefromhome')
plt.title('Distribución - distancefromhome')

plt.subplot(1,3,3)
dfnum['distancefromhome'].plot(kind='box')
plt.title('Boxplot - distancefromhome')

plt.show()

# Visualización variables numéricas
# Monthlyincome
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.monthlyincome,50)
plt.xlabel('monthlyincome')
plt.title('Histograma - monthlyincome')

plt.subplot(1,3,2)
sns.kdeplot(x = "monthlyincome", data = dfnum)
plt.xlabel('Monthlyincome')
plt.title('Distribución - monthlyincome')

plt.subplot(1,3,3)
dfnum['monthlyincome'].plot(kind='box')
plt.title('Boxplot - monthlyincome')

plt.show()

# Visualización variables numéricas
# Numcompaniesworked
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.numcompaniesworked,20)
plt.xlabel('numcompaniesworked')
plt.title('Histograma - numcompaniesworked')

plt.subplot(1,3,2)
sns.kdeplot(x = "numcompaniesworked", data = dfnum)
plt.xlabel('Numcompaniesworked')
plt.title('Distribución - numcompaniesworked')

plt.subplot(1,3,3)
dfnum['numcompaniesworked'].plot(kind='box')
plt.title('Boxplot - numcompaniesworked')

plt.show()

# Visualización variables numéricas
# Percentsalaryhike
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.percentsalaryhike,30)
plt.xlabel('percentsalaryhike')
plt.title('Histograma - percentsalaryhike')

plt.subplot(1,3,2)
sns.kdeplot(x = "percentsalaryhike", data = dfnum)
plt.xlabel('Percentsalaryhike')
plt.title('Distribución - percentsalaryhike')

plt.subplot(1,3,3)
dfnum['percentsalaryhike'].plot(kind='box')
plt.title('Boxplot - percentsalaryhike')

plt.show()

# Visualización variables numéricas
# Totalworkingyears
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.totalworkingyears,30)
plt.xlabel('totalworkingyears')
plt.title('Histograma - totalworkingyears')

plt.subplot(1,3,2)
sns.kdeplot(x = "totalworkingyears", data = dfnum)
plt.xlabel('Totalworkingyears')
plt.title('Distribución - totalworkingyears')

plt.subplot(1,3,3)
dfnum['totalworkingyears'].plot(kind='box')
plt.title('Boxplot - totalworkingyears')

plt.show()

# Visualización variables numéricas
# Trainingtimeslastyear
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.trainingtimeslastyear,30)
plt.xlabel('trainingtimeslastyear')
plt.title('Histograma - trainingtimeslastyear')

plt.subplot(1,3,2)
sns.kdeplot(x = "trainingtimeslastyear", data = dfnum)
plt.xlabel('Trainingtimeslastyear')
plt.title('Distribución - trainingtimeslastyear')

plt.subplot(1,3,3)
dfnum['trainingtimeslastyear'].plot(kind='box')
plt.title('Boxplot - trainingtimeslastyear')

plt.show()

# Visualización variables numéricas
# Yearsatcompany
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.yearsatcompany,40)
plt.xlabel('yearsatcompany')
plt.title('Histograma - yearsatcompany')

plt.subplot(1,3,2)
sns.kdeplot(x = "yearsatcompany", data = dfnum)
plt.xlabel('Yearsatcompany')
plt.title('Distribución - yearsatcompany')

plt.subplot(1,3,3)
dfnum['yearsatcompany'].plot(kind='box')
plt.title('Boxplot - yearsatcompany')

plt.show()

# Visualización variables numéricas
# Yearssincelastpromotion
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.yearssincelastpromotion,30)
plt.xlabel('yearssincelastpromotion')
plt.title('Histograma - yearssincelastpromotion')

plt.subplot(1,3,2)
sns.kdeplot(x = "yearssincelastpromotion", data = dfnum)
plt.xlabel('Yearssincelastpromotion')
plt.title('Distribución - yearssincelastpromotion')

plt.subplot(1,3,3)
dfnum['yearssincelastpromotion'].plot(kind='box')
plt.title('Boxplot - yearssincelastpromotion')
plt.show()

# Visualización variables numéricas
# Yearswithcurrmanager
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.yearswithcurrmanager,30)
plt.xlabel('yearswithcurrmanager')
plt.title('Histograma - yearswithcurrmanager')

plt.subplot(1,3,2)
sns.kdeplot(x = "yearswithcurrmanager", data = dfnum)
plt.xlabel('Yearswithcurrmanager')
plt.title('Distribución - yearswithcurrmanager')

plt.subplot(1,3,3)
dfnum['yearswithcurrmanager'].plot(kind='box')
plt.title('Boxplot - yearswithcurrmanager')
plt.show()

# Visualización variables numéricas
# Mean_time
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(dfnum.mean_time,50)
plt.xlabel('mean_time')
plt.title('Histograma - mean_time')

plt.subplot(1,3,2)
sns.kdeplot(x = "mean_time", data = dfnum)
plt.xlabel('Mean_time')
plt.title('Distribución - mean_time')

plt.subplot(1,3,3)
dfnum['mean_time'].plot(kind='box')
plt.title('Boxplot - mean_time')
plt.show()

# En las variables numéricas, se evidencia una variabilidad significativa. No obstante, en la mayoría de variables excepto en la variable Age, se presentan asimetrías positivas, lo cual indica una tendencia de los datos a ser mayor que la media de la distribución, esto puede inducir a sesgos. Así, puede considerarse realizar transformaciones matemáticas como la logaritmica para disminuir el sesgo y posiblemente, obtener mejores resultados.

# Por otro lado, se evidencian valores atípicos en las siguientes variables:

# Monthlyincome
# Numcompaniesworked
# Totalworkingyears
# Trainingtimeslastyear
# Yearsatcompany
# Yearssincelastpromotion
# Yearswithcurrmanager
# Mean_time

# Análisis de colinealidad - Correlación entre variables numéricas independientes
# Obtener matriz de correlación (coeficiente de correlación de pearson) para las variables numéricas
dfnum.drop(["employeecount","standardhours","employeeid"], axis = 1, inplace = True)
corr_df = dfnum.corr(method='pearson')

plt.figure(figsize=(20, 10))
sns.heatmap(corr_df, annot=True)
plt.title("Correlación entre variables numéricas")
plt.show()

# En la gráfica se evidencia problemas de colinealidad con los pares de variables:

# yearsatcompany y yearswithcurrmanager
# totalworkinyears y age
# Por lo tanto, es necesario seleccionar entre los pares, la variable más significativa para predecir la variable objetivo.

#Visualización de las variables categóricas nominales
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(df1.businesstravel,10)
plt.xlabel('Businesstravel')

plt.subplot(1,3,2)
plt.hist(df1.department,10)
plt.xlabel('Department')

plt.subplot(1,3,3)
plt.hist(df1.educationfield, 15)
plt.xlabel('Educationfield')

plt.show()

# Mayoritariamente los empleados viajan rara vez por cuestiones laborales
# Se evidencia que el departamento con más número de empleados es R&D y el menor es human resources
# Aproximadamente 1.750 empleados tienen educación en ciencias de la vida, la cual es más repetitiva entre los empleados. Seguida por medicina, alrededor de 1.300 empleados tienen conocimientos en este campo de la educación. por otra parte, los recursos humanos son el campo con menos número de empleados en este conocimiento

#Visualización de las variables categóricas nominales
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt.hist(df1.gender,10)
plt.xlabel('Gender')

plt.subplot(1,3,2)
plt.hist(df1.maritalstatus,10)
plt.xlabel('Maritalstatus')

plt.subplot(1,3,3)
plt.hist(df1.over18, 10)
plt.xlabel('Over18')

plt.show()

# Se evidencia que en la empresa la mayoria de los empleados son de género masculino con un total de aproximadamente 3.000 personas, y el del género femenino hay un aproximado de 1.800 personas
# Se puede observar que aproximadamente 2.000 empleados son casados, que 1.300 empleados son solteros y alrededor de 1.000 empleados son divorciados
# Se evidencia que todos los empleados de la base de datos son mayores de 18 años, por lo tanto, la variable no es significativa para el modelo de predicción requerido.

#Visualización de las variables categóricas nominales
plt.figure(figsize=(30, 6))

plt.subplot(1,2,1)
plt.hist(df1.stockoptionlevel,10)
plt.xlabel('Stockoptionlevel')

plt.subplot(1,2,2)
plt.hist(df1.worklifebalance,10)
plt.xlabel('Worklifebalance')

plt.show()

# Se evidencia que la mayoría de empleados tienen un nivel muy bueno de conciliación de la vida laboral y familiar.

#Visualización de las variables categóricas nominales
plt.figure(figsize=(30, 6))

plt.subplot(1,1,1)
plt.hist(df1.jobrole, 30)
plt.xlabel('jobrole')

plt.show()

# Los roles dentro de la empresa se encuentran organizados de la siguiente manera:

# Hay aproximadamente 1.000 empleados dedicados a "Sales Executive"
# En el área de Research scientist hay 900 empleados aproximadamente
# En el laboratory technician hay alrededor de 800 empleados
# En manufacturing director y healhcare representative hay 400 empleados
# En los cargos más estratégicos hay alrededor de 150 empleados en cada una de las áreas

#Visualización de variables categóricas ordinales
plt.figure(figsize=(30, 6))

plt.subplot(1,3,1) #df_real.attrition = df_real['attrition'].astype(int)
edu = df1['education']. astype(float)
plt.hist(edu, 10,color="purple")
plt.xlabel('Education - 1:No universitario, 2:Universitatio, 3:Licenciado, 4:Máster, 5:Doctor')

plt.subplot(1,3,2)
environ = df1['environmentsatisfaction']. astype(float)
plt.hist(environ,10, color="purple")
plt.xlabel('Environmentsatisfaction - 1:Bajo, 2:Medio, 3:Alto, 4:Muy alto')

plt.subplot(1,3,3)
jobin = df1['jobinvolvement']. astype(float)
plt.hist(df1.jobinvolvement,10, color="purple")
plt.xlabel('Jobinvolvement - 1:Bajo, 2:Medio, 3:Alto, 4:Muy alto')

plt.show()

# Los gráficos anteriores puntuan cada uno de los aspectos importantes en la vida del empleado:

# Se encuentra que los empleados en su mayoría tienen una educación en un nivel 3 que puede ser el promedio.
# El grado de satisfacción con el ambiente laboral puede ser considerado bueno, pues las puntuaciones más altas giran alrededor de 3 y 4
# Se puede decir que los trabajadores se encuentran medianamente involucrados con sus trabajos

#Visualización de variables categóricas ordinales
plt.figure(figsize=(30, 6))

plt.subplot(1,4,1)
jobs = df1['jobsatisfaction']. astype(float)
plt.hist(jobs, 10, color="purple")
plt.xlabel('Jobsatisfaction - 1:Bajo, 2:Medio, 3:Alto, 4:Muy alto')

plt.subplot(1,4,2)
jobl = df1['joblevel']. astype(float)
plt.hist(jobl,10, color="purple")
plt.xlabel('Joblevel')

plt.subplot(1,4,3)
perf = df1['performancerating']. astype(float)
plt.hist(perf,10, color="purple")
plt.xlabel('Performancerating - 1:Bajo, 2:Bueno, 3:Excelente, 4:Sobresaliente')

plt.subplot(1,4,4)
wok = df1['worklifebalance']. astype(float)
plt.hist(wok,10, color="purple")
plt.xlabel('worklifebalance - 1:Mala, 2:Buena, 3:Muy buena, 4:La mejor')

plt.show()

# se puede observar que:

# los empleados tienen un nivel alto y muy alto con respecto a la satisfaccion con rspecto a su trabajo
# La mayoria de los trabajadores se encuentran en el nivel 1 y nivel 2 en la organizacion, se puede deducir que los mandos operativos son el grueso de esta población.
# La valoracion repecto a su desempeño es en general muy buena, no hay empleados puntuados por debajo de 3
# la gran mayoria de los empleados sienten que hay un nivel muy bueno con la conciliacion entre su trabajo y su vida personal, es decir que tienen un balancen demasiado bueno

# Visualización de la variable objetivo
plt.figure(figsize=(10, 6))

plt.subplot(1,1,1)
plt.hist(df1.attrition,10)
plt.xlabel('Target: Attrition')

plt.show()

# La variable objetivo tiene un desbalance en la cantidad de muestras positivas y negativas. Se evidencia una cantidad mayoritaria de datos negativos, lo que puede generar sesgos en el modelo de predicción debido a que el modelo de entrenamiento no tendrá las muestras positivas necesarias para acertar en la predicción. Por lo tanto, se hace necesario integrar un hiperparámetro para intentar equilibrar el desempeño del modelo.

# Se renombran las categorias de la variable respuesta
df1['attrition'] = df1['attrition'].map( {'Yes':'1', 'No':'0'} ).fillna(df1['attrition'])
atr = df1['attrition']
df1.head(3)

# Exportar base de datos
df1.to_csv("df1.csv")

# Se hace copia al data set anterior
df2=df1.copy()

# Eliminación de columna atrition
# esta sera anexada luego, para poder hacer las demas variables dummies
df2.drop("attrition", axis = 1, inplace = True)

# eliminar variables que no representan importancia en el data frame
df2.drop("employeecount", axis = 1, inplace = True)
df2.drop("employeeid", axis = 1, inplace = True)
df2.drop("over18", axis = 1, inplace = True)
df2.drop("standardhours", axis = 1, inplace = True)

# Se convierten las variables a dummies
df2=pd.get_dummies(df2)
df2.head(2)

# Se vuelve a poner nuestra variable objetivo
df_real = pd.concat([atr,df2], axis=1)
df_real.attrition = df_real['attrition'].astype(int)
df_real.info()

# Exportar base de datos 
df_real.to_csv("df_real.csv")