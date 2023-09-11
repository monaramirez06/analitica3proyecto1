# Importar librerias necesarias para el procesamiento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.impute import SimpleImputer

# Evitar salidas de Warnings
import warnings
warnings.filterwarnings("ignore")

#lectura de df1 
df1 = 'https://raw.githubusercontent.com/monaramirez06/analitica3proyecto1/main/df1.csv'

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

