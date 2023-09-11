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