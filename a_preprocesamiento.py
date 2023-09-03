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

df_empl = pd.read_csv((employee), sep=',')
df_general =pd.read_csv((general), sep=';')
df_manager =pd.read_csv((manager), sep=',')
df_retirement =pd.read_csv((retirement), sep=';')

#verificar que las bases se hayan cargado de forma correcta
df_empl.head(5)
df_general.head(5) 
df_manager.head(5) 
df_retirement.head(5)