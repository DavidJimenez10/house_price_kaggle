#Importando librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, probplot
from scipy.special import boxcox1p
#lowess en  sns.residplot
import statsmodels.api as sm
#categoria a numero
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Analisis mutual information
from sklearn.feature_selection import mutual_info_regression
#config general
pd.set_option('display.max_columns',0)
sns.set_theme(style="darkgrid")

def count_null(df):
    """
    in: df <dataframe>
    out: <dataframe>
    Retorna un DF con el numero de valores nulos, el porcentaje de valores nulos y el tipo de la columna, para las columnas con al menos un valor nulo del df original
    """
    count_null = df.isnull().sum()
    porcen_null = count_null*100/df.shape[0]
    df_desc_nulos = pd.concat([count_null,porcen_null,df.dtypes],axis=1,keys= ["Count","Porcentaje","Type"])
    df_desc_nulos = df_desc_nulos[df_desc_nulos['Count']>0]
    df_desc_nulos.sort_values('Porcentaje',ascending=False,inplace=True)
    return np.transpose(df_desc_nulos)