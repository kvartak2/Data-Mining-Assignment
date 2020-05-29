import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("water-treatment.data",sep=',',decimal=',',header=None)

df1 = pd.read_csv("water-treatment.data",sep=',',decimal=',',header=None)
df2=df1.replace({'?': pd.np.nan})


for i in range(1,len(df2.columns)):
    df2[i]=df2[i].astype(float)
    df2[i].fillna(df2[i].median(),inplace=True)


Q1 = df2.quantile(0.25)
Q3 = df2.quantile(0.75)
IQR = Q3 - Q1


#gives number of outliers in 38 attributes
print(((df2 < (Q1 - 1.5 * IQR)) | (df2 > (Q3 + 1.5 * IQR))).sum())

#this is a boxplot of instance 32 which shows 44 outliers
#sns.boxplot(x=df2[32])

for i in range(1,len(df2.columns)):
    df2[i].fillna(df2[i].median(),inplace=True)

x = (df2.iloc[:, 1:]).values
normalized = preprocessing.normalize(x, axis = 1)
df_normalized=pd.DataFrame(normalized)

#print(df_normalized)
