import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import operator

df = pd.read_csv("water-treatment.data",sep=',',decimal=',',header=None)
df_NaN=df.replace({'?': pd.np.nan})


for i in range(1,len(df_NaN.columns)):
    df_NaN[i]=df_NaN[i].astype(float)

Q1 = df_NaN .quantile(0.25)
Q3 = df_NaN.quantile(0.75)
IQR = Q3 - Q1


for i in range(1,len(df_NaN.columns)):
    df_NaN[i].fillna(df_NaN[i].median(),inplace=True)

#removing first attribute
x = (df_NaN.iloc[:, 1:]).values

normalized = preprocessing.normalize(x, axis = 1)
df_normalized=pd.DataFrame(normalized)

#print(df_normalized)

Sum_of_squared_distances = []
K = range(1,12)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_normalized)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#from the graph, number of clusters are 4
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_normalized)

Dict = {} 
L1=[]
L=[]
dortedDictCnt = {}
x={}
cluster_instance_dict={}
for i in range(527):
    Dict.update({i:kmeans.labels_[i]})
#print(Dict)
    
    
for key,value in Dict.items():
    if(value not in L1): 
        print("Element Exists") 
        L1.append(value)  
#print(L1)    




    
x=Counter(Dict.values())
cluster_instance_dict=dict((l,k) for k,l in sorted([(j,i) for i,j in x.items()]))
print(cluster_instance_dict)
