import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter

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
'''
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
'''
#print(df_normalized)

pca = PCA().fit(df_normalized)

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

pca = PCA(n_components=5)

principalComponents = pca.fit_transform(df_normalized)
#principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10'])
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2','pc3','pc4','pc5'])
#print(principalDf)



'''-------------------------PCA END-------------------------'''

'''-------------------------k-means START-------------------'''

#print(principalDf.size)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(principalDf)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#from the graph, number of clusters are 4
kmeans = KMeans(n_clusters=4)
kmeans.fit(principalDf)


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
        #print("Element Exists")
        L1.append(value)
#print(L1)


x=Counter(Dict.values())
cluster_instance_dict=dict((l,k) for k,l in sorted([(j,i) for i,j in x.items()]))
print(cluster_instance_dict)