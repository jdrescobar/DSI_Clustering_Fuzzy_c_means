# =============================================================================
# 
# #0. Proceso básico inicial
# 
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy import stats
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D    

#Definición de paleta de colores
colors = ['b', 'orange', 'g', 'grey', 'c', 'm', 'y', 'turquoise', 'Brown',
          'ForestGreen', 'cornflowerblue']
#Número de columnas a mostrar de un dataframe
pd.options.display.max_columns = 50

# =============================================================================
# 
# #1. Carga de datos
# 
# =============================================================================

df = pd.read_csv("nts_data.csv")
print(df.describe())

# =============================================================================
# 
# #2. Transformación y Normalización de los Datos
# 
# =============================================================================

def normalize(dataframe):
    from sklearn import preprocessing 
    min_max_scaler = preprocessing.MinMaxScaler()
    datanorm = min_max_scaler.fit_transform(dataframe)
    return datanorm

#Quitamos mode_main por definición del enunciado del trabajo
df1 = df.drop(['mode_main'], axis=1)
#Quitamos las variables categóricas
df1 = df1.drop(['license', 'male', 'ethnicity', 'education',
                'income', 'weekend'], axis=1) 

#print(df1.head().T)

#Normalizamos los datos de los grupos ya que las magnitudes son muy dispares
datanorm = normalize(df1) #Min_Max_Scaler

# =============================================================================
# 
# #3. PCA. Análisis de Componentes Principales
# 
# =============================================================================

def pca(n, datanorm):
    from sklearn.decomposition import PCA

    estimator = PCA (n_components = n)
    X_pca = estimator.fit_transform(datanorm)
    return X_pca, estimator.explained_variance_ratio_, estimator.components_

#Reducimos la dimensionalidad a 4 componentes (~76% de ratio de varianza)
X_pca, vr, ec = pca(4, datanorm)
variance_ratio = 0
for v in vr:
    variance_ratio = variance_ratio + v
print('Variance Ratio =', variance_ratio, ':', vr)

print('\nEstimator Components:')
df2 = pd.DataFrame(np.matrix.transpose(ec),
               columns=['PC-1', 'PC-2', 'PC-3', 'PC-4'], index=df1.columns)
print(df2.head(10))

#Representación 3D de los componentes principales PCA
fig = plt.figure(figsize=(8,8))
fig.suptitle('Representación 3D PCA', fontsize=16)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=0.3)
fig.tight_layout()

# =============================================================================
# 
# #4. Análisis Cluster (Fuzzy c-means)
# 
# =============================================================================

#Parámetros iniciales
m = 1.4                  #Parámetro de fcm, 2 es el defecto
max_iteraciones = 100    #Número de iteraciones
tolerancia = 1e-5        #Tolerancia en el criterio de parada

def estimate_k_fuzzy(X_pca, m, tolerancia, max_iteraciones):
    fpcs = []
    fig1 = plt.figure(figsize=(15,15))
    for ncenters in range(1, 10):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X_pca.T, ncenters, m, error=tolerancia,
                maxiter=max_iteraciones, init=None)
        
        #Guardamos los valores fpc
        fpcs.append(fpc)
        #Mostramos los cluster asignados por elemento del conjunto de datos
        cluster_membership = np.argmax(u, axis=0)
        ax = fig1.add_subplot(3, 3, ncenters, projection='3d')
        for j in range(ncenters):
            ax.scatter(X_pca[(cluster_membership == j),0],
                    X_pca[(cluster_membership == j),1],
                    X_pca[(cluster_membership == j),2],
                    color=colors[j], s=1)
        #Marcamos el centro de cada fuzzy cluster
        for pt in cntr:
            ax.scatter(pt[0], pt[1], pt[2], color='r', s=50)
        ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    fig1.tight_layout()
    return fpcs

def fuzzy_c_means(X_pca, m, tolerancia, max_iteraciones, n_centers):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X_pca.T, n_centers, m, error=tolerancia,
                maxiter=max_iteraciones, init=None)
    cluster_membership = np.argmax(u, axis=0)
    return cntr, cluster_membership

#Análisis previo para averiguar el número de clústers   
fpcs = estimate_k_fuzzy(X_pca, m, tolerancia, max_iteraciones)

#Muestra FPC vs K
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[1:10], fpcs)
ax2.set_title('FPC vs K', fontsize=16)
ax2.set_xlabel("Numero de centros")
ax2.set_ylabel("Fuzzy partition coefficient")
fig2.tight_layout()

#Segun el análisis previo, seleccionamos k=2 para el grupo inicial
k=2
cntr, lbls = fuzzy_c_means(X_pca, m, tolerancia, max_iteraciones, k)

# =============================================================================
# 
# #5. Validación y Descripción de los Grupos
# 
# =============================================================================

#Validación
print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(df1, lbls))

fig3 = plt.figure(figsize=(8,8))
fig3.suptitle('Representación 3D Grupos', fontsize=16)
ax = fig3.add_subplot(111, projection='3d')
for j in range(k):
    ax.scatter(X_pca[(lbls == j),0], X_pca[(lbls == j),1],
                     X_pca[(lbls == j),2], color=colors[j], s=1)
#Marcamos el centro de cada fuzzy cluster
for pt in cntr:
    ax.scatter(pt[0], pt[1], pt[2], color='r', s=50)

fig3.tight_layout()

df2 = df1
df2['fcm'] = lbls


res_mean = df2[['distance', 'age', 'bicycles', 'cars', 'diversity',
           'green', 'precip', 'temp', 'wind',
           'fcm']].groupby(('fcm')).mean()
res_max = df2[['distance', 'age', 'bicycles', 'cars', 'diversity',
           'green', 'precip', 'temp', 'wind',
           'fcm']].groupby(('fcm')).max()
res_min = df2[['distance', 'age', 'bicycles', 'cars', 'diversity',
           'green', 'precip', 'temp', 'wind',
           'fcm']].groupby(('fcm')).min()

axis = res_mean.plot(kind='barh', legend=True)
axis.set_title('Descripción Clusters', fontsize=16)
axis.set_ylabel('Grupos')
axis.set_xlabel('Valores medios')

axis = res_max.plot(kind='barh', legend=True)
axis.set_title('Descripción Clusters', fontsize=16)
axis.set_ylabel('Grupos')
axis.set_xlabel('Valores maximos')

axis = res_min.plot(kind='barh', legend=True)
axis.set_title('Descripción Clusters', fontsize=16)
axis.set_ylabel('Grupos')
axis.set_xlabel('Valores minimos')

df3 = df.drop(['mode_main'], axis=1)
df3['fcm'] = lbls

res = df3[['distance', 'weekend', 'male', 'age', 'education', 'ethnicity',
           'license', 'income', 'cars', 'bicycles', 'density', 'diversity',
           'green', 'temp', 'precip', 'wind', 'fcm']].groupby(('fcm'))

print(res.describe(include='all').T)

