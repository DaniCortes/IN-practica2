# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas y Daniel Molina Cabrera
Fecha:
    Noviembre/2022
Contenido:
    Ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''


# Increment font size
import plotly.express as px
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
from math import floor
import seaborn as sns
sns.set(font_scale=2)


def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


datos = pd.read_csv('Fire-Incidents_v2.csv')
'''
for col in datos:
   missing_count = sum(pd.isnull(datos[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

# Se pueden reemplazar los valores desconocidos por un número
# datos = datos.replace(np.NaN,0)

# O imputar, por ejemplo con la media
# print(datos.head())
# for col in datos:
#     if col != 'DB040':
#         datos[col].fillna(datos[col].mean(), inplace=True)

# Seleccionar casos
subset = datos
# subset = datos[datos["Fire_Alarm_System_Presence"]=="Fire alarm system present"]
# subset = datos[datos["Arrival_Time"] < 500]
# subset = datos[datos["Month"].isin(["June", "July", "August", "September"])]
# subset = datos[datos["Business_Impact"]!="No business interruption"]
# subset = datos[~datos["Day_Of_Week"].isin(["Saturday", "Sunday", "Friday"])]
# subset = datos[~datos["Area_of_Origin"].str.contains("Cooking", regex=True)]
subset = datos[datos["Possible_Cause"].str.contains("Electrical Failure")]
title_centroides = "Causa Eléctrica"
print(subset.shape)

# Seleccionar variables de interés para clustering
# renombramos las variables por comodidad
subset = subset.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados",
                       "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Estimated_Dollar_Loss": "Coste Material", "Arrival_Time": "Tiempo"})
usadas = ["Muertos", "Rescatados", "Desplazados", "Coste Material", "Tiempo"]
# usadas = ["Muertos", "Rescatados", "Desplazados", "Coste Material"]

n_var = len(usadas)
X = subset[usadas]
# eliminar outliers como aquellos casos fuera de 1.5 veces el rango intercuartil
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
# X = X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)]

# normalizamos
X_normal = X.apply(norm_to_zero_one)

print('----- Ejecutando k-Means', end='')
k_means = KMeans(init='k-means++', n_clusters=5, n_init=5, random_state=123456)
t = time.time()
cluster_predict = k_means.fit_predict(X_normal)
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')

# Esto es opcional, el cálculo de Silhouette puede consumir mucha RAM.
# Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
muestra_silhoutte = 0.2 if (len(X) > 12000) else 1.0
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(
    muestra_silhoutte*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

metric_SC_samples = metrics.silhouette_samples(
    X_normal, cluster_predict, metric='euclidean')


# se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict, index=X.index, columns=['cluster'])

print("Tamaño de cada cluster:")
size = clusters['cluster'].value_counts()
for num, i in size.items():
    print('%s: %5d (%5.2f%%)' % (num, i, 100*i/len(clusters)))

size = size.sort_index()

centers = pd.DataFrame(k_means.cluster_centers_, columns=list(X))
centers_desnormal = centers.copy()

# se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * \
        (X[var].max() - X[var].min())

# '''
print("---------- Heatmap de centroides...")
plt.figure()
centers.index += 1
plt.figure()
hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal,
                 annot_kws={"fontsize": 18}, fmt='.3f')
plt.xticks(rotation=30)
hm.set_ylim(len(centers), 0)
# increase font size of all elements

if title_centroides:
    hm.set_title(title_centroides)

hm.figure.set_size_inches(15, 15)
hm.figure.savefig("centroides.pdf")
centers.index -= 1
# '''

k = len(size)
colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)

# se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)

'''
print("---------- Scatter matrix...")
plt.figure()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans.sample(frac=0.15), vars=variables, hue="cluster", palette=colors, plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
sns_plot.fig.set_size_inches(40,15)
sns_plot.savefig("scatter.pdf")
#plt.show()
'''

print("---------- Distribución por variable y cluster...")
plt.figure()
mpl.style.use('default')
fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(15, 15))
fig.subplots_adjust(wspace=0, hspace=0)


# ordenamos por renta para el plot
centers_sort = centers.sort_values(by=['Coste Material'])

rango = []
for j in range(n_var):
    rango.append([X_kmeans[usadas[j]].min(), X_kmeans[usadas[j]].max()])

for i in range(k):
    c = centers_sort.index[i]
    dat_filt = X_kmeans.loc[X_kmeans['cluster'] == c]
    for j in range(n_var):
        # ax = sns.kdeplot(x=dat_filt[usadas[j]], label="", shade=True, color=colors[c], ax=axes[i,j])
        # mejor si se usa weights de 'DB090'
        ax = sns.histplot(
            x=dat_filt[usadas[j]], label="", color=colors[c], ax=axes[i, j], kde=True)
        # ax = sns.boxplot(x=dat_filt[usadas[j]], notch=True, color=colors[c], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

        ax.set(xlabel=usadas[j] if (i == k-1) else '',
               ylabel='Cluster '+str(c+1) if (j == 0) else '')

        ax.set(yticklabels=[])
        ax.tick_params(left=False)
        ax.grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
        ax.grid(axis='y', visible=False)

        ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]),
                    rango[j][1]+0.05*(rango[j][1]-rango[j][0]))

fig.set_size_inches(15, 15)
fig.savefig("distribucion.pdf")

# '''
print("---------- Distancia intercluster...")
fig = plt.figure()
mpl.style.use('default')

mds = MDS(random_state=123456)
centers_mds = mds.fit_transform(centers)

# mejor si se usa weights de 'DB090' para size
plt.scatter(centers_mds[:, 0], centers_mds[:, 1],
            s=size**1.6, alpha=0.75, c=colors)
for i in range(k):
    plt.annotate(str(i+1), xy=centers_mds[i],
                 fontsize=18, va='center', ha='center')
xl, xr = plt.xlim()
yl, yr = plt.ylim()
plt.xlim(xl-(xr-xl)*0.13, xr+(xr-xl)*0.13)
plt.ylim(yl-(yr-yl)*0.13, yr+(yr-yl)*0.13)
plt.xticks([])
plt.yticks([])
fig.set_size_inches(15, 15)
plt.savefig("intercluster.pdf")
# '''


# '''
print("---------- Parallel coordinates...")
plt.figure()
mpl.style.use('default')


X = X.assign(cluster=clusters)
# X.loc[:, 'cluster'] = clusters
X.loc[:, 'SC'] = metric_SC_samples
df = X

# si se desea aclarar la figura, se pueden eliminar los objetos más lejanos, es decir, SC < umbral, p.ej., 0.3
df = df.loc[df['SC'] >= 0.3]

colors_parcoor = [(round((i//2)/k+(1/k)*(i % 2), 3), 'rgb'+str(colors[j//2]))
                  for i, j in zip(range(2*k), range(2*k))]

fig = px.parallel_coordinates(df, dimensions=usadas,
                              color="cluster", range_color=[-0.5, k-0.5],
                              color_continuous_scale=colors_parcoor)

fig.update_layout(coloraxis_colorbar=dict(
    title="Clusters",
    tickvals=[i for i in range(k)],
    ticktext=["Cluster "+str(i+1) for i in range(k)],
    lenmode="pixels", len=500,
))

fig.write_html("parallel.html")
# '''
