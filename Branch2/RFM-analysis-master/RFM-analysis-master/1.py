from sklearn.decomposition import PCA
import time , matplotlib.pyplot as plt,seaborn as sns,sklearn.cluster as cluster, numpy as np,pandas as pd

pddf1 = pd.read_csv('rfm-table.csv',encoding = "ISO-8859-1")[['customer','frequency','recency','monetary_value']]
pddf2 = pddf1

pddf1_norm = pd.concat([pddf1['customer'],(pddf1[['frequency','recency','monetary_value']] - pddf1[['frequency','recency','monetary_value']].mean()) / (pddf1[['frequency','recency','monetary_value']].max() - pddf1[['frequency','recency','monetary_value']].min())],axis=1)
print (pddf1_norm)

customers = pddf1['customer']
measures = pddf1[['frequency','recency','monetary_value']]
normalize_measures = pddf1_norm[['frequency','recency','monetary_value']]

pca = PCA(n_components=2 )
# X is the matrix transposed (n samples on the rows, m features on the columns)
pca.fit(normalize_measures)

data = pca.transform(normalize_measures)

print (data)

def plot_clusters(pddf2,data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    pd_lables =pd.DataFrame(labels)
    pd_lables.columns = [str(algorithm)[str(algorithm).rfind('.')+1:str(algorithm).rfind("'")]]
    pddf2 = pd.concat([pddf2[pddf2.columns],pd_lables ], axis=1)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    #plt.show()
    return pddf2


sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

pddf2=plot_clusters(pddf2,data, cluster.KMeans, (), {'n_clusters':6})

pddf2=plot_clusters(pddf2,data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})

pddf2=plot_clusters(pddf2,data, cluster.MeanShift, (0.175,), {'cluster_all':False})

pddf2=plot_clusters(pddf2,data, cluster.SpectralClustering, (), {'n_clusters':6})

pddf2=plot_clusters(pddf2,data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})



print(pddf2)