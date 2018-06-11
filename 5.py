import pandas as pd,matplotlib.pyplot as plt,numpy as np,matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering,AffinityPropagation,MeanShift

def silhouette(X,n_clusters,algorithm,monthNo):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    X=np.array(X)
    if algorithm == KMeans:
        clusterer = algorithm(n_clusters=n_clusters, random_state=10)
    elif algorithm == AgglomerativeClustering:
        clusterer = algorithm(n_clusters=n_clusters,linkage='ward')
    elif algorithm == SpectralClustering:
        clusterer = SpectralClustering (n_clusters=n_clusters)
    elif algorithm== AffinityPropagation:
        clusterer = AffinityPropagation(preference=-5.0, damping=0.95)
    elif algorithm==MeanShift:
        clusterer = MeanShift(0.175,cluster_all=False)
    cluster_labels = clusterer.fit_predict(X)

    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For algorithm =" , algorithm ,"   n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    X=np.array(X)
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    alg= ''
    if algorithm==KMeans :
        alg="Silhouette analysis for Preprocessing = real , algorith = KMeans "+str(n_clusters)+" , month = "+ str(monthNo)+",average silhouette_score = "+ str(silhouette_avg)
        plt.suptitle((alg),  fontsize=14, fontweight='bold')
        plt.savefig('result/Kmean/real'+'_Kmeans'+str(n_clusters)+'_m'+ str(monthNo))
    elif algorithm == AgglomerativeClustering :
        alg="Silhouette analysis for Preprocessing = real , algorith = AgglomerativeClustering "+str(n_clusters)+" , month = "+ str(monthNo)+",average silhouette_score = "+ str(silhouette_avg)
        plt.suptitle((alg),  fontsize=14, fontweight='bold')
        plt.savefig('result/AgglomerativeClustering/real'+'_Agg'+str(n_clusters)+'_m'+ str(monthNo))
    elif algorithm == AffinityPropagation:
        alg="Silhouette analysis for Preprocessing = real , algorith = AffinityPropagation , month = "+ str(monthNo)+",average silhouette_score = "+ str(silhouette_avg)
        plt.suptitle((alg),  fontsize=14, fontweight='bold')
        plt.savefig('result/AffinityPropagation/real'+'_Aff_m'+ str(monthNo))
    elif algorithm == MeanShift :
        alg="Silhouette analysis for Preprocessing = real , algorith = AffinityPropagation , month = "+ str(monthNo)+",average silhouette_score = "+ str(silhouette_avg)
        plt.suptitle((alg),  fontsize=14, fontweight='bold')
        plt.savefig('result/MeanShift/real'+'_MeanShift_m'+ str(monthNo))
    # plt.suptitle((alg),
    #              fontsize=14, fontweight='bold')

    # plt.show()

l1 =['result_fixed3/tempresult_3Fixed3.csv','result_fixed6/tempresult_3Fixed6.csv','result_fixed9/tempresult_3Fixed9.csv']

exit(-1)
for i in range(11,15):
    X = pd.read_excel('raw data/real/Merchants_Real Features_'+str(i)+'_960922.xlsx', sheet_name='Sheet1')[['all_transactions','sum_amounts' ]]
    # silhouette(X,2,SpectralClustering)
    # silhouette(X,3,SpectralClustering)
    # silhouette(X,4,SpectralClustering)
    # silhouette(X,5,SpectralClustering)
    # silhouette(X,6,SpectralClustering)
    # silhouette(X,7,SpectralClustering)
    # silhouette(X,8,SpectralClustering)
    # silhouette(X,9,SpectralClustering)
    # silhouette(X,10,SpectralClustering)
    silhouette(X,2,KMeans,i)
    silhouette(X,3,KMeans,i)
    silhouette(X,4,KMeans,i)
    silhouette(X,5,KMeans,i)
    silhouette(X,6,KMeans,i)
    silhouette(X,7,KMeans,i)
    silhouette(X,8,KMeans,i)
    silhouette(X,9,KMeans,i)
    silhouette(X,10,KMeans,i)
    silhouette(X,2,AgglomerativeClustering,i)
    silhouette(X,3,AgglomerativeClustering,i)
    silhouette(X,4,AgglomerativeClustering,i)
    silhouette(X,5,AgglomerativeClustering,i)
    silhouette(X,6,AgglomerativeClustering,i)
    silhouette(X,7,AgglomerativeClustering,i)
    silhouette(X,8,AgglomerativeClustering,i)
    silhouette(X,9,AgglomerativeClustering,i)
    silhouette(X,10,AgglomerativeClustering,i)
    #silhouette(X,0,AffinityPropagation,i)
    #silhouette(X,0,MeanShift,i)