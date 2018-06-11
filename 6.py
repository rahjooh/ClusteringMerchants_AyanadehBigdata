import glob , pandas as pd ,time,seaborn as sns,matplotlib.pyplot as plt ,sklearn.cluster as cluster,numpy as np,matplotlib.cm as cm,textwrap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import livy as ssn

plot_kwds = {'alpha': 0.75, 's': 80, 'linewidths': 0}

# filelist = glob.glob("raw data/*.xlsx")
filelist = ''

def silhouette(X,cluster_labels,n_clusters,clusterer):
    fig, (ax1, ax2) = plt.subplots(1, 2)
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
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg
          , '   DS TYPE =', type(X), '   DS = ', X, '   lable type =', type(cluster_labels))

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

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

def dimentionreduction(ds , algorithm='PCA',Recency=False):
    if Recency:
        measures = ds[['R', 'F', 'M']]
    else:
        measures = ds[['F', 'M']]
    if algorithm=='PCA':
        pca = PCA(n_components=2 )
        # X is the matrix transposed (n samples on the rows, m features on the columns)
        pca.fit(ds)
        np_result= pca.transform(ds)
        pd_result = pd.DataFrame(np_result)
        pd_result.columns = ['PCA1','PCA2']
        plt.scatter(pd_result['PCA1'],pd_result['PCA2'])
        plt.show()
        pd_result=pd.concat([ds[['MerchantNo','R','F','M']],pd_result[['PCA1','PCA2']]],axis=1)

        print(pd_result)
        print(type(pd_result))
        return pd_result
def sortLables(ds,column_name):
    labels =np.sort( ds[column_name].unique())
    l1 = []
    print('______________________________________________')
    print(ds.head())
    for l in labels:
        l1.append([l,ds[(ds[column_name] == l)]['F'].mean(),ds[(ds[column_name] == l) ]['M'].mean()])
    l1 = pd.DataFrame(l1)
    l1.columns = ['labelsOld','F','M']
    l1= l1.sort_values(by=['M','F'], ascending=[True,True])
    l1['labelsNew']=labels
    ds[column_name] = pd.DataFrame(ds[column_name]).replace(list(l1['labelsOld']), list(l1['labelsNew']))
    return ds
def plot_clusters(pddf2,data, algorithm, args, kwds , st, Recency = False ):
    start_time = time.time()
    if Recency:
        data = data[['M','F','R']]
    else:
        data = data[['M','F']]
    labels = algorithm(*args, **kwds).fit_predict(data)
    pd_lables =pd.DataFrame(labels)
    column_name =st +'_'+str(algorithm)[str(algorithm).rfind('.')+1:str(algorithm).rfind("'")]+str(args)+str(kwds)
    pd_lables.columns = [column_name]
    print(pddf2.columns)
    pddf2 = pd.concat([pddf2[pddf2.columns],pd_lables ], axis=1)
    pddf2 = sortLables(pddf2,column_name)
    end_time = time.time()
    palette = sns.color_palette('Set2', np.unique(labels).max() + 1)
    data= np.array(data[['M','F']])
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {} on {}'.format(str(algorithm.__name__),str(kwds)[str(kwds).index(':')+1:-1]), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    #plt.show()
    print(str(algorithm.__name__),str(kwds)[str(kwds).index(':')+1:-1])
    #silhouette(X=data,cluster_labels=labels,n_clusters=int(str(kwds)[str(kwds).index(':')+1:-1]),clusterer=algorithm)
    return pddf2[['MerchantNo',column_name]]
def charak(pddf,Recency):
    quantiles = pddf.quantile(q=[0.25,0.5,0.75])
    print('===============charak===============')
    print(quantiles)
    quantiles = quantiles.to_dict()

    # Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
    def RClass(x, p, d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]:
            return 3
        else:
            return 4

    # Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
    def FMClass(x, p, d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]:
            return 2
        else:
            return 1


    pddf['R'] = pddf['R'].apply(RClass, args=('R',quantiles,))
    pddf['F'] = pddf['F'].apply(FMClass, args=('F',quantiles,))
    pddf['M'] = pddf['M'].apply(FMClass, args=('M',quantiles,))

    if Recency :
        pddf['RFM_Quartile'] = pddf.R.map(str) + pddf.F.map(str) + pddf.M.map(str)
        pddf = pddf[['MerchantNo', 'R', 'F', 'M', 'RFM_Quartile']]
        pddf.columns = ['MerchantNo', 'R', 'F', 'M', 'RFM']
    else:
        pddf['RFM_Quartile'] =  pddf.F.map(str) + pddf.M.map(str)
        pddf = pddf[['MerchantNo', 'F', 'M', 'RFM_Quartile']]
        pddf.columns = ['MerchantNo', 'F', 'M', 'FM']

    return pddf

def panjak(pddf,Recency):
    pddf2 = pddf
    quantiles = pddf.quantile(q=[0.2, 0.4,0.6, 0.8])
    print('===============panjak===============')
    print(quantiles)
    quantiles = quantiles.to_dict()

    # Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
    def RClass(x, p, d):
        if x <= d[p][0.2]:
            return 1
        elif x <= d[p][0.4]:
            return 2
        elif x <= d[p][0.6]:
            return 3
        elif x <= d[p][0.8]:
            return 4
        else:
            return 5

    # Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
    def FMClass(x, p, d):
        if x <= d[p][0.2]:
            return 1
        elif x <= d[p][0.4]:
            return 2
        elif x <= d[p][0.6]:
            return 3
        elif x <= d[p][0.8]:
            return 4
        else:
            return 5

    pddf['R'] = pddf['R'].apply(RClass, args=('R',quantiles,))
    pddf['F'] = pddf['F'].apply(FMClass, args=('F',quantiles,))
    pddf['M'] = pddf['M'].apply(FMClass, args=('M',quantiles,))

    if Recency :
        pddf['RFM_Quartile'] = pddf.R.map(str) + pddf.F.map(str) + pddf.M.map(str)
        pddf = pddf[['MerchantNo', 'R', 'F', 'M', 'RFM_Quartile']]
        pddf.columns = ['MerchantNo', 'R', 'F', 'M', 'RFM']
    else:
        pddf['RFM_Quartile'] =  pddf.F.map(str) + pddf.M.map(str)
        pddf = pddf[['MerchantNo', 'F', 'M', 'RFM_Quartile']]
        pddf.columns = ['MerchantNo', 'F', 'M', 'FM']

    return pddf
def fixed (pddf , n,Recency):
    size =int(pddf.shape[0]/(n)) +1
    # cluster M
    m = pddf[['MerchantNo','M']].sort_values(['M'])
    list0 = m.values.tolist()
    list1 = []
    for i,line in enumerate(list0):
            list1.append([line[0],int(i/size)])
    pddf = pd.merge(pddf,pd.DataFrame(list1,columns=['MerchantNo','M_f'+str(n)]),on='MerchantNo')

    # cluster F
    f = pddf[['MerchantNo','F']].sort_values(['F'])
    list0 = f.values.tolist()
    list1 = []
    for i,line in enumerate(list0):
            list1.append([line[0],int(i/size)])
    pddf = pd.merge(pddf,pd.DataFrame(list1,columns=['MerchantNo','F_f'+str(n)]),on='MerchantNo')

    # cluster M
    r = pddf[['MerchantNo','R']].sort_values(['R'])
    list0 = r.values.tolist()
    list1 = []
    for i,line in enumerate(list0):
            list1.append([line[0],int(i/size)])
    pddf = pd.merge(pddf,pd.DataFrame(list1,columns=['MerchantNo','R_f'+str(n)]),on='MerchantNo')
    if Recency:
        pddf1 = pddf[['MerchantNo','M_f'+str(n),'F_f'+str(n),'R_f'+str(n)]]
        pddf1.columns = ['MerchantNo','M','F','R']
        pddf1['RFM'] = pddf1.R.map(str)+ pddf1.F.map(str) +pddf1.M.map(str)
    else:
        pddf1 = pddf[['MerchantNo','M_f'+str(n),'F_f'+str(n)]]
        pddf1.columns = ['MerchantNo','M','F']
        pddf1['FM'] =  pddf1.F.map(str) +pddf1.M.map(str)
    return pddf1
def clusterit1 (pddf,data,algorithm,):
    range_n_clusters = [2, 3, 4, 5, 6 ,7 ,8 ,9 ,10]
    sns.set_context('poster')
    sns.set_color_codes()
    plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

    plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    pddf2 = plot_clusters(pddf, data, cluster.KMeans, (), {'n_clusters': 6})

    pddf2 = plot_clusters(pddf2, data, cluster.AffinityPropagation, (), {'preference': -5.0, 'damping': 0.95})

    pddf2 = plot_clusters(pddf2, data, cluster.MeanShift, (0.175,), {'cluster_all': False})

    pddf2 = plot_clusters(pddf2, data, cluster.SpectralClustering, (), {'n_clusters': 6})

    pddf2 = plot_clusters(pddf2, data, cluster.AgglomerativeClustering, (), {'n_clusters': 6, 'linkage': 'ward'})
def sendlivy():
    code_data1 = {'code': textwrap.dedent("""spark.conf.set("spark.sql.shuffle.partitions",5)
        spark.conf.get("spark.sql.shuffle.partitions")
        spark.conf.set("spark.driver.memory","10g")
        from pyspark.sql import Row
        from pyspark.mllib.clustering import KMeans, KMeansModel
        from pyspark.sql.functions import monotonically_increasing_id
        import pyspark.sql.functions as sf    # for sum agg
        pqDF = spark.read.parquet("hdfs://10.100.136.40:9000/user/hduser/pqTotal")
        Sp_Df = pqDF.select("Merchantnumber","Amount",((((pqDF.FinancialDate.substr(4,2).cast('int'))+(12*((pqDF.FinancialDate.substr(1,2).cast('int'))-94)))- {0})*30+(pqDF.FinancialDate.substr(7,2).cast('int'))).alias('dayNum')).filter("ProccessCode='000000' AND MessageType='200'AND SuccessOrFailure='S'AND FinancialDate between {1} and {2}")
        Sp_Df= Sp_Df.groupBy(Sp_Df.Merchantnumber,Sp_Df.dayNum).agg(sf.sum(Sp_Df.Amount).alias("TxnSum"),sf.count(Sp_Df.Amount).alias("TxnNo"))
        Sp_Df = Sp_Df.groupBy(Sp_Df.Merchantnumber).agg(sf.sum(Sp_Df.TxnNo).alias("all_transactions"),sf.sum(Sp_Df.TxnSum).alias("sum_amounts"),sf.sum((Sp_Df.TxnNo)/Sp_Df.dayNum).alias("harmonic"))
        Sp_rdd = Sp_Df.rdd
        Sp_Piperdd = Sp_rdd.map(lambda x:(x[1],x[2],x[3]))
        model = KMeans.train(Sp_Piperdd, {3})
        labels =model.predict(Sp_Piperdd)
        labels = labels.map(Row("label")).toDF()   #add schema
        df11 =  Sp_Df.withColumn("id", monotonically_increasing_id())
        df22 =  labels.withColumn("id", monotonically_increasing_id())
        newDF = df11.join(df22, df11.id == df22.id, 'inner').drop(df22.id)
        pandas_df=newDF.toPandas()
        pandas_df.to_json()""").format( 1,"'" + 2 + "'", "'" + 3 + "'",4)}
def clusterit(pddf,data ,status, Recency = False):
    print(pddf)
    range_n_clusters = [3, 4, 5, 6 ,7 ,8 ,9 ,10]
    sns.set_context('poster')
    sns.set_color_codes()
    if Recency and (status == 'real' or status =='wt'):
        columns = ['MerchantNo', 'M', 'F', 'R','Lable' ]
    elif Recency and status != 'real' and status!='wt':
        columns = ['MerchantNo', 'M', 'F', 'R', 'RFM','Lable']
    elif status == 'real' or status=='wt':
        columns = ['MerchantNo', 'M', 'F','Lable']
    else:
        columns = ['MerchantNo', 'M', 'F', 'FM','Lable']
    columns='MerchantNo'
    plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    MasterDF = pd.DataFrame()
    for i in range_n_clusters:
        pddf2 = plot_clusters(pddf2=pddf, data=data, algorithm=cluster.KMeans,args=(), kwds={'n_clusters': i},st=status)
        if MasterDF.empty:
            MasterDF = pddf2
        else:
            MasterDF = pd.merge(MasterDF, pddf2, how='outer', on=columns)
    pddf2 = plot_clusters(pddf, data, cluster.AffinityPropagation, (), {'preference': -5.0, 'damping': 0.95},st=status)
    MasterDF = pd.merge(MasterDF, pddf2, how='outer', on=columns)

    pddf2 = plot_clusters(pddf2=pddf, data=data,algorithm=cluster.MeanShift, args=(0.175,), kwds={'cluster_all': False},st=status)
    MasterDF = pd.merge(MasterDF, pddf2, how='outer', on=columns)

    for i in range_n_clusters:
        pddf2 = plot_clusters(pddf, data, cluster.SpectralClustering, (), {'n_clusters': i},st=status)
        MasterDF = pd.merge(MasterDF, pddf2, how='outer', on=columns)
    for i in range_n_clusters:
        pddf2 = plot_clusters(pddf, data, cluster.AgglomerativeClustering, (), {'n_clusters': i, 'linkage': 'ward'},st=status)
        MasterDF = pd.merge(MasterDF, pddf2, how='outer', on=columns)
    return MasterDF
def preprocessing (path , status , normalization =False ,DimReduc=False , Recency = False):
    '''
    :param path: 
    :param status: 
                    wt  use RFM wieghted tag with included in file
                    real   use the exact value of the RFM 
                    Quirtile   create 3 threshold of 25% 50% 75% and create 4 part for each measure
                    Quintuple   create 4 threshold of 20% 40% 60% 80% and create 5 part for each measure
                    Quintuple   create 4 threshold of 20% 40% 60% 80% and create 5 part for each measure
                    Fixed#     create # equal cluster in each measure of RFM which # could be any number
    :param normalization: 
    :param DimReduc: 
    :return: 
    '''
    df = pd.read_excel(path, sheet_name='Sheet1')
    t = True
    #pddf.columns = ['F','R','M','MerchantNo','Mwt','Fwt','Rwt','Lable','month']
    l1=df.month.unique().tolist()
    result = pd.DataFrame()
    for m in l1 :
        print('number of month :',m)
        pddf = df[(df.month == m)]
        if status == 'wt' :
            if Recency:
                pddf = pddf[['MerchantNo','Rwt','Fwt','Mwt','Lable']]
                pddf.columns =['MerchantNo','R','F','M','Lable']
            else:
                pddf = pddf[['MerchantNo','Fwt','Mwt','Lable']]
                pddf.columns =['MerchantNo','F','M','Lable']
        elif status == 'real' :
            if Recency:
                pddf = pddf[['MerchantNo','R','F','M','Lable']]
            else:
                pddf = pddf[['MerchantNo','F','M','Lable']]
        elif status=='Quirtile':
            pddf = charak(pddf,Recency)
        elif status == 'Quintuple':
            pddf = panjak(pddf,Recency)
        elif status[:-1]=='Fixed':
            pddf = fixed(pddf,int(status[-1]),Recency)
        if normalization:
            if Recency :
                pddf = pd.concat([pddf['MerchantNo'], (pddf[['R','F','M']] - pddf[['R','F','M']].mean()) / ( pddf[['R','F','M']].max() - pddf[['R','F','M']].min())], axis=1)
            else:
                pddf = pd.concat([pddf['MerchantNo'], (pddf[['F', 'M']] - pddf[['F', 'M']].mean()) / ( pddf[['F', 'M']].max() - pddf[['F', 'M']].min())], axis=1)
        if DimReduc :
            pddf = dimentionreduction(pddf,Recency=Recency)

        if Recency :clusterit(pddf,pddf[['MerchantNo','R','F','M']],status,Recency)
        else:clusterit(pddf,pddf[['MerchantNo','F','M']],status,Recency)
        #pddf['month']=m
        if Recency :
            pddf_all=clusterit(pddf,pddf[['MerchantNo','R','F','M']],status,Recency)
        else:
            pddf_all =clusterit(pddf,pddf[['MerchantNo','F','M']],status,Recency)
        #pddf_all['month'] = m
        pddf_all.insert(0, 'month1', m)
        print('++++++++++++++++++++++++++++++')
        print(pddf_all.head())
        if result.empty:
            result = pddf_all[:]
        else:
            result = result.append(pddf_all)
    return result


pddf1 =preprocessing('Merchant_tagged_all.xlsx',status='wt')
print('===============================================================')
pddf1.to_csv('result.csv')