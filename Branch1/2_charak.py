import pandas as pd
rfmSegmentation =pd.read_csv('1_rfm/rfmTable.csv')
def charak(pddf):
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

    pddf['R_Quartile'] = pddf['recency'].apply(RClass, args=('recency',quantiles,))
    pddf['F_Quartile'] = pddf['frequency'].apply(FMClass, args=('frequency',quantiles,))
    pddf['M_Quartile'] = pddf['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))

    pddf['RFMClass'] = pddf.R_Quartile.map(str) \
                                + pddf.F_Quartile.map(str) \
                                + pddf.M_Quartile.map(str)

    pddf.to_csv('2_charak/charak_cluster.csv')


def panjak(pddf):
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

    pddf['R_Quintuple'] = pddf['recency'].apply(RClass, args=('recency', quantiles,))
    pddf['F_Quintuple'] = pddf['frequency'].apply(FMClass, args=('frequency', quantiles,))
    pddf['M_Quintuple'] = pddf['monetary_value'].apply(FMClass, args=('monetary_value', quantiles,))

    pddf['RFMClass'] = pddf.R_Quartile.map(str) \
                       + pddf.F_Quartile.map(str) \
                       + pddf.M_Quartile.map(str)

    pddf.to_csv('2_charak/panjak_cluster.csv')

charak(rfmSegmentation)
panjak(rfmSegmentation)