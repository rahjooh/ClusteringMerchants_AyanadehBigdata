import pandas as pd
import datetime as dt
NOW = dt.datetime(2016,9,22)
g_days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
j_days_in_month = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]
# err : one day
def __gregorianToJalali(gyear,gmonth,gday):

    """ 
             g_y: gregorian year 
             g_m: gregorian month 
             g_d: gregorian day 
         """

    global g_days_in_month, j_days_in_month

    gy = gyear - 1600
    gm = gmonth - 1
    gd = gday - 1

    g_day_no = 365 * gy + (gy + 3) / 4 - (gy + 99) / 100 + (gy + 399) / 400

    for i in range(gm):
        g_day_no += g_days_in_month[i]
    if gm > 1 and ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
        # leap and after Feb
        g_day_no += 1
    g_day_no += gd

    j_day_no = g_day_no - 79

    j_np = j_day_no / 12053
    j_day_no %= 12053
    jy = 957 + 33 * j_np + 4 * int(j_day_no / 1461)

    j_day_no %= 1461

    if j_day_no >= 366:
        jy += (j_day_no - 1) / 365
        j_day_no = (j_day_no - 1) % 365

    for i in range(11):
        if not j_day_no >= j_days_in_month[i]:
            i -= 1
            break
        j_day_no -= j_days_in_month[i]

    jm = i + 2
    jd = j_day_no + 1

    jmonth = jm
    jday = jd
    jyear = jy

    jmonth = str(int(jmonth))
    jday = str(int(jday))
    jyear = str(int(jyear))
    return jday+'/'+jmonth+'/'+jyear
def __jalaliToGregorian(jyear,jmonth,jday):

    global g_days_in_month, j_days_in_month
    jy = jyear - 979
    jm = jmonth - 1
    jd = jday - 1

    j_day_no = 365 * jy + int(jy / 33) * 8 + (jy % 33 + 3) / 4
    for i in range(jm):
        j_day_no += j_days_in_month[i]

    j_day_no += jd
    g_day_no = j_day_no + 79

    gy = 1600 + 400 * int(g_day_no / 146097)  # 146097 = 365*400 + 400/4 - 400/100 + 400/400
    g_day_no = g_day_no % 146097

    leap = 1
    if g_day_no >= 36525:  # 36525 = 365*100 + 100/4
        g_day_no -= 1
        gy += 100 * int(g_day_no / 36524)  # 36524 = 365*100 + 100/4 - 100/100
        g_day_no = g_day_no % 36524

        if g_day_no >= 365:
            g_day_no += 1
        else:
            leap = 0

    gy += 4 * int(g_day_no / 1461)  # 1461 = 365*4 + 4/4
    g_day_no %= 1461

    if g_day_no >= 366:
        leap = 0
        g_day_no -= 1
        gy += g_day_no / 365
        g_day_no = g_day_no % 365

    i = 0
    while g_day_no >= g_days_in_month[i] + (i == 1 and leap):
        g_day_no -= g_days_in_month[i] + (i == 1 and leap)
        i += 1

    gmonth = i + 1
    gday = g_day_no + 1
    gyear = gy

    gmonth = str(int(gmonth))
    gday = str(int(gday))
    gyear = str(int(gyear))
    return gmonth+'/'+gday+'/'+gyear

pddf = pd.read_csv('0_rawData/rfm94.csv')
months = pddf['FinancialDate'].str.slice(start=0,stop=5).unique()
for mah in months :
    pddf1 = pddf[pddf['FinancialDate'].str.slice(start=0,stop=5)==mah]
    temp =[]
    # for index,row in pddf1.iterrows():
    #     print(row['FinancialDate'] , __jalaliToGregorian(int('13'+row['FinancialDate'][:2]),int(row['FinancialDate'][3:5]),int(row['FinancialDate'][6:])))
    pddf1['date'] = pd.to_datetime(pddf1['FinancialDate'].apply(lambda row: __jalaliToGregorian(int('13'+row[:2]),int(row[3:5]),int(row[6:]))))
    pddf1['Amount'] = pd.to_numeric(pddf1['Amount'])
    print(pddf1)
    NOW = max(pddf1['date'])
    rfmTable = pddf1.groupby('Merchantnumber').agg({'date': lambda x: (NOW - x.max()).days,  # Recency
                                                    'FinancialDate': lambda x: len(x),  # Frequency
                                                    'Amount': lambda x: x.sum()})  # Monetary Value
    rfmTable['date'] = rfmTable['date'].astype(int)
    rfmTable.rename(columns={'Merchantnumber': 'Merchantnumber',
                             'date': 'recency',
                             'FinancialDate': 'frequency',
                             'Amount': 'monetary_value'}, inplace=True)
    print('=============================RFM Table',mah,'===================================')
    print(rfmTable.head())
    pd.DataFrame(rfmTable).to_csv('1_rfm/rfmTable'+mah.replace('/','')+'.csv')

print('jj')