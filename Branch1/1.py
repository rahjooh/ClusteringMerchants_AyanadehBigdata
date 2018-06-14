import pandas as pd

#tabeston = pd.read_csv('haditabestan95.csv')

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

print(__jalaliToGregorian(1395,7,1))
print(__gregorianToJalali(2017,11,13))

import datetime as dt
NOW = dt.datetime(2016,9,22)

#tabeston['FinancialDate'] = pd.to_datetime(tabeston['FinancialDate'])
df =[]
file= open('0_rawData/haditabestan95.csv')
header = file.readline().strip().split(',')

c= 0
for line in file:
    c+=1
    if c %100000==0 :print(c)
    line=line.strip().split(',')
    t = __jalaliToGregorian(int('13'+line[3][:2]),int(line[3][3:5]),int(line[3][6:]))
    df.append([line[0],line[1],line[2],t])

pd_DF = pd.DataFrame(df)
pd_DF.columns = header
pd_DF['FinancialDate']=pd.to_datetime(pd_DF['FinancialDate'])
print('done')
pd_DF['Amount']=pd.to_numeric(pd_DF['Amount'])
print('done done')
pd_DF.to_csv('0_rawData/hadiSummer95.csv',index=False)
print('done done done')

rfmTable = pd_DF.groupby('Merchantnumber').agg( {'FinancialDate': lambda x: (NOW - x.max()).days, # Recency
                                                 'id': lambda x: len(x),      # Frequency
                                                 'Amount': lambda x: x.sum()}) # Monetary Value


rfmTable['FinancialDate'] = rfmTable['FinancialDate'].astype(int)
print(rfmTable)
rfmTable.rename(columns={'Merchantnumber':'Merchantnumber',
                         'FinancialDate': 'recency',
                         'id': 'frequency',
                         'Amount': 'monetary_value'}, inplace=True)

print('=============================RFM Table===================================')
print(rfmTable.head())
pd.DataFrame(rfmTable).to_csv('1_rfm/rfmTable.csv')

