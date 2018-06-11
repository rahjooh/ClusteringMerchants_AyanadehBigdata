import glob , pandas as pd
filelist = glob.glob("raw data/*.xlsx")
t = True
for path in filelist :
    if (path=='raw data\Merchants_Tagged_all_2_960927.xlsx') : continue
    pddf = pd.read_excel(path, sheetname='Sheet1')
    pddf.columns = ['F', 'R', 'M', 'MerchantNo', 'Mwt', 'Fwt', 'Rwt', 'Lable']
    pddf['month'] = path[-14:-11].replace('_','')
    print(path)
    if t :
        pddfall = pddf[:]
        t = False
    else:
        pddfall = pd.concat([pddfall ,pddf],axis=0)
pddfall.to_csv('Merchant_tagged_all.csv')
writer = pd.ExcelWriter('Merchant_tagged_all.xlsx')
pddfall.to_excel(writer,'Sheet1')