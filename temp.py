import pandas as pd
xl = pd.ExcelFile("Merchant_tagged_all.xlsx")
df2 = xl.parse("Sheet1")

df2[(df2.MerchantNo == 62001420300002)]
print(df2)