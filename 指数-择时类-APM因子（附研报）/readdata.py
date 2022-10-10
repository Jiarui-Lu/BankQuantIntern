import pandas as pd
from WindPy import *

w.isconnected()
w.start()

stock_list = pd.read_excel(r'data\stock_daily_data.xlsx', index_col=0)
result1 = pd.DataFrame()
result2 = pd.DataFrame()
for i in stock_list.columns:
    stock = i.strip('.1')
    error_code, data1 = w.wsi(stock, "open", "2021-09-01 09:30:00", "2022-08-31 15:30:00", "BarSize=30", usedf=True)
    error_code2, data2 = w.wsi(stock, "close", "2021-09-01 09:30:00", "2022-08-31 15:30:00", "BarSize=30", usedf=True)
    data1.columns = [stock]
    data2.columns = [stock]
    result1 = pd.concat([result1, data1], axis=1)
    result2 = pd.concat([result2, data2], axis=1)
    print('{} is done'.format(stock))
result1.to_excel(r'data\open.xlsx')
result2.to_excel(r'data\close.xlsx')
