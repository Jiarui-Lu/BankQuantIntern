import time
import pandas as pd
import numpy as np

test1 = pd.read_csv(r'wind-1\ashareeodprices.csv', index_col=2)
# test2=pd.read_csv(r'wind-1\AShareEODDerivativeIndicator.csv',index_col=2)
raw_data = pd.read_csv(r'industry_zx_CN_STOCK_A\industry_zx_CN_STOCK_A.csv', index_col=0)


# test2 = test2.set_index('S_INFO_WINDCODE', append=True)
def ConcatDf(df1, df2, df1_col_list):
    df1 = df1.set_index('S_INFO_WINDCODE', append=True)
    df2['instrument'] = df2['instrument'].apply(lambda x: x.strip('A'))
    df2.index = [int(str(i).replace('-', '')) for i in df2.index]
    df2 = df2.set_index('instrument', append=True)
    new_df = pd.DataFrame()
    for date, code in df2.index:
        try:
            tmp = (df1.loc[date, :]).loc[code, df1_col_list]
        except:
            tmp = pd.Series(np.nan, index=df1_col_list)
        new_df = pd.concat([new_df, tmp], axis=1)
    new_df = new_df.T
    new_df = new_df.reset_index()
    df2 = df2.reset_index()
    df2[new_df.columns] = new_df[new_df.columns]
    return df2


df1_col_list = ["S_DQ_PRECLOSE", "S_DQ_OPEN"]
time_start = time.time()
df2 = ConcatDf(test1, raw_data, df1_col_list)
print(df2)
time_end = time.time()
time_dur = time_end - time_start
print(time_dur)
