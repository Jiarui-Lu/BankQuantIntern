import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dealAmount = pd.read_csv(
    r'data\data2013_2020.csv', index_col=0, parse_dates=['tradeDate'])

dealAmount = pd.pivot_table(dealAmount, index='tradeDate', columns='secID', values='dealAmount')
# dealAmount=dealAmount.drop(columns=dealAmount.columns[-3000:])

#
money_df = pd.read_csv(r'data\money_df.csv', index_col=0, parse_dates=[0])
# money_df=money_df.drop(columns=money_df.columns[-3000:])
close_df = pd.read_csv(r'data\close_df.csv', index_col=0, parse_dates=[0])
# close_df=close_df.drop(columns=close_df.columns[-3000:])

pre_close = pd.read_csv(r'data\pre_close.csv', index_col=0, parse_dates=[0])
next_ret = (close_df / pre_close - 1)
amt_df = money_df / dealAmount  # 计算平均单笔成交金额
ret_df = close_df / pre_close - 1  # 当日涨跌幅


def cala_w_factor(df1: pd.DataFrame, df2: pd.DataFrame, win_size: int) -> pd.Series:
    '''
    df1,df2:数据需要对齐 以df1为基准对齐
    df1,
    '''

    # 数据对齐
    df1, df2 = df1.align(df2, join='right')

    # rolling
    iidx = np.arange(len(df1))
    shape = (iidx.size - win_size + 1, win_size)
    strides = (iidx.strides[0], iidx.strides[0])
    res = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)

    # 因子计算
    def _cal_m(df1: pd.DataFrame, df2: pd.DataFrame, res: list) -> pd.Series:
        rank_df = df1.iloc[res].rank()
        cond = (rank_df >= 11)
        m_high = cond * df2.iloc[res]
        m_low = ~cond * df2.iloc[res]

        m_ser = m_high.sum() - m_low.sum()
        m_ser.name = rank_df.index[-1]

        return m_ser

    return pd.concat([_cal_m(df1, df2, i) for i in res], axis=1).T


M_factor = cala_w_factor(amt_df, ret_df, 20)
M_factor = M_factor.replace(0, np.nan)
M_factor = M_factor.dropna(axis=1)
M_factor = M_factor.drop(columns=M_factor.columns[50:])
next_ret = next_ret.loc[M_factor.index, M_factor.columns]
next_ret = pd.DataFrame(next_ret.stack(), columns=['next_ret'])
# print(next_ret)
factor_df = pd.DataFrame(index=M_factor.index, columns=[str(i) for i in range(1, 6)])
M_factor = pd.DataFrame(M_factor.stack(), columns=['w_factor'])
M_factor['next_ret'] = next_ret['next_ret']
M_factor['qcut'] = pd.qcut(M_factor['w_factor'], q=5, labels=[str(i) for i in range(1, 6)])
M_factor = M_factor.reset_index()
M_factor.columns = ['date', 'code', 'w_factor', 'next_ret', 'qcut']
M_factor = M_factor.set_index('date')
M_factor = M_factor.set_index('qcut', append=True)
for date, q in M_factor.index:
    date_df = M_factor.loc[date, :]
    for i in [str(i) for i in range(1, 6)]:
        try:
            factor_df.loc[date, i] = date_df.loc[i, 'next_ret'].mean()
        except:
            pass
factor_df.to_excel(r'result\factor_df.xlsx')

fig = plt.figure(figsize=(16, 8))
for col in factor_df.columns:
    cum_return = (1 + factor_df[col]).cumprod()
    plt.plot(factor_df.index, cum_return, label=col)
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('return')
plt.title('Strategy return')
plt.savefig(r'result\Strategy return.jpg')
plt.show()
