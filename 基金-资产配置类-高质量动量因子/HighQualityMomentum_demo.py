import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')

close_data = pd.read_excel(r'data\stock.xlsx', index_col=0, sheet_name=3).dropna(axis=1)

pct_df = close_data.pct_change().iloc[1:]

month = pd.read_excel(r'data\monthly_data.xlsx', index_col=0)

momentum_factor = pd.DataFrame()
for m in pct_df.index:
    if m in month.index:
        m_index = list(pct_df.index).index(m)
        if m_index >= 60:
            df_tmp = pct_df.iloc[m_index - 60:m_index, :]
        else:
            df_tmp = pct_df.iloc[:m_index, :]
        ret = df_tmp.iloc[-1] / df_tmp.iloc[0] - 1
        std_df = df_tmp.std()
        momentum = ret - 3000 * std_df ** 2
        momentum_factor = pd.concat([momentum_factor, pd.DataFrame(momentum)], axis=0)
# print(momentum_factor)

factor_normal = pd.DataFrame(0, index=[i for i in range(len(month.index) * len(pct_df.columns))],
                             columns=['date', 'code'])
_index = 0
factor_normal['momentum_factor'] = list(momentum_factor[momentum_factor.columns[0]])
for m in month.index:
    for c in pct_df.columns:
        factor_normal.loc[_index, 'next_ret'] = month.loc[m, c]
        factor_normal.loc[_index, 'date'] = m
        factor_normal.loc[_index, 'code'] = c
        _index = _index + 1
index = pd.MultiIndex.from_frame(factor_normal.loc[:, ['date', 'code']])
factor_normal.index = index
del factor_normal['date']
del factor_normal['code']
factor_normal.to_excel(r'result\factor normal.xlsx')


def get_group(df: pd.DataFrame, target_factor: str, num_group: int = 5) -> pd.DataFrame:
    '''
    分组
    ----
        target_factor:目标列
        num_group:默认分5组
    '''
    df = df.copy()
    label = [i for i in range(1, num_group + 1)]
    df['group'] = df.groupby(level='date')[target_factor].transform(
        lambda x: pd.qcut(x, 5, labels=label))

    return df


def get_algorithm_return(factor_df: pd.DataFrame) -> pd.DataFrame:
    '''
    获取分组收益率
    ---------
        传入df数据结构
           ----------------------------------------------
                      |       |factor|group| next_return|
           ----------------------------------------------
               date   | asset |      |     |            |
           ----------------------------------------------
                      | AAPL  |  0.5 |  G1 |   0.23     |
                      -----------------------------------
                      | BA    | -1.1 |  G2 |   -0.7     |
                      -----------------------------------
           2014-01-01 | CMG   |  1.7 |  G2 |   0.023    |
                      -----------------------------------
                      | DAL   | -0.1 |  G3 |   -0.03    |
                      -----------------------------------
                      | LULU  |  2.7 |  G1 |   -0.21    |
                      -----------------------------------
    '''

    returns = pd.pivot_table(factor_df.reset_index(
    ), index='date', columns='group', values='next_ret')
    returns.columns = [str(i) for i in returns.columns]
    returns.index = pd.to_datetime(returns.index)
    label = ['G%s' % i for i in range(1, 6)]
    returns.columns = label

    return returns


group_df1 = get_group(factor_normal, 'momentum_factor')
returns1 = get_algorithm_return(group_df1)
# print(returns1)
benchmark = pd.read_excel(r'data\HS300.xlsx', index_col=0)
benchmark = benchmark.reindex(returns1.index)
benchmark = benchmark['close'].pct_change().fillna(0)
returns1['benchmark'] = benchmark

returns1['excess_ret'] = returns1['G5'] - returns1['G1']
cum_df1 = np.exp(np.log1p(returns1).cumsum())

nav = cum_df1
fig = plt.figure()
plt.plot(nav.index, nav['G1'], color='Navy', label='G1')
plt.plot(nav.index, nav['G2'], color='LightGrey', ls='-.', label='G2')
plt.plot(nav.index, nav['G3'], color='DimGray', ls='-.', label='G3')
plt.plot(nav.index, nav['G4'], color='DarkKhaki', ls='-.', label='G4')
plt.plot(nav.index, nav['G5'], color='LightSteelBlue', label='G5')
plt.axhline(1, color='black', lw=0.5)
# 多空单独反应
plt.plot(nav.index, nav['excess_ret'], color='r', ls='--', label='excess_ret')
plt.plot(nav.index, nav['benchmark'], color='black', ls='--', label='benchmark')
plt.legend(loc='best')
plt.savefig(r'result\momentum_factor.jpg')
plt.grid(True)
plt.show()
