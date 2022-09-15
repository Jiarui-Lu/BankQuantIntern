import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['font.family'] = ['SimHei']

HS300 = pd.read_excel(r'data\HS300.xlsx', index_col=0)
HS300['pre_close'] = HS300['close'].shift(1)
HS300 = HS300.dropna()
std_window = 22
mean_window = 10

# 计算收益率
ret = HS300['close'] / HS300['pre_close'] - 1
# 收益率标准差
ret_std = ret.rolling(std_window).std().dropna()
ret_mean = ret_std.rolling(mean_window).mean().dropna()

# 计算振幅
amplitude = (HS300['high'] - HS300['low']) / HS300['pre_close']

# 计算振幅标准差
amplitude_std = amplitude.rolling(std_window).std().dropna()
amplitude_mean = amplitude_std.rolling(mean_window).mean().dropna()

# 观察期
begin_date = '2006-01-01'
watch_date = '2018-12-31'

# 统一观察窗口
amplitude_std = amplitude_mean.loc[begin_date:watch_date]
amplitude_mean = amplitude_std.loc[begin_date:watch_date]
ret_std = ret_std.loc[begin_date:watch_date]
ret_mean = ret_mean.loc[begin_date:watch_date]


# 获取granger因果检验结果的p值显示
def grangercausalitytests_pvalue(ret: pd.DataFrame, singal: pd.DataFrame, title: str):
    result = grangercausalitytests(
        np.c_[ret.reindex(singal.index), singal], maxlag=31, verbose=False)

    p_value = []
    for i, items_value in result.items():
        p_value.append(items_value[0]['params_ftest'][1])

    plt.figure(figsize=(18, 6))
    plt.title(title)
    plt.bar(range(len(p_value)), p_value, width=0.4)
    plt.xticks(range(len(p_value)), np.arange(1, 32, 1))
    plt.axhline(0.5, ls='--', color='black', alpha=0.5, label='p值0.05显著水平')
    plt.legend()
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()


# 检验信号与滞后期收益率的相关系数
def show_corrocef(close_df: pd.DataFrame, singal: pd.DataFrame, title: str):
    period = np.arange(1, 32, 1)  # 滞后周期间隔

    temp = []  # 储存数据

    for i in period:
        # 收益未来收益与信号的相关系数
        lag_ret = close_df['close'].pct_change(i).shift(-i)
        temp.append(
            np.corrcoef(lag_ret.reindex(singal.index), singal)[0][1])

    plt.figure(figsize=(18, 6))
    plt.title(title)
    plt.bar(range(len(temp)), temp, width=0.4)
    plt.xticks(range(len(temp)), period)
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()


show_corrocef(HS300, ret_mean, "收益率标准差均值与未来收益的关系")
show_corrocef(HS300, amplitude_std, "振幅标准差均值与未来收益的关系")
grangercausalitytests_pvalue(ret, ret_mean, '收益率波动率均值与收益率的granger检验的p值')
grangercausalitytests_pvalue(ret, amplitude_mean, '振幅波动率均值与收益率的granger检验的p值')

# 振幅剪刀差
Upward_volatility = HS300['high'] / HS300['open'] - 1
Downside_volatility = 1 - HS300['low'] / HS300['open']

diff_vol = Upward_volatility - Downside_volatility

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1)
ax1 = sns.distplot(Upward_volatility.dropna(), label='上行波动率')
ax1 = sns.distplot(Downside_volatility.dropna(), label='下行波动率')
plt.legend()

ax2 = fig.add_subplot(1, 3, 2)
ax2 = sns.distplot(Upward_volatility.dropna(), color='g', label='上行波动率')
plt.legend()

ax3 = fig.add_subplot(1, 3, 3)
ax3 = sns.distplot(Downside_volatility.dropna(), label='下行波动率')
plt.legend()
plt.savefig(r'result\振幅剪刀差上行和下行波动率.jpg')
plt.show()

show_corrocef(HS300, diff_vol.loc[begin_date:watch_date], "振幅剪刀差与未来收益的关系")

# 收益率剪刀差
Upward_volatility = np.where(ret.loc[begin_date:watch_date] > 0, ret_std, 0)
Downside_volatility = np.where(ret.loc[begin_date:watch_date] < 0, ret_std, 0)

diff_vol = Upward_volatility - Downside_volatility

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1)
ax1 = sns.distplot(Upward_volatility, label='上行波动率')
ax1 = sns.distplot(Downside_volatility, label='下行波动率')
plt.legend()

ax2 = fig.add_subplot(1, 3, 2)
ax2 = sns.distplot(Upward_volatility, color='g', label='上行波动率')
plt.legend()

ax3 = fig.add_subplot(1, 3, 3)
ax3 = sns.distplot(Downside_volatility, label='下行波动率')
plt.legend()
plt.savefig(r'result\收益率剪刀差上行和下行波动率.jpg')
plt.show()

show_corrocef(HS300, pd.Series(diff_vol, ret_std.index), "收益率标准差刀差与未来收益的关系")

# 振幅剪刀差
Upward_volatility = HS300['high'] / HS300['open'] - 1
Downside_volatility = 1 - HS300['low'] / HS300['open']

diff_vol = Upward_volatility - Downside_volatility

diff_ma = diff_vol.rolling(60).mean()

'''
strategy 1:
单向波动差移动平均为正：买入
单向波动差移动平均为负：卖出
'''

flag = np.where(diff_ma.loc[begin_date:watch_date] > 0, 1, 0)
slice_ser = ret.shift(-1).loc[begin_date:watch_date]
slice_benchmark = HS300.loc[begin_date:watch_date, 'close']

strategy_ret = flag * slice_ser
strategy_cum = (1 + strategy_ret).cumprod()
benchmark = slice_benchmark / slice_benchmark[0]

plt.figure(figsize=(18, 8))
plt.title('波动剪刀差策略与沪深 300净值')
strategy_cum.plot(label='strategy1')
benchmark.plot(color='r', ls='--', label='HS300', alpha=0.5)
plt.legend(loc='best')
plt.savefig(r'result\振幅剪刀差策略收益.jpg')
plt.show()

excess_ret = strategy_ret - ret.loc[begin_date:watch_date]
excess_cum = (1 + excess_ret).cumprod()

show_excess = excess_cum.groupby(
    pd.Grouper(freq='Y')).apply(lambda x: pow(x[-1] / x[0], 244 / len(x)) - 1)

plt.figure(figsize=(10, 6))
plt.title('相对强弱RPS值与指数历史趋势')

plt.bar(range(len(show_excess)), show_excess.values)
plt.xticks(range(len(show_excess)), ['%s年' % x.strftime('%Y') for x in show_excess.index])
plt.savefig(r'result\相对强弱RPS值与指数历史趋势.jpg')
plt.show()

'''
strategy 1:
单向波动差移动平均为正：买入
单向波动差移动平均为负：卖出
'''

diff_ma10 = diff_vol.rolling(10).mean()

flag_1 = np.where(diff_ma10.loc[begin_date:watch_date] > 0, 1, 0)
slice_ser = ret.shift(-1).loc[begin_date:watch_date]

strategy_ret1 = flag_1 * slice_ser
strategy_cum1 = (1 + strategy_ret1).cumprod()

plt.figure(figsize=(18, 8))
plt.title('10天与60天移动平均波动差值策略净值对比')
strategy_cum.plot(label='strategy_ma60')
strategy_cum1.plot(label='strategy_ma1')
benchmark.plot(color='r', ls='--', alpha=0.5, label='HS300')
plt.legend(loc='best')
plt.savefig(r'result\10天与60天移动平均波动差值策略净值对比.jpg')
plt.show()
