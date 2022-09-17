import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

hs300 = pd.read_excel(r'data\HS300.xlsx', index_col=0)
hs300['ATR'] = (hs300['high'] - hs300['low']).rolling(window=100).mean()
hs300['ma1'] = hs300['close'].rolling(window=12).mean()
hs300['ma2'] = hs300['close'].rolling(window=26).mean()
hs300['dif'] = hs300['ma1'] - hs300['ma2']
hs300['dea'] = hs300['dif'].rolling(window=9).mean()
hs300 = hs300.dropna()

fig = plt.figure()
ax1 = fig.add_subplot(111)
lin1 = ax1.plot(hs300.index, hs300['close'], label='收盘价', color='r')
ax1.set_title('收盘价与DIF,DEA比对')
ax1.set_xlabel('时间')
ax1.set_ylabel('收盘价')
ax2 = ax1.twinx()
lin2 = ax2.plot(hs300.index, hs300['dif'], label='dif')
lin3 = ax2.plot(hs300.index, hs300['dea'], label='dea')
ax2.set_ylabel('均线指标')
lins = lin1 + lin2 + lin3
labs = [l.get_label() for l in lins]
ax1.legend(lins, labs, loc='best')
plt.savefig(r'result\收盘价与DIF,DEA比对.jpg')
plt.show()


def Dir(df, rate):
    df['dir'] = 0
    df['delta'] = rate * df['ATR']
    df['integral'] = 0
    df['diff'] = df['dif'] - df['dea']
    diff_index = list(df.columns).index('diff')
    integral_index = list(df.columns).index('integral')
    _index = 0
    while True:
        tmp_sum = 0
        while df.iloc[_index, diff_index] < 0:
            tmp_sum = tmp_sum + df.iloc[_index, diff_index]
            df.iloc[_index, integral_index] = tmp_sum
            _index = _index + 1
        tmp_sum = 0
        while df.iloc[_index, diff_index] > 0:
            tmp_sum = tmp_sum + df.iloc[_index, diff_index]
            df.iloc[_index, integral_index] = tmp_sum
            _index = _index + 1
        if _index + 1 == len(df):
            break
    if df.iloc[-1, diff_index] * df.iloc[-2, diff_index] < 0:
        df.iloc[-1, integral_index] = df.iloc[-1, diff_index]
    else:
        df.iloc[-1, integral_index] = df.iloc[-1, diff_index] + df.iloc[-2, integral_index]
    df['dir'] = np.where(df['integral'] >= df['delta'], 1, -1)
    return df


new_df = Dir(hs300, 2)

slicer = new_df[-150:]
fig = plt.figure()
ax1 = fig.add_subplot(111)
lin1 = ax1.plot(slicer.index, slicer['close'], label='收盘价', color='r')
ax1.set_title('上下行比对')
ax1.set_xlabel('时间')
ax1.set_ylabel('收盘价')
ax2 = ax1.twinx()
lin2 = ax2.plot(slicer.index, slicer['dir'], label='划分上下行')
ax2.set_ylabel('value')
lins = lin1 + lin2
labs = [l.get_label() for l in lins]
ax1.legend(lins, labs, loc='best')
plt.savefig(r'result\上下行比对.jpg')
plt.show()

new_df['ret'] = new_df['close'].pct_change()
new_df['positions'] = np.where(new_df['dir'] == 1, 1, 0)
new_df['signals'] = new_df['positions'].diff()
new_df = new_df.dropna()


def plot(signals):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # mostly the same as other py files
    # the only difference is to create an interval for signal generation
    ax.plot(signals.index, signals['close'], label='close')
    ax.plot(signals.loc[signals['signals'] == 1].index, signals['close'][signals['signals'] == 1], lw=0, marker='^',
            markersize=10, c='g', label='LONG')
    ax.plot(signals.loc[signals['signals'] == -1].index, signals['close'][signals['signals'] == -1], lw=0, marker='v',
            markersize=10, c='r', label='SHORT')
    #
    # change legend text color
    lgd = plt.legend(loc='best').get_texts()
    for text in lgd:
        text.set_color('#6C5B7B')
    #
    plt.ylabel('close')
    plt.xlabel('Date')
    plt.title('多空策略')
    plt.grid(True)
    plt.savefig(r'result\多空策略.jpg')
    plt.show()


# 风险报告
def summary(df):
    summary_dic = {}
    index_name = '年化收益率,累计收益率,累计超额收益率,夏普比率,最大回撤,持仓总天数,交易次数,' \
                 '平均持仓天数,获利天数,亏损天数,胜率(按天),平均盈利率(按天),平均亏损率(按天),' \
                 '平均盈亏比(按天),盈利次数,亏损次数,单次最大盈利,单次最大亏损,' \
                 '胜率(按此),平均盈利率(按次),平均亏损率(按次),平均盈亏比(按次)'.split(
        ',')
    signal_name = ['signals']
    col_name = ['点位效率理论']

    def format_x(x):
        return '{:.2%}'.format(x)

    for signal in signal_name:
        RET = df['ret'] * df[signal]
        CUM_RET = (1 + RET).cumprod()

        # 计算年华收益率
        annual_ret = CUM_RET[-1] ** (250 / len(RET)) - 1

        # 计算累计收益率
        cum_ret_rate = CUM_RET[-1] - 1

        # 计算累计超额收益率
        alpha_ret = CUM_RET - (1 + df['ret']).cumprod()
        alpha_ret_rate = alpha_ret[-1]

        # 最大回撤
        max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
        mdd = -np.min(CUM_RET / max_nv - 1)

        # 夏普
        sharpe_ratio = np.mean(RET) / np.nanstd(RET, ddof=1) * np.sqrt(250)

        # 标记买入卖出时点
        mark = df[signal]
        pre_mark = np.nan_to_num(df[signal].shift(-1))
        # 买入时点
        trade = (mark == 1) & (pre_mark < mark)

        # 交易次数
        trade_count = np.nansum(trade)

        # 持仓总天数
        total = np.sum(mark)
        # 平均持仓天数
        mean_hold = total / trade_count
        # 获利天数
        win = np.sum(np.where(RET > 0, 1, 0))
        # 亏损天数
        lose = np.sum(np.where(RET < 0, 1, 0))
        # 胜率
        win_ratio = win / total
        # 平均盈利率（天）
        mean_win_ratio = np.sum(np.where(RET > 0, RET, 0)) / win
        # 平均亏损率（天）
        mean_lose_ratio = np.sum(np.where(RET < 0, RET, 0)) / lose
        # 盈亏比(天)
        win_lose = win / lose

        # 盈利次数
        temp_df = df.copy()
        diff = temp_df[signal] != temp_df[signal].shift(1)
        temp_df['mark'] = diff.cumsum()
        # 每次开仓的收益率情况
        temp_df = temp_df.query(signal + '==1').groupby('mark')['ret'].sum()

        # 盈利次数
        win_count = np.sum(np.where(temp_df > 0, 1, 0))
        # 亏损次数
        lose_count = np.sum(np.where(temp_df < 0, 1, 0))
        # 单次最大盈利
        max_win = np.max(temp_df)
        # 单次最大亏损
        max_lose = np.min(temp_df)
        # 胜率
        win_rat = win_count / len(temp_df)
        # 平均盈利率（次）
        mean_win = np.sum(np.where(temp_df > 0, temp_df, 0)) / len(temp_df)
        # 平均亏损率（天）
        mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0)) / len(temp_df)
        # 盈亏比(次)
        mean_wine_lose = win_count / lose_count

        summary_dic[signal] = [format_x(annual_ret), format_x(cum_ret_rate), format_x(alpha_ret_rate), sharpe_ratio,
                               format_x(
                                   mdd), total, trade_count, mean_hold, win, lose, format_x(win_ratio),
                               format_x(mean_win_ratio),
                               format_x(mean_lose_ratio), win_lose, win_count, lose_count, format_x(
                max_win), format_x(max_lose),
                               format_x(win_rat), format_x(mean_win), format_x(mean_lose), mean_wine_lose]

    summary_df = pd.DataFrame(summary_dic, index=index_name)
    summary_df.columns = col_name
    summary_df.to_excel(r'result\风险报告.xlsx')
    print(summary_df)


plot(new_df)
summary(new_df)
