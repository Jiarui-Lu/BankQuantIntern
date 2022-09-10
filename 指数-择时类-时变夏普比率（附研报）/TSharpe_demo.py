import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family']='serif' # pd.plot中文
# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('seaborn')
# 查看整体走势
plt.rcParams['font.family'] = 'SimHei'


# 利用 Robert Whitelaw（1997）给出的直接估算模型计算Tsharpe
#  Estimating Sharpe Ratios Directly

def EstimatingSharpeRatiosDirectly():
    '''
    使用1年期国债到期收益率充当无风险收益率
    ------------
    return 返回频率为月度的指标值
    '''
    df = pd.read_excel(r'data\HS300.xlsx', index_col=0)
    idx = df.index
    df['pre_close'] = df['close'].shift(1)
    df = df.dropna()

    ret = np.log(df['close'] / df['pre_close'])

    m_r = ret.groupby(pd.Grouper(freq='M')).sum()

    # 无风险收益率设置
    r_f = 0.03

    r_e = m_r - r_f

    # 计算月度标准差
    m_std = ret.groupby(
        pd.Grouper(freq='M')).apply(lambda x: np.sqrt(x @ x))

    ESRD = r_e / m_std

    return ESRD, df['close']


# 使用直接估算模型取得share
ESRD, price_df = EstimatingSharpeRatiosDirectly()

# print(ESRD)

# # 月度收盘价
tmp = list(price_df.groupby(pd.Grouper(freq='M')))
M_close = []
# 元组第一个元素是分组的label，第二个是dataframe
for tuple in tmp:
    M_close.append(tuple[1][-1])
M_close = pd.Series(M_close, index=ESRD.index)


# print(M_close)


# M_close.plot(figsize=(18,8),title='沪深300指数与时变夏普比率',color='black',ls='--')
# plt.twinx()
# ESRD.plot(color='red',alpha=0.5)
# plt.savefig(r'result\沪深300指数与时变夏普比率.jpg')
# plt.show()

def signal_generation(ESRD):
    df = pd.DataFrame(index=ESRD.index)
    df = pd.concat([df, M_close], axis=1)
    df.columns = ['close']
    df['ret'] = df['close'].pct_change()
    df['positions'] = 0
    df['signals'] = 0
    df['positions'] = np.where(ESRD > 0.3, 1, 0)
    for i, e in enumerate(ESRD):
        if df['positions'][i] == 1 and -0.3 <= e <= 0.3:
            df['positions'] = 1
    df['signals'] = df['positions'].diff()
    return df.dropna()


def plot(signals, column):
    # we have to do a lil bit slicing to make sure we can see the plot clearly
    # the only reason i go to -3 is that day we execute a trade
    # give one hour before and after market trading hour for as x axis
    # date = pd.to_datetime(intraday['date']).iloc[-3]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # mostly the same as other py files
    # the only difference is to create an interval for signal generation
    ax.plot(signals.index, signals['close'], label=column)
    ax.plot(signals.loc[signals['signals'] == 1].index, signals[column][signals['signals'] == 1], lw=0, marker='^',
            markersize=10, c='g', label='LONG')
    ax.plot(signals.loc[signals['signals'] == -1].index, signals[column][signals['signals'] == -1], lw=0, marker='v',
            markersize=10, c='r', label='SHORT')
    #
    # change legend text color
    lgd = plt.legend(loc='best').get_texts()
    for text in lgd:
        text.set_color('#6C5B7B')
    #
    plt.ylabel(column)
    plt.xlabel('Date')
    plt.title('ESRD')
    plt.grid(True)
    plt.savefig(r'result\Position_Strategy.jpg')
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
    col_name = ['ESRD']

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
    summary_df.to_excel(r'result\Strategy summary.xlsx')
    print(summary_df)


sig = signal_generation(ESRD)
plot(sig, 'close')
summary(sig)
