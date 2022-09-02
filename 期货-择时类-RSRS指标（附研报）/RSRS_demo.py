import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import matplotlib.dates as mdate

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')

# 忽略报错
import warnings

warnings.filterwarnings("ignore")


def CalRSRS(df, N):
    df = df.copy()
    # 填充空缺,注意这里填充的是18，t日计算的信号，在T+1日，所以收益不需要在滞后一期
    temp = [np.nan] * N

    for row in range(len(df) - N):
        y = df['high'][row:row + N]
        x = df['low'][row:row + N]

        # 计算系数
        beta = np.polyfit(x, y, 1)[0]

        temp.append(beta)

    df['RSRS'] = temp
    return df


def stat_depict_plot(df, col, title):
    df = df[~df[col].isna()].copy()

    avgRet = np.mean(df[col])
    medianRet = np.median(df[col])
    stdRet = np.std(df[col])
    skewRet = st.skew(df[col])
    kurtRet = st.kurtosis(df[col])

    plt.style.use('ggplot')
    # 画日对数收益率分布直方图
    fig = plt.figure(figsize=(18, 9))
    plt.suptitle(title)
    v = df[col]
    x = np.linspace(avgRet - 3 * stdRet, avgRet + 3 * stdRet, 100)
    y = st.norm.pdf(x, avgRet, stdRet)
    kde = st.gaussian_kde(v)

    # plot the histogram
    plt.subplot(121)
    plt.hist(v, 50, weights=np.ones(len(v)) / len(v), alpha=0.4)
    plt.axvline(x=avgRet, color='red', linestyle='--',
                linewidth=0.8, label='Mean Count')
    plt.axvline(x=avgRet - stdRet, color='blue', linestyle='--',
                linewidth=0.8, label='-1 Standard Deviation')
    plt.axvline(x=avgRet + stdRet, color='blue', linestyle='--',
                linewidth=0.8, label='1 Standard Deviation')
    plt.ylabel('Percentage', fontsize=10)
    plt.legend(fontsize=12)

    # plot the kde and normal fit
    plt.subplot(122)
    plt.plot(x, kde(x), label='Kernel Density Estimation')
    plt.plot(x, y, color='black', linewidth=1, label='Normal Fit')
    plt.ylabel('Probability', fontsize=10)
    plt.axvline(x=avgRet, color='red', linestyle='--',
                linewidth=0.8, label='Mean Count')
    plt.legend(fontsize=12)
    plt.savefig(r'result\{}.jpg'.format(title))
    return plt.show()


# 低阶距统计描述
def stat_depict(df, col):
    df = df[~df[col].isna()].copy()
    # 计算总和的统计量
    avgRet = np.mean(df[col])
    medianRet = np.median(df[col])
    stdRet = np.std(df[col])
    skewRet = st.skew(df[col])
    kurtRet = st.kurtosis(df[col])
    result = pd.DataFrame([avgRet, medianRet, stdRet, skewRet, kurtRet, avgRet + stdRet, avgRet - stdRet],
                          index=['平均数', '中位数', '标准差', '偏度', '峰度', '1 Standard Deviation', '-1 Standard Deviation']
                          )
    result.to_excel(r'result\statistics description.xlsx')
    print(result)


# 构造标准分RSRS

def Cal_RSRS_Zscore(df, M):
    df['RSRS_temp'] = df['RSRS'].fillna(0)
    # df = Cal_RSRS(df, N)  # 计算基础斜率
    ZSCORE = (df['RSRS_temp'] - df['RSRS_temp'].rolling(M).mean()
              ) / df['RSRS_temp'].rolling(M).std()
    df['RSRS_Z'] = ZSCORE
    df = df.drop(columns='RSRS_temp')
    return df


# 择时指标回测
def RSRS_Strategy(RSRS_Z, S1, S2):
    print('回测起始日：', min(RSRS_Z.index))
    # 基础信号回测
    basic_singal = []
    for row in range(len(RSRS_Z)):

        if RSRS_Z['RSRS'][row] > S2:
            basic_singal.append(1)
        else:
            if row != 0:
                if basic_singal[-1] and RSRS_Z['RSRS'][row] > S1:
                    basic_singal.append(1)
                else:
                    basic_singal.append(0)
            else:
                basic_singal.append(0)

    # 储存基础信号
    RSRS_Z['basic_singal'] = basic_singal

    # 计算标准信号，S=0.7 研报给出的 我都不知到怎么来的
    z_singal = []
    S = 0.7
    for row in range(len(RSRS_Z)):

        if RSRS_Z['RSRS_Z'][row] > S:
            z_singal.append(1)

        else:
            if row != 0:
                if z_singal[-1] and RSRS_Z['RSRS_Z'][row] > -S:
                    z_singal.append(1)
                else:
                    z_singal.append(0)
            else:
                z_singal.append(0)

    # 储存标准分信号
    RSRS_Z['z_singal'] = z_singal

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 斜率净值
    BASIC_CUM = (1 + RSRS_Z['basic_singal'] * RSRS_Z['ret']).cumprod()
    # 标准分净值
    Z_CUM = (1 + RSRS_Z['z_singal'] * RSRS_Z['ret']).cumprod()
    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(BASIC_CUM, label='斜率指标策略')
    ax1.plot(Z_CUM, label='标准分指标策略')
    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('RSRS指标策略净值曲线')
    plt.savefig(r'result\Stategy backtest.jpg')
    plt.show()
    return RSRS_Z


# 风险报告
def summary(df, singal_name=['basic_singal', 'z_singal']):
    summary_dic = {}
    index_name = '年化收益率,累计收益率,夏普比率,最大回撤,持仓总天数,交易次数,平均持仓天数,获利天数, \
    亏损天数,胜率(按天),平均盈利率(按天),平均亏损率(按天),平均盈亏比(按天),盈利次数,亏损次数, \
    单次最大盈利,单次最大亏损,胜率(按此),平均盈利率(按次),平均亏损率(按次),平均盈亏比(按次)'.split(
        ',')

    col_dic = dict(zip(['RSRS_singal', 'RSRS_Z_singal', 'RSRS_Revise_singal', 'RSRS_Positive_singal']
                       , ['斜率指标策略', '标准分指标策略', '修正标准分策略', '右偏标准分策略']))

    # 判断是否是默认的singal_name
    if singal_name[0] in col_dic:
        col_name = [col_dic[x] for x in singal_name]
    else:
        col_name = '斜率指标策略,标准分指标策略'.split(',')

    def format_x(x):
        return '{:.2%}'.format(x)

    for singal in singal_name:
        RET = df['ret'] * df[singal]
        CUM_RET = (1 + RET).cumprod()

        # 计算年华收益率
        annual_ret = CUM_RET[-1] ** (250 / len(RET)) - 1

        # 计算累计收益率
        cum_ret_rate = CUM_RET[-1] - 1

        # 最大回撤
        max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
        mdd = -np.min(CUM_RET / max_nv - 1)

        # 夏普
        sharpe_ratio = np.mean(RET) / np.nanstd(RET, ddof=1) * np.sqrt(250)

        # 标记买入卖出时点
        mark = df[singal]
        pre_mark = np.nan_to_num(df[singal].shift(-1))
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
        diff = temp_df[singal] != temp_df[singal].shift(1)
        temp_df['mark'] = diff.cumsum()
        # 每次开仓的收益率情况
        temp_df = temp_df.query(singal + '==1').groupby('mark')['ret'].sum()

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

        summary_dic[singal] = [format_x(annual_ret), format_x(cum_ret_rate), sharpe_ratio, format_x(
            mdd), total, trade_count, mean_hold, win, lose, format_x(win_ratio), format_x(mean_win_ratio),
                               format_x(mean_lose_ratio), win_lose, win_count, lose_count, format_x(
                max_win), format_x(max_lose),
                               format_x(win_rat), format_x(mean_win), format_x(mean_lose), mean_wine_lose]

    summary_df = pd.DataFrame(summary_dic, index=index_name)
    summary_df.columns = col_name
    summary_df.to_excel(r'result\Strategy summary.xlsx')
    print(summary_df)


def main():
    df = pd.read_excel(r'data\HS300.xlsx', index_col=0)
    new_df = CalRSRS(df, 18).dropna()
    plt.figure(figsize=(18, 8))
    plt.title('沪深300各时期斜率均值')
    plt.plot(new_df['RSRS'].rolling(250).mean())
    plt.savefig(r'result\Slope data.jpg')
    stat_depict_plot(new_df, 'RSRS', 'Slope distribution')
    # 计算标准分斜率
    RSRS_Z = Cal_RSRS_Zscore(new_df, 600).dropna()
    stat_depict_plot(RSRS_Z, 'RSRS_Z', 'Standard Slope distribution')
    stat_depict(new_df, 'RSRS')
    # 1 Standard Deviation
    # 1.021451
    # -1 Standard Deviation
    # 0.783374
    # 取S1=0.78，S2=1.02
    S1 = 0.78
    S2 = 1.02
    RSRS_Z_Strategy = RSRS_Strategy(RSRS_Z, S1, S2)
    summary(RSRS_Z_Strategy)


if __name__ == '__main__':
    main()
