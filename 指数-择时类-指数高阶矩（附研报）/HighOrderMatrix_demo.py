import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.family']='serif'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 图表主题
plt.style.use('seaborn')

# 用来显示中文
plt.rcParams['font.family'] = 'SimHei'

price = pd.read_excel(r'data\HS300.xlsx', index_col=0)
price['pre_close'] = price['close'].shift(1)
ret = price['close'] / price['pre_close'] - 1
ret = ret.dropna()


# 计算N阶矩
def cal_moment(arr: np.array, Order: int):
    return np.mean(arr ** Order)


# 计算N阶矩
temp = {}
for n in range(2, 8):
    temp['moment_' + str(n)] = ret.rolling(20).apply(
        cal_moment, kwargs={'Order': n}, raw=False)

# 加入收盘价
temp['close'] = price['close']
temp_df = pd.DataFrame(temp)


# print(temp_df)
# 可视化
def plot_twin(df: pd.DataFrame):
    for i, col_name in enumerate(
            [x for x in df.columns.tolist() if x != "close"]):
        fig = plt.figure(figsize=(18, 10))
        ax1 = fig.add_subplot(111)
        plt.plot(df.index, df[col_name], label=col_name, color='red')
        ax1.set_ylabel('moment')  # 设置左边纵坐标标签
        ax2 = ax1.twinx()
        plt.plot(df.index, df['close'], label='close', color='blue')
        ax2.set_ylabel('price')  # 设置右边纵坐标标签
        plt.xlabel('year')
        plt.legend(loc='best')
        plt.grid(True)
        plt.title('Higher order matrix versus price_moment_{}'.format(i + 2))
        plt.savefig(r'result\Higher order matrix versus price_moment_{}.jpg'.format(i + 2))
        plt.show()


# plot_twin(temp_df)

# 五阶矩阵简单择时
def cal_ema(arr, alpha):
    series = pd.Series(arr)
    return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]


# # 选取5阶作为信号
# ema_window = 90
# alpha = ema_window + 1
# # singal_series = temp_df['moment_5'].ewm(alpha=2/alpha,adjust=False).mean()
# singal_series = temp_df['moment_5'].rolling(ema_window).apply(cal_ema, kwargs={'alpha': 2 / alpha}, raw=False)
# # 获取昨日信号
# per_singal = singal_series.shift(1)
# # 当然信号大于上日信号
# cond = singal_series > per_singal


# 获取持仓

def get_position(ret: pd.Series, cond: pd.Series) -> pd.DataFrame:
    df = pd.concat([ret, cond], axis=1)  # 收益率与信号合并
    df.columns = ['ret', 'cond']

    position = []  # 储存开仓信号，1为持仓，0为空仓
    for idx, row in df.iterrows():

        if position:

            # 当然出现开仓信号，上一日未持仓
            if row['cond'] and position[-1] == 0:

                position.append(1)

            # 当然有开仓信号，上日有持仓，大于止损线
            elif row['cond'] and position[-1] == 1 and row['ret'] >= -0.1:

                position.append(1)

            else:

                position.append(0)

        else:

            if row['cond']:

                position.append(1)

            else:

                position.append(0)

    df['position'] = position

    return df


# # 获取
# algorithm_return = get_position(ret, cond)
#
# algorithm_return = algorithm_return['ret'].shift(
#     -1) * algorithm_return['position']
#
# cum = (1 + algorithm_return).cumprod()
# benchmark = (1 + ret).cumprod()

# # 高阶矩净值
# new_df=pd.DataFrame({'algorithm_return':cum,'benchmark':benchmark})
# new_df.to_excel(r'result\五阶矩简单择时.xlsx')
def Plot(df, title):
    plt.figure()
    plt.plot(df.index, df['algorithm_return'], label='algorithm_return')
    plt.plot(df.index, df['benchmark'], label='benchmark')
    plt.xlabel('time')
    plt.ylabel('return')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title(title)
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()


# Plot(new_df,'五阶矩简单择时')


# pandas+1次循环
## 五阶矩计算窗口cal_momentt_window = 20
## 外推窗口rolling_window = 90
# 沪深300的起始日2005-04-08

cal_momentt_window = 20
rolling_window = 90
# # 获取数据
price_data = price.copy()

# 计算收益
ret_df = price_data.pct_change()
ret_df.rename(columns={'close': 'ret'}, inplace=True)

# 计算5阶矩
momentt = ret_df['ret'].rolling(cal_momentt_window).apply(
    cal_moment, kwargs={'Order': 5}, raw=False)

# 计算alpha取值范围
alpha = np.arange(0.05, 0.55, step=0.05)

# 计算不同参数的EMA
ema_momentt = pd.concat(
    [momentt.ewm(alpha=x, adjust=False).mean() for x in alpha], axis=1)

ema_momentt.columns = alpha

# 之后一期T日信号为T-1日产生的
diff_ema = ema_momentt.diff().shift(1)

# 各个alpha计算出的ema累计90日收益
## 大于0多头，小于0空头
## 逻辑等价于EMA_{t-1} > EMA_{t-2} 多头,反之
cond = ((diff_ema > 0) * 1 + (diff_ema < 0) * -1)

cum90rate = cond * np.broadcast_to(
    np.expand_dims(ret_df['ret'].values, axis=1), diff_ema.shape)

# 90日累计收益
cum90rate = cum90rate.fillna(0).rolling(rolling_window).sum()

# 去除前序20日的高阶矩日期
slice_ret = ret_df['ret'].iloc[cal_momentt_window:]
slice_momentt = momentt.iloc[cal_momentt_window:]
slice_ema = ema_momentt.iloc[cal_momentt_window:]
slice_diff_ema = diff_ema.iloc[cal_momentt_window:]
slice_cum90rate = cum90rate.iloc[cal_momentt_window:]

lossflag = 0  # 记录单次亏损阈值
loss_position = 0  # 记录发生亏损的方向

flag = np.zeros(len(slice_ret))  # 持仓 1多头,0空仓,-1空仓

set_alpha = 0.4  # 初始ema的alpha值

for i in range(1, len(slice_ret)):

    if i % 90 == 0:
        set_alpha = slice_cum90rate.iloc[i].idxmax()

    if lossflag < -0.1:
        flag[i] = 0  # 上一日发生亏损，本日空仓
        loss_position = flag[i - 1]  # 记录的持仓方向为上个发生亏损的方向
        lossflag = 0  # 记录的数据清0
        continue  # 之间执行下个循环

    # 多头信号 且 亏损方向不为多头
    if slice_diff_ema[set_alpha].iloc[i] > 0 and loss_position != 1:
        flag[i] = 1
        loss_position = 0

    # 空头信号 且 亏损方向不为空头
    if slice_diff_ema[set_alpha].iloc[i] < 0 and loss_position != -1:
        flag[i] = -1
        loss_position = 0

    if flag[i] == flag[i - 1]:

        lossflag = lossflag + flag[i] * slice_ret[i]
        lossflag = min(lossflag, 0)
    else:
        lossflag = 0

strategy_rate = slice_ret * flag
nav = (1 + strategy_rate).cumprod()
new_benchmark = (1 + slice_ret).cumprod()
new_df = pd.DataFrame({'algorithm_return': nav, 'benchmark': new_benchmark})
new_df.to_excel(r'result\五阶矩结合EMA择时.xlsx')
# print(nav)
plt.figure(figsize=(18, 8))
plt.title('策略净值')
plt.plot(nav, label='策略净值', c='r')
plt.plot((1 + slice_ret).cumprod(), label='基准净值', ls='--')
plt.savefig(r'result\五阶矩结合EMA择时.jpg')
plt.legend()
