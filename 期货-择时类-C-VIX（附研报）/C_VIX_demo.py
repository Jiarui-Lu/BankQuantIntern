from typing import Tuple,Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from calc_func import (CVIX, prepare_data2calc, get_n_next_ret,
                           get_quantreg_res,create_quantile_bound)
class Order(object):
    def __init__(self, commision, slippery):
        self.commision = commision
        self.slippery = slippery
        self.ref = 0  # 订单编号
        self.status = 0  # 订单状态
        self.position = []

    def Buy(self, stock, price, value):
        if self.status == 1:
            self.status = 0
            # 重置订单状态
        cash=self.commision * value+value+price * self.slippery
        self.status = 1
        self.ref += 1
        self.position.append(stock)
        print('BUY EXECUTED, ref:{}, Price:{}, Cost:{}, Stock:{}'.format(self.ref, price, value, stock))
        return (cash, value / price)

    def Sell(self, stock, price, value):
        if self.status == 1:
            self.status = 0
        cash = value
        self.status = 1
        self.ref += 1
        self.position.remove(stock)
        print('SELL EXECUTED, ref:{}, Price:{}, Cost:{}, Stock:{}'.format(self.ref, price, value, stock))
        return (cash)

    def PositionStatus(self):
        if len(self.position)==0:
            return False
        else:
            return True

def plot_indicator(xlabel,ylabel,index,title,price,indicator):
    label1 = xlabel
    label2 = ylabel
    fig = plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax_twin = fig.add_subplot(111, label="2", frame_on=False)
    ax.plot(index,price,color='#FFD700', label=label1)
    ax.set_ylabel(label1)
    ax_twin.plot(index,indicator,color='r', label=label2)
    ax_twin.set_ylabel(label2)
    ax_twin.xaxis.tick_top()
    ax_twin.yaxis.tick_right()
    ax_twin.xaxis.set_label_position('top')
    ax_twin.yaxis.set_label_position('right')
    plt.title(title)
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()

def plot_three_fig(model,k,title):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, label="1")
    ax1.set_title(title)
    ax1.plot(model['q'], model['vix'], color='black')
    ax1.fill_between(model['q'], model['ub'], model['lb'], alpha=0.2)
    ax2 = fig.add_subplot(132, label="2")
    ax2.set_title(title)
    ax2.hist2d(algin_vix, algin_next_chg[k], bins=10, cmap='Blues')
    ax2.set_ylabel('收益率')
    ax2.set_xlabel('vix')
    ax2.axhline(0, ls='--', color='black')
    ax3 = fig.add_subplot(133, label="3")
    group_ser: pd.Series = pd.qcut(algin_vix, 10, False) + 1
    df: pd.DataFrame = group_ser.to_frame('group')
    df['next'] = algin_next_chg[k]
    df.index.names = ['date']

    group_avg_ret: pd.Series = pd.pivot_table(df.reset_index(),
                                              index='date',
                                              columns='group',
                                              values='next').mean()

    ax3.set_title(title)
    xmajor_formatter = mpl.ticker.FuncFormatter(lambda x, pos: '%.2f%%' %
                                                               (x * 100))
    ax3.yaxis.set_major_formatter(xmajor_formatter)
    group_avg_ret.plot.bar(ax=ax3, color='#1f77b4')
    ax3.axhline(0, color='black')
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()

def plot_qunatile_signal(price: pd.Series,
                         signal: pd.Series,
                         window: int,
                         bound: Tuple,
                         title: str = '') -> mpl.axes:
    """画价格与信号的关系图

    Args:
        price (pd.Series): 价格
        signal (pd.Series): 信号
        window (int): 滚动时间窗口
        bound (Tuple): bound[0]-上轨百分位数,bound[1]-下轨百分位数

    Returns:
        mpl.axes: _description_
    """
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(3, 1)

    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[2:, :])

    price.plot(ax=ax1, title=title)

    signal.plot(ax=ax2, color='darkgray', label='signal')

    # 构建上下轨
    up, lw = bound
    ub: pd.Series = signal.rolling(window).apply(
        lambda x: np.percentile(x, up), raw=True)

    lb: pd.Series = signal.rolling(window).apply(
        lambda x: np.percentile(x, lw), raw=True)
    # 画上下轨
    ub.plot(ls='--', color='r', ax=ax2, label='ub')
    lb.plot(ls='--', color='green', ax=ax2, label='lb')
    ax2.legend()

    plt.subplots_adjust(hspace=0)
    plt.savefig(r'result\{}.jpg'.format(title))
    return gs


# 设置字体 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False

# 数据获取
opt_data=pd.read_csv(r'data\opt_data.csv',index_col=0, parse_dates=['date'])
interpld_shibor=pd.read_csv(r'data\interpld_shibor.csv',index_col=0,parse_dates=True)
interpld_shibor.columns = list(map(int, interpld_shibor.columns))
# print(opt_data)
# print(interpld_shibor)
price=pd.read_csv(r'data\price.csv',index_col=1)
del price[price.columns[0]]
# print(price)
hs300=pd.DataFrame(price.query('code == "000300.XSHG"').copy())
hs300=hs300.drop(columns=['code'])

data_all = prepare_data2calc(opt_data, interpld_shibor)

vix_func = CVIX(data_all)
# 计算vix
vix=pd.Series(vix_func.vix())

# 计算skew
skew=pd.Series(vix_func.skew())

plot_indicator('沪深300','VIX',hs300.index,'VIX与沪深300',hs300['close'],vix)
plot_indicator('沪深300','SKEW',hs300.index,'SKEW与沪深300',hs300['close'],skew)


## 对齐未来收益与vix
algin_next_chg, algin_vix = get_n_next_ret(hs300['close'], vix)

## 获取模型结果
models_dic: Dict = {
    name: get_quantreg_res(algin_vix,ser)
    for name, ser in algin_next_chg.items()
}


## 画图
for k, v in models_dic.items():
    plot_three_fig(v,k, title=k)

plot_qunatile_signal(hs300['close'], vix, 60, (85, 20), '沪深300指数走势与信号的关系')
quantile_bound:pd.DataFrame = create_quantile_bound(vix,60,(85,20))
# print(quantile_bound)
# print(hs300)
data_all=hs300[-len(quantile_bound.index):]
for col in quantile_bound.columns:
    data_all[col]=quantile_bound[col]

def signal_generation(df):
    df['open_signal']=0
    df['close_signal']=0
    df['open_signal']=np.where(df['signal']<df['lb'],1,0)
    df['close_signal']=np.where(df['signal']>df['ub'],1,0)
    df['cumsum']=df['open_signal']-df['close_signal']
    return df
# print(signal_generation(data_all))
sig=signal_generation(data_all)
cash=1000000
commision=0.0003
slippery=0.0001
perctarget=0.95
cashArr=[cash]
Or=Order(commision,slippery)
l=sig.index
for i in l:
    try:
        if list(l).index(sell_time)>list(l).index(i):
            continue
    except:
        pass
    if sig.loc[i,'cumsum']==1:
        buy_cashdelta,v_buy=Or.Buy('HS300',sig.loc[i,'close'],cash*perctarget)
        cash=cash-buy_cashdelta
        # print('buy:',cash)
        count = 0
        for j in l[list(l).index(i):]:
            if sig.loc[j,'cumsum']==-1 and Or.PositionStatus():
                sell_price=sig.loc[j, 'close']
                sell_cashdelta= Or.Sell('HS300', sell_price,sell_price*v_buy)
                cash=cash+sell_cashdelta
                # print('sell:',cash)
                sell_time=j
                count=count+1
                cashArr.append(cash)
                break
        if count==0:
            break
# print(cashArr)
def Portfolio_plot(datetime, cashArr, title):
    plt.figure(figsize=(10, 5))
    plt.plot(datetime, cashArr, label=title)
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('Datetime')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()
Portfolio_plot([i for i in range(len(cashArr))],cashArr,'回测持仓总价值序列')









