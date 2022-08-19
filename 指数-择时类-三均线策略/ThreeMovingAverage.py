import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf


class MySignal(bt.Indicator):
    lines = ('signal',)  # 声明 signal 线，交易信号放在 signal line 上
    params = dict(
        short_period=5,
        median_period=20,
        long_period=60)

    def __init__(self):
        self.s_ma = bt.ind.SMA(period=self.p.short_period)
        self.m_ma = bt.ind.SMA(period=self.p.median_period)
        self.l_ma = bt.ind.SMA(period=self.p.long_period)
        # 短期均线在中期均线上方，且中期均取也在长期均线上方，三线多头排列，取值为1；反之，取值为0
        self.signal1 = bt.And(self.m_ma > self.l_ma, self.s_ma > self.m_ma)
        # 求上面 self.signal1 的环比增量，可以判断得到第一次同时满足上述条件的时间，第一次满足条件为1，其余条件为0
        self.buy_signal = bt.If((self.signal1 - self.signal1(-1)) > 0, 1, 0)
        # 短期均线下穿长期均线时，取值为1；反之取值为0
        self.sell_signal = bt.ind.CrossDown(self.s_ma, self.m_ma)
        # 将买卖信号合并成一个信号
        self.lines.signal = bt.Sum(self.buy_signal, self.sell_signal * (-1))


# class MovingAverageStrategy(bt.Strategy):
#     params = dict(
#         short_period=5,
#         median_period=20,
#         long_period=60,
#         reserve=0.05
#     )
#     def __init__(self):
#         self.s_ma = bt.ind.SMA(period=self.p.short_period)
#         self.m_ma = bt.ind.SMA(period=self.p.median_period)
#         self.l_ma = bt.ind.SMA(period=self.p.long_period)
#         self.signal1 = bt.And(self.m_ma > self.l_ma, self.s_ma > self.m_ma)
#         # 求上面 self.signal1 的环比增量，可以判断得到第一次同时满足上述条件的时间，第一次满足条件为1，其余条件为0
#         self.buy_signal = bt.If((self.signal1 - self.signal1(-1)) > 0, 1, 0)
#         # 短期均线下穿长期均线时，取值为1；反之取值为0
#         self.sell_signal = bt.ind.CrossDown(self.s_ma, self.m_ma)
#     def log(self, txt, dt=None):
#         ''' 策略日志打印函数'''
#         dt = dt or self.datas[0].datetime.date(0)
#         print('%s, %s' % (dt.isoformat(), txt))
#     def next(self):
#         if self.position.size:
#             if self.buy_signal:
#                 self.buy()
#             elif self.sell_signal:
#                 self.sell()
#     def notify_order(self, order):
#         # 未被处理的订单
#         if order.status in [order.Submitted, order.Accepted]:
#             return
#         # 已经处理的订单
#         if order.status in [order.Completed, order.Canceled, order.Margin]:
#             if order.isbuy():
#                 self.log(
#                     'BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
#                     (order.ref,  # 订单编号
#                      order.executed.price,  # 成交价
#                      order.executed.value,  # 成交额
#                      order.executed.comm,  # 佣金
#                      order.executed.size,  # 成交量
#                      order.data._name))  # 股票名称
#             else:  # Sell
#                 self.log('SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
#                          (order.ref,
#                           order.executed.price,
#                           order.executed.value,
#                           order.executed.comm,
#                           order.executed.size,
#                           order.data._name))
#
#
#
# # 实例化大脑
cerebro = bt.Cerebro()
# 加载数据
# 读取行情数据
stock_price = pd.read_excel(r"data\data.xls", index_col=0)
# 按股票代码，依次循环传入数据
for stock in stock_price['sec_code'].unique():
    # 日期对齐
    data = pd.DataFrame(index=stock_price.index.unique())  # 获取回测区间内所有交易日
    df = stock_price.query(f"sec_code=='{stock}'")[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
    # print(df)
    data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
    # 缺失值处理：日期对齐时会使得有些交易日的数据为空，所以需要对缺失数据进行填充
    data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
    data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].fillna(
        method='pad')
    data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].fillna(
        0.0000001)
    # 导入数据
    datafeed = bt.feeds.PandasData(dataname=data_,
                                   fromdate=pd.to_datetime('2020-01-02'),
                                   todate=pd.to_datetime('2021-12-31'))
    cerebro.adddata(datafeed, name=stock)  # 通过 name 实现数据集与股票的一一对应
    print(f"{stock} Done !")
# 初始资金 1,000,000
cerebro.broker.setcash(1000000.0)
# 佣金，双边各 0.0003
cerebro.broker.setcommission(commission=0.0003)
# 滑点：双边各 0.0001
cerebro.broker.set_slippage_perc(perc=0.0001)
# 每次固定交易100股
cerebro.addsizer(bt.sizers.FixedSize, stake=100)
# 添加交易信号
# cerebro.addstrategy(MovingAverageStrategy)
cerebro.add_signal(bt.SIGNAL_LONG, MySignal)
# 回测时需要添加 PyFolio 分析器
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

result = cerebro.run()
# 借助 pyfolio 进一步做回测结果分析
pyfolio = result[0].analyzers.pyfolio  # 注意：后面不要调用 .get_analysis() 方法
#
returns, positions, transactions, gross_lev = pyfolio.get_pf_items()
pf.create_full_tear_sheet(returns)
plt.show()
