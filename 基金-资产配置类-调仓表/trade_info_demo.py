import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Order(object):
    def __init__(self, cash, commision, slippery):
        self.cash = cash
        self.commision = commision
        self.slippery = slippery
        self.ref = 0  # 订单编号
        self.status = 0  # 订单状态

    def Buy(self, stock, price, value):
        if self.status == 1:
            self.status = 0
            # 重置订单状态
        self.cash -= self.commision * value
        self.cash -= value
        self.cash -= price * self.slippery
        self.status = 1
        self.ref += 1
        print('BUY EXECUTED, ref:{}, Price:{}, Cost:{}, Stock:{}'.format(self.ref, price, value, stock))
        return (self.cash, value / price)

    def Sell(self, stock, price, value):
        if self.status == 1:
            self.status = 0
        self.cash += value
        self.status = 1
        self.ref += 1
        print('SELL EXECUTED, ref:{}, Price:{}, Cost:{}, Stock:{}'.format(self.ref, price, value, stock))
        return (self.cash, value / price)


class StockSelectStrategy(object):
    '''多因子选股 - 基于调仓表'''

    def __init__(self, buy_stock, stock_price):
        self.cash = 1000000
        self.commission = 0.0003
        self.slippery = 0.0001
        self.stock_price = stock_price
        self.buy_stock = buy_stock
        # 读取调仓日期，即每月的最后一个交易日，回测时，会在这一天下单，然后在下一个交易日，以开盘价买入
        self.trade_dates = self.buy_stock.index.unique().tolist()
        self.buy_stocks_pre = []  # 记录上一期持仓
        self.volume = dict()
        self.cashArr = []
        self.positionArr = []

    def Portfolio_plot(self, datetime, cashArr, title):
        plt.figure(figsize=(10, 5))
        plt.plot(datetime, cashArr, label=title)
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('Datetime')
        plt.ylabel(title)
        plt.title(title)
        plt.savefig(r'result\{}.jpg'.format(title))
        plt.show()

    def main(self, datatime):
        for dt in datatime:
            Or = Order(self.cash, self.commission, self.slippery)
            # 如果是调仓日，则进行调仓操作
            for td in self.trade_dates:
                # 遍历调仓日
                if dt == td:
                    print("--------------{} 为调仓日----------".format(dt))
                    # 提取当前调仓日的持仓列表
                    buy_stocks_data = self.buy_stock.loc[td, :]
                    stock_price_data = self.stock_price.loc[td, :]
                    long_list = buy_stocks_data['sec_code'].tolist()
                    print('long_list', long_list)  # 打印持仓列表
                    # 对现有持仓中，调仓后不再继续持有的股票进行卖出平仓
                    sell_stock = [i for i in self.buy_stocks_pre if i not in long_list]
                    print('sell_stock', sell_stock)  # 打印平仓列表
                    if len(sell_stock) > 0:
                        print("-----------对不再持有的股票进行平仓--------------")
                        for stock in sell_stock:
                            try:
                                price = stock_price_data.query(f"sec_code=='{stock}'")['close'].iloc[0]
                                v = self.volume[stock]
                                self.cash, v_sell = Or.Sell(stock, price, v * price)
                            except:
                                pass
                        self.volume = dict()
                    # 买入此次调仓的股票：多退少补原则
                    print("-----------买入此次调仓期的股票--------------")
                    position_sum = 0
                    for stock in long_list:
                        w = buy_stocks_data.query(f"sec_code=='{stock}'")['weight'].iloc[0]  # 提取持仓权重
                        price = stock_price_data.query(f"sec_code=='{stock}'")['close'].iloc[0]
                        self.cash, v = Or.Buy(stock, price, self.cash * w * 0.95)  # 预留5%的资金
                        self.volume[stock] = v
                        position_sum += price * v
                    self.positionArr.append(position_sum)
                    self.buy_stocks_pre = long_list  # 保存此次调仓的股票列表
                    self.cashArr.append(self.cash)
                    print('当前持仓总价值:{}'.format(self.cash))
        self.Portfolio_plot(self.trade_dates, self.cashArr, 'total cash')
        self.Portfolio_plot(self.trade_dates, self.positionArr, 'position value')


if __name__ == '__main__':
    stock_price = pd.read_csv(r'data\daily_price.csv', index_col=0)
    buy_stock = pd.read_csv("./data/trade_info.csv", index_col=0)
    datetime = stock_price.index.unique().tolist()
    x = StockSelectStrategy(buy_stock, stock_price)
    x.main(datetime)
