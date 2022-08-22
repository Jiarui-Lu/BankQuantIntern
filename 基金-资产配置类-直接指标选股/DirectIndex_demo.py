import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Order(object):
    def __init__(self, commision, slippery):
        self.commision = commision
        self.slippery = slippery
        self.ref = 0  # 订单编号
        self.status = 0  # 订单状态

    def Buy(self, stock, price, value):
        if self.status == 1:
            self.status = 0
            # 重置订单状态
        cash=self.commision * value+value+price * self.slippery
        self.status = 1
        self.ref += 1
        print('BUY EXECUTED, ref:{}, Price:{}, Cost:{}, Stock:{}'.format(self.ref, price, value, stock))
        return (cash, value / price)

    def Sell(self, stock, price, value):
        if self.status == 1:
            self.status = 0
        cash = value
        self.status = 1
        self.ref += 1
        print('SELL EXECUTED, ref:{}, Price:{}, Cost:{}, Stock:{}'.format(self.ref, price, value, stock))
        return (cash, value / price)

class StockSelectStrategy(object):
    def __init__(self,data):
        self.data=data
        self.datetime=self.data.index.unique().tolist()
        self.selnum = 5  # 设置持仓股数在总的股票池中的占比，如买入表现最好的前30只股票
        self.vperiod = 6  # 计算波动率的周期，过去6个月的波动率
        self.mperiod = 3  # 计算动量的周期，如过去2个月的收益
        self.reserve = 0.05  # 5% 为了避免出现资金不足的情况，每次调仓都预留 5% 的资金不用于交易
        self.perctarget = (1.0 - self.reserve) / self.selnum
        self.cash=1000000
        self.commision=0.0003
        self.slippery=0.0001
        self.positionvalue=[]
        self.volume=[]
        self.priceArr=[]
    def rs(self):
        d=dict()
        for stock in self.data['sec_code'].unique():
            try:
                price_list=[]
                for dt in self.datetime:
                        close_data=self.data.loc[dt, :]
                        price = close_data.query(f"sec_code=='{stock}'")['close'].iloc[0]
                        price_list.append(price)
                pct_list=list(pd.Series(price_list).pct_change())
                d[stock]=pct_list
            except:
                pass
        return d
    def vs(self):
        d=dict()
        # signals['ma3'] = signals['close'].rolling(window=ma3, min_periods=1, center=False).mean()
        rs=self.rs()
        for stock,ret in rs.items():
            vol=pd.Series(ret).rolling(window=self.vperiod,min_periods=1,center=False).std().dropna()
            vol_list=list(1/vol)
            d[stock]=vol_list
        return d
    def ms(self):
        d=dict()
        for stock in self.data['sec_code'].unique():
            try:
                price_list=[]
                for dt in self.datetime:
                        close_data=self.data.loc[dt, :]
                        price = close_data.query(f"sec_code=='{stock}'")['close'].iloc[0]
                        price_list.append(price)
                roc_list=[]
                for i in range(self.mperiod,len(price_list)):
                    roc=(price_list[i]-price_list[i-self.mperiod])/price_list[i-self.mperiod]
                    roc_list.append(roc)
                d[stock]=roc_list
            except:
                pass
        return d
    def EP(self):
        d=dict()
        for stock in self.data['sec_code'].unique():
            try:
                EP_list=[]
                for dt in self.datetime:
                        close_data=self.data.loc[dt, :]
                        price = close_data.query(f"sec_code=='{stock}'")['EP'].iloc[0]
                        EP_list.append(price)
                d[stock]=EP_list
            except:
                pass
        return d
    def ROE(self):
        d=dict()
        for stock in self.data['sec_code'].unique():
            try:
                ROE_list=[]
                for dt in self.datetime:
                        close_data=self.data.loc[dt, :]
                        price = close_data.query(f"sec_code=='{stock}'")['ROE'].iloc[0]
                        ROE_list.append(price)
                d[stock]=ROE_list
            except:
                pass
        return d
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
    def main(self):
        all_factors = [self.rs(), self.vs(), self.ms(), self.EP(), self.ROE()]
        for dt in self.datetime:
            Or = Order(self.commision, self.slippery)
            ranks = dict()
            stocks = self.data['sec_code'].unique()
            for stock in stocks:
                ranks[stock] = 0
            # print(ranks)
            d_tmp = dict()
            for factor in all_factors:
                for k,arr in factor.items():
                    for i in range(len(arr)):
                        d_tmp[k]=arr[i]
                d_tmp_sorted=sorted(d_tmp.items(),key=lambda x:x[1],reverse=True)
                # 里面元素为(key,value)的元素列
                for tuple_index in (range(1, len(d_tmp_sorted) + 1)):
                    key = d_tmp_sorted[tuple_index - 1][0]
                    ranks[key] = tuple_index + ranks[key]
                # print(ranks)
            for k, v in ranks.items():
                if v == 0:
                    del ranks[k]
            ranks_sorted = sorted(ranks.items(), key=lambda x: x[1], reverse=False)
            # 选取前 self.p.selnum 只股票作为持仓股
            rtop = dict(ranks_sorted[:self.selnum])

            # 剩余股票将从持仓中剔除（如果在持仓里的话）
            rbot = dict(ranks_sorted[self.selnum:])
            buy_stocks_data = self.data.loc[dt, :]
            for stock in rtop.keys():
                price = buy_stocks_data.query(f"sec_code=='{stock}'")['close'].iloc[0]
                self.priceArr.append(price)
                cash_delta, v_buy = Or.Buy(stock, price, self.cash*self.perctarget)
                self.volume.append(v_buy)
        v0=self.volume[:5]
        for dt in self.datetime:
            for p in range(0,len(self.priceArr),5):
                position=0
                for i in range(5):
                    position+=self.priceArr[p+i]*v0[i]
                self.positionvalue.append(position)
        self.Portfolio_plot(self.datetime,self.positionvalue[:len(list(self.datetime))],'Portfolio position value')

if __name__=='__main__':
    month_price = pd.read_csv("./data/month_price.csv", index_col=0)
    X=StockSelectStrategy(month_price)
    X.main()