"""
Python版本号：3.6.8

菲阿里四价策略是一种简单趋势型日内交易策略。昨天最高点、昨天最低点、昨日收盘价、今天开盘价,可并称为菲阿里四价。
没有持仓下，当现价突破上轨时做多，当现价跌穿下轨时做空；以开盘价作为止损价，尾盘平仓，其中
上轨=昨日最高点；
下轨=昨日最低点；
止损=今日开盘价。
注：受目前回测机制限制，期货主力合约只能回测最近三年的数据，连续合约不受影响
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def min2day(df, column, year, month):
    # lets create a dictionary
    # we use keys to classify different info we need
    memo = {'date': [], 'open': [], 'close': [], 'high': [], 'low': []}
    for i in range(1, 32):

        try:
            temp = df['%s-%s-%s 00:02:00' % (year, month, i):'%s-%s-%s 23:58:00' % (year, month, i)][column]

            memo['open'].append(temp[0])
            memo['close'].append(temp[-1])
            memo['high'].append(max(temp))
            memo['low'].append(min(temp))
            memo['date'].append('%s-%s-%s' % (year, month, i))


        except Exception:
            pass

    intraday = pd.DataFrame(memo)
    intraday.set_index(pd.to_datetime(intraday['date']), inplace=True)


    # preparation
    # intraday['range1'] = intraday['high'].rolling(rg).max() - intraday['close'].rolling(rg).min()
    # intraday['range2'] = intraday['close'].rolling(rg).max() - intraday['low'].rolling(rg).min()
    # intraday['range'] = np.where(intraday['range1'] > intraday['range2'], intraday['range1'], intraday['range2'])

    return intraday

# signal generation
# even replace assignment with pandas.at
# it still takes a while for us to get the result
# any optimization suggestion besides using numpy array?
def signal_generation(df, intraday):
    # as the lags of days have been set to 5
    # we should start our backtesting after 4 workdays of current month
    # cumsum is to control the holding of underlying asset
    # sigup and siglo are the variables to store the upper/lower threshold
    # upper and lower are for the purpose of tracking sigup and siglo
    signals = df[df.index >= intraday['date'].iloc[1]]
    signals['signals'] = 0
    signals['cumsum'] = 0
    # print(intraday['high'][signals.index[0].day-1])
    Dict={}
    for d in intraday.index:
        Dict[d.day]=list(intraday.index).index(d)
    for i in signals.index:
        if signals['price'][i] > intraday['high'][Dict[i.day]-1]:
            signals.at[i, 'signals'] = 1
        if signals['price'][i] < intraday['low'][Dict[i.day] - 1]:
            signals.at[i, 'signals'] = -1
            # check if signal has been generated
            # if so, use cumsum to verify that we only generate one signal for each situation
        if pd.Series(signals['signals'])[i] != 0:
            signals['cumsum'] = signals['signals'].cumsum()
            if (pd.Series(signals['cumsum'])[i] > 1 or pd.Series(signals['cumsum'])[i] < -1):
                signals.at[i, 'signals'] = 0
        if i.hour == 23 and (i.minute == 58 or i.minute == 59):
            signals['cumsum'] = signals['signals'].cumsum()
            signals.at[i, 'signals'] = -signals['cumsum'][i:i]
    return signals

def plot(signals, intraday):
    # we have to do a lil bit slicing to make sure we can see the plot clearly
    # the only reason i go to -3 is that day we execute a trade
    # give one hour before and after market trading hour for as x axis
    date = pd.to_datetime(intraday['date']).iloc[-3]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # mostly the same as other py files
    # the only difference is to create an interval for signal generation
    ax.plot(signals.index, signals['price'], label=column)
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
    plt.title('Fairy4Price')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    df = pd.read_excel(r'data\eurusd.xls')[:50000]
    df.set_index(pd.to_datetime(df['date']), inplace=True)

    year = df.index[0].year
    month = df.index[0].month
    column = 'price'

    intraday = min2day(df, column, year, month)
    # print(intraday)
    signals = signal_generation(df, intraday)
    plot(signals, intraday)