import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ma3cd(signals):
    signals['ma1'] = signals['close'].rolling(window=ma1, min_periods=1, center=False).mean()
    signals['ma2'] = signals['close'].rolling(window=ma2, min_periods=1, center=False).mean()
    signals['ma3'] = signals['close'].rolling(window=ma3, min_periods=1, center=False).mean()

    return signals


def signal_generation(df, method):
    signals = method(df)
    signals['positions'] = 0
    for i in range(ma3,len(signals['ma1'])):
        if signals['ma1'][i]>=signals['ma2'][i] and signals['ma2'][i]>=signals['ma3'][i]:
            signals['positions'][i]=1
        else:
            signals['positions'][i] = 0
    signals['signals'] = signals['positions'].diff()

    return signals

def plot(new, ticker):
    # the first plot is the actual close price with long/short positions
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new['close'].plot(label=ticker)
    ax.plot(new.loc[new['signals'] == 1].index, new['close'][new['signals'] == 1], label='LONG', lw=0, marker='^',
            c='g')
    ax.plot(new.loc[new['signals'] == -1].index, new['close'][new['signals'] == -1], label='SHORT', lw=0, marker='v',
            c='r')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    plt.savefig(r'result\positions_strategy.jpg')
    plt.show()

def main():

    global ma1, ma2, ma3, ticker, slicer

    ma1 = 5
    ma2 = 20
    ma3 = 60
    ticker = 'HS300'


    df = pd.read_excel(r'data\data.xls', index_col=0)

    new = signal_generation(df, ma3cd)
    plot(new, ticker)



if __name__ == '__main__':
    main()
