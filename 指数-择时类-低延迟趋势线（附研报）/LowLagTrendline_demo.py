import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def LLT(price, a):
    price_new = np.zeros(price.shape)
    for i in range(2):
        price_new[i] = price[i]
    for i in range(2, len(price_new)):
        price_new[i] = (2 - 2 * a) * price_new[i - 1] - (1 - a) ** 2 * price_new[i - 2] + 0.5 * a ** 2 * price[
            i - 1] - (a - 0.75 * a ** 2) * price[i - 2] + (a - 0.25 * a ** 2) * price[i]
    return price_new


def Cala(d):
    a = 2 / (1 + d)
    return a


def lowlagtrend(df):
    d = [20, 50, 90]  # 移动平均天数
    color = ['r-', 'g-', 'y-']
    price = df['close']
    for i in range(len(d)):
        df['LLT_{:.2f}'.format(Cala(d[i]))] = LLT(price, Cala(d[i]))
    plt.figure()
    plt.plot(df.index, price, 'b-', label='original data')
    for i in range(len(d)):
        plt.plot(df.index, df['LLT_{:.2f}'.format(Cala(d[i]))], color[i], label='LLT:d={}'.format(d[i]))
    plt.legend()
    plt.grid(True)
    plt.title('lowlagtrend')
    plt.xlabel('datetime')
    plt.ylabel('price')
    plt.savefig(r'result\lowlagtrend.jpg')
    plt.show()
    return df


# 以a=0.04为例
def signal_generation(new):
    new['positions'], new['signals'] = 0, 0
    columns = new.columns
    # Index(['open', 'high', 'low', 'close', 'LLT_0.10', 'LLT_0.04', 'LLT_0.02',
    #        'positions', 'signals'],
    #       dtype='object')
    # print(new['LLT_0.10'])
    # print(new['LLT_0.10'].shift(1))
    new['positions'] = np.where(new['LLT_0.04'] > new['LLT_0.04'].shift(1), 1, 0)
    new['signals'] = new['positions'].diff()
    return new


def strategy_plot(new, ticker):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new['close'].plot(lw=3, label='%s' % ticker)
    new['LLT_0.04'].plot(linestyle=':', label='LLT:d=50', color='k')
    ax.plot(new.loc[new['signals'] == 1].index, new['close'][new['signals'] == 1], marker='^', color='g', label='LONG',
            lw=0, markersize=10)
    ax.plot(new.loc[new['signals'] == -1].index, new['close'][new['signals'] == -1], marker='v', color='r',
            label='SHORT', lw=0, markersize=10)

    plt.legend()
    plt.grid(True)
    plt.title('positions strategy when LLT:d=50')
    plt.ylabel('price')
    plt.savefig(r'result\positions_strategy.jpg')
    plt.show()


if __name__ == '__main__':
    df = pd.read_excel(r'data\data.xls', index_col=0)
    new = lowlagtrend(df)
    sig = signal_generation(new)
    slicer = 100
    sig = sig[-slicer:]
    strategy_plot(sig, '000300.SH')
