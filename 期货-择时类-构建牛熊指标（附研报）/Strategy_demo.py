import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ConsturctBullishBearIndexPlot

def CalBullishBearIndex(df,turnover):
    df['pre_close'] = df['close'].shift(1)
    df['pct'] = df['close'] / df['pre_close'] - 1
    df['std_200'] = df['pct'].rolling(200).std()
    df['turnover_200']=turnover['turnover_rate_f'].rolling(200).mean()
    df['kernel_index'] = df['std_200']/df['turnover_200']
    df=df.dropna()
    return df

def signal_generation(df,ma1,ma2):
    df['ma1'] = df['kernel_index'].rolling(window=ma1, center=False).mean()
    df['ma2'] = df['kernel_index'].rolling(window=ma2, center=False).mean()
    df=df.dropna()
    df['positions']=0
    df['positions'][ma1:] = np.where(df['ma1'][ma1:] >= df['ma2'][ma1:], 1, 0)


    # as positions only imply the holding
    # we take the difference to generate real trade signal
    df['signals'] = df['positions'].diff()


    return df

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
    df=pd.read_excel(r'data\HS300.xlsx',index_col=0)
    turnover=pd.read_excel(r'data\turnover.xlsx',index_col=0)
    new_df=CalBullishBearIndex(df,turnover)
    ma1=20
    ma2=60
    sig=signal_generation(new_df,ma1,ma2)
    ticker='000300.SH'
    plot(sig,ticker)


if __name__=='__main__':
    main()

