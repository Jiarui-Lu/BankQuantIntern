import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    '''
    This reads the .csv stored at the 'filename' location and returns a DataFrame
    with two-level columns. The first level column contains the Exchange and the
    second contains the type of market data, e.g. bid/ask, price/volume.
    '''
    df = pd.read_csv(filename, index_col=0)

    return df


filename = r'data\HWG.csv'
market_data = read_data(filename)
# Select the first 250 rows
market_data_250 = market_data.iloc[:250]


# Set figsize of plot
plt.figure(figsize=(16, 10))


# Create a plot showing the bid and ask prices on different exchanges
def Plot_Bid_Ask(stock1='I-XCHNG', stock2='Z-XCHNG'):
    plt.plot(market_data_250.index, market_data_250['BidPrice-I-XCHNG'])
    plt.plot(market_data_250.index, market_data_250['AskPrice-Z-XCHNG'])
    plt.xticks([])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Bid Price on ' + stock1 + ' vs Ask Price on ' + stock2)
    plt.legend([stock1 + ' BidPrice', stock2 + ' AskPrice'])
    plt.savefig(r'result\Bid price versus Ask price.jpg')
    plt.show()


# Note arbitrage possible in case the BidPrice is higher than the AskPrice.
Plot_Bid_Ask()
market_data['I-Bid-Z-Ask-Spread'] = market_data['BidPrice-I-XCHNG'] - market_data['AskPrice-Z-XCHNG']
market_data['Z-Bid-I-Ask-Spread'] = market_data['BidPrice-Z-XCHNG'] - market_data['AskPrice-I-XCHNG']
# print(market_data)
# Create new DataFrame containing all arbitrage opportunities for comparison
arbitrage = market_data.loc[(market_data['I-Bid-Z-Ask-Spread'] > 0) | (market_data['Z-Bid-I-Ask-Spread'] > 0)]
# print(arbitrage)
# Design arbitrage strategy that gives all positions

positions = {'Timestamp': [],
             'Position-I-XCHNG': [],
             'Position-Z-XCHNG': []}

current_position = 0

for time, mkt_data_at_time in market_data.iterrows():

    if mkt_data_at_time['I-Bid-Z-Ask-Spread'] > 0:
        buy = min(mkt_data_at_time['BidVolume-I-XCHNG'],
                  mkt_data_at_time['AskVolume-Z-XCHNG'], (250 - current_position))
        spread = mkt_data_at_time['I-Bid-Z-Ask-Spread']
        positions['Timestamp'].append(time)
        positions['Position-I-XCHNG'].append(- buy - current_position)
        positions['Position-Z-XCHNG'].append(+ buy + current_position)
        current_position = buy

    elif mkt_data_at_time['Z-Bid-I-Ask-Spread'] > 0:
        buy = min(mkt_data_at_time['BidVolume-Z-XCHNG'],
                  mkt_data_at_time['AskVolume-I-XCHNG'], (250 - current_position))
        spread = mkt_data_at_time['Z-Bid-I-Ask-Spread']
        positions['Timestamp'].append(time)
        positions['Position-I-XCHNG'].append(+ buy + current_position)
        positions['Position-Z-XCHNG'].append(- buy - current_position)
        current_position = buy

positions = pd.DataFrame(positions).set_index('Timestamp')

positions.to_csv(r'result\positions_result.csv')

print(positions)
