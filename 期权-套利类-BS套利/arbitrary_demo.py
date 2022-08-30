import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from black_scholes import call_value, put_value, call_delta, put_delta, call_vega, put_vega
def read_data(filename):
    df = pd.read_csv(filename, index_col=0)[:1000]

    time_to_expiry = df.filter(like='TimeToExpiry')

    stock = df.filter(like='Stock')
    stock.columns = [stock.columns.str[-5:], stock.columns.str[:-6]]

    options = pd.concat((df.filter(like='-P'), df.filter(like='-C')), axis=1)
    options.columns = [options.columns.str[-3:], options.columns.str[:-4]]

    market_data = pd.concat((stock, options), axis=1)

    return time_to_expiry, market_data

# Read the market data
filename = r'data\Options Arbitrage.csv'
time_to_expiry, market_data = read_data(filename)
# print(time_to_expiry)
# print(market_data)

# Get a list of all instrument names including the stock, and of the options only
instrument_names = list(market_data.columns.get_level_values(0).unique())
# print(instrument_names)

option_names = instrument_names[1:]
# print(option_names)

# Add time_to_expiry to market_data
market_data['TTE'] = time_to_expiry['TimeToExpiry']

# Store timestamp in variable to prevent
# errors with multiplications and such
timestamp = market_data.index

# Set the Time to Expiry as Index
market_data = market_data.set_index('TTE')

# Create Empty Dictionaries
short_call_values = {}
long_call_values = {}
long_put_values = {}
short_put_values = {}
short_call_deltas = {}
long_call_deltas = {}
long_put_deltas = {}
short_put_deltas = {}
option_values = {}
option_deltas = {}

# Set known attributes
r = 0
sigma = 0.20

# Forloop to create new columns with Call/Put names
for option in option_names:
    # Retrieve K from the Option
    K = int(option[-2:])

    if 'C' in option:
        short_call_values[option] = []
        long_call_values[option] = []
        short_call_deltas[option] = []
        long_call_deltas[option] = []

        # Forloop to calculate short/long call values and deltas
        for time, stock_value in market_data.iterrows():
            short_call_values[option].append(call_value(
                stock_value['Stock', 'AskPrice'], K, time, r, sigma))
            long_call_values[option].append(call_value(
                stock_value['Stock', 'BidPrice'], K, time, r, sigma))
            long_call_deltas[option].append(call_delta(
                stock_value['Stock', 'BidPrice'], K, time, r, sigma))
            short_call_deltas[option].append(-call_delta(
                stock_value['Stock', 'AskPrice'], K, time, r, sigma))

        option_values['Short Call', option] = short_call_values[option]
        option_values['Long Call', option] = long_call_values[option]
        option_deltas['Short Call', option] = short_call_deltas[option]
        option_deltas['Long Call', option] = long_call_deltas[option]

    if 'P' in option:
        long_put_values[option] = []
        short_put_values[option] = []
        long_put_deltas[option] = []
        short_put_deltas[option] = []

        # Forloop to calculate short/long put values and deltas
        for time, stock_value in market_data.iterrows():
            long_put_values[option].append(
                put_value(stock_value['Stock', 'AskPrice'], K, time, r, sigma))
            short_put_values[option].append(
                put_value(stock_value['Stock', 'BidPrice'], K, time, r, sigma))
            long_put_deltas[option].append(
                put_delta(stock_value['Stock', 'AskPrice'], K, time, r, sigma))
            short_put_deltas[option].append(-put_delta(
                stock_value['Stock', 'BidPrice'], K, time, r, sigma))

        option_values['Long Put', option] = long_put_values[option]
        option_values['Short Put', option] = short_put_values[option]
        option_deltas['Long Put', option] = long_put_deltas[option]
        option_deltas['Short Put', option] = short_put_deltas[option]

 # Create DataFrames with index market_data
option_values = pd.DataFrame(option_values, index=market_data.index)
option_deltas = pd.DataFrame(option_deltas, index=market_data.index)

# Sort the DataFrames
option_values = option_values.reindex(sorted(option_values.columns), axis=1)
option_deltas = option_deltas.reindex(sorted(option_deltas.columns), axis=1)

# Rounding
option_values = round(option_values, 2)

# # Show DataFrames
# print('Option Values')
# print(option_values)
# print('Option Deltas')
# print(option_deltas)

# Create Columns for Black Scholes Value in the Data Set
# This is used for later calculations (algorithm and such)

for option in option_names:
    if "C" in option:
        market_data[option,
                    'Expected AskPrice'] = option_values['Short Call', option]
        market_data[option,
                    'Expected BidPrice'] = option_values['Long Call', option]
        market_data[option,
                    'Delta Short'] = option_deltas['Short Call', option].values
        market_data[option,
                    'Delta Long'] = option_deltas['Long Call', option].values

    elif "P" in option:
        market_data[option,
                    'Expected AskPrice'] = option_values['Short Put', option]
        market_data[option,
                    'Expected BidPrice'] = option_values['Long Put', option]
        market_data[option,
                    'Delta Short'] = option_deltas['Short Put', option].values
        market_data[option,
                    'Delta Long'] = option_deltas['Long Put', option].values

# Sort Columns
market_data = market_data.reindex(sorted(market_data.columns), axis=1)

# def option_opportunities(option):
#     '''
#     This function gives arbitrage opportunities based on whether the price
#     of the option is too high or too low. The results are used to 'eyeball'
#     if our final results match what this function displays. This works for
#     all Calls and Puts.
#     '''
#     if "C" in option:
#         expected1 = market_data[option][(market_data[option, 'BidPrice'] - market_data[option,
#                                                                                        'Expected AskPrice']) >= 0.10].drop('Expected BidPrice', axis=1)
#         expected2 = market_data[option][(market_data[option, 'Expected BidPrice'] -
#                                          market_data[option, 'AskPrice']) >= 0.10].drop('Expected AskPrice', axis=1)
#
#     elif "P" in option:
#         expected1 = market_data[option][(market_data[option, 'BidPrice'] - market_data[option,
#                                                                                        'Expected AskPrice']) >= 0.10].drop('Expected BidPrice', axis=1)
#         expected2 = market_data[option][(market_data[option, 'Expected BidPrice'] -
#                                          market_data[option, 'AskPrice']) >= 0.10].drop('Expected AskPrice', axis=1)
#
#     print('BidPrice is at least 0.10 higher than Expected AskPrice for Option ' + option)
#     print(expected1)
#     print('AskPrice is at least 0.10 lower than Expected BidPrice for Option ' + option)
#     print(expected2)
#     print('The amount of trades are', len(expected1) + len(expected2))

# option_opportunities('C80')


#  Create a Dictionary with Timestamp and Time to Expiry
# Index of market_data was changed earlier to time to expiry
trades = {('Timestamp', ''): timestamp,
          ('Time to Expiry', ''): market_data.index}

# Forloop that adds columns for the Call/Put Positions and Deltas
# Global function is a changing variable name based on the option
# For option C60 it will create a variable named positions_call_C60
for option in option_names:

    if 'C' in option:
        trades['Call Position', option] = []
        trades['Call Delta', option] = []
        globals()['positions_call_' + option] = 0

    if 'P' in option:
        trades['Put Position', option] = []
        trades['Put Delta', option] = []
        globals()['positions_put_' + option] = 0


# Forloop over the rows of market_data
for time, data in market_data.iterrows():

    max_delta = min(data['Stock', 'AskVolume'], data['Stock', 'BidVolume'])

    # Forloop over the option_names with conditions
    # if-statements if Call or Put + if Short/Long in Call or Put
    for option in option_names:

        if 'C' in option:

            # Short Call
            if (data[option, 'BidPrice'] - data[option, 'Expected AskPrice']) >= 0.10:
                short_call_volume = data[option, 'BidVolume']
                long_call_volume = 0

            # Long Call
            elif (data[option, 'Expected BidPrice'] - data[option, 'AskPrice']) >= 0.10:
                long_call_volume = data[option, 'AskVolume']
                short_call_volume = 0

            else:
                long_call_volume = short_call_volume = 0

            call_trade = long_call_volume - short_call_volume

            # Define variable, as set earlier. Note the first position is set to zero otherwise
            # One would get an error here since the variable is then not yet defined.
            globals()['positions_call_' + option] = call_trade + \
                globals()['positions_call_' + option]

            # Add Positions (cumulative)
            trades['Call Position', option].append(
                globals()['positions_call_' + option])

            if globals()['positions_call_' + option] >= 0:
                long_call_delta = data[option, 'Delta Long']
                short_call_delta = 0

            elif globals()['positions_call_' + option] < 0:
                short_call_delta = data[option, 'Delta Short']
                long_call_delta = 0

            # Add Deltas (cumulative)
            trades['Call Delta', option].append(
                abs(globals()['positions_call_' + option]) * (long_call_delta + short_call_delta))

        if 'P' in option:

            # Short Put
            if (data[option, 'BidPrice'] - data[option, 'Expected AskPrice']) >= 0.10:
                short_put_volume = data[option, 'BidVolume']
                long_put_volume = 0

            # Long Put
            elif (data[option, 'Expected BidPrice'] - data[option, 'AskPrice']) >= 0.10:
                long_put_volume = data[option, 'AskVolume']
                short_put_volume = 0

            else:
                long_put_volume = short_put_volume = 0

            put_trade = long_put_volume - short_put_volume

            globals()['positions_put_' + option] = put_trade + \
                globals()['positions_put_' + option]

            trades['Put Position', option].append(
                globals()['positions_put_' + option])

            if globals()['positions_put_' + option] >= 0:
                long_put_delta = data[option, 'Delta Long']
                short_put_delta = 0

            elif globals()['positions_put_' + option] < 0:
                short_put_delta = data[option, 'Delta Short']
                long_put_delta = 0

            trades['Put Delta', option].append(
                abs(globals()['positions_put_' + option]) * (long_put_delta + short_put_delta))

# Create DataFrame with Index Timestamp
trades = pd.DataFrame(trades).set_index('Timestamp')

# Sort Columns
trades = trades.reindex(sorted(trades.columns), axis=1)

# Calculate Total Option Delta (based on sorted columns)
trades['Total Option Delta', ''] = np.sum(
    trades['Call Delta'], axis=1) + np.sum(trades['Put Delta'], axis=1)

# Calculate Cumulative Stock Position (floored if positive, ceiled if negative)
trades['Stock Position', 'Stock'] = -np.where(trades['Total Option Delta', ''] >= 0, np.floor(
    trades['Total Option Delta', '']), np.ceil(trades['Total Option Delta', '']))

# Calculate remaining option delta (that remains unhedged)
# This delta is included in the Total Option Delta again which ensures
# It always remains below zero
trades['Remaining Option Delta', ''] = trades['Total Option Delta',
                                              ''] + trades['Stock Position', 'Stock']

# Show DataFrame

# Create trades_diff dataframe that gives all actual trades (not positions)
# Also drop columns that are not required for PnL calculations and such
trades_diff = trades.diff()[1:].drop(
    ['Call Delta', 'Put Delta', 'Time to Expiry', 'Total Option Delta', 'Remaining Option Delta'], axis=1)

# Drop the 'Call Position','Put Position' and 'Stock Position' top level
# Makes forlooping easier
trades_diff.columns = trades_diff.columns.droplevel(level=0)

# Since positions are not neccesarily zero at the last timestamp, final positions are calculated to be able to valuate these
final_positions = trades[-1:].drop(['Call Delta', 'Put Delta', 'Time to Expiry',
                                    'Total Option Delta', 'Remaining Option Delta'], axis=1)

final_positions.columns = final_positions.columns.droplevel(level=0)

# # Show DataFrames
# print('Actual Trades/Volumes')
# print(trades_diff.head())
#
# print("Final Positions that we currently 'own'")
# print(final_positions)

# Including Timestamp again to match trades_diff index
market_data['Timestamp'] = timestamp
market_data = market_data.set_index('Timestamp')

# Create Dataframe with index market_data (timestamp)
cashflow_dataframe = pd.DataFrame(index=market_data.index[1:])

# Forloop on all instruments (including stock) to calculate PnL
for instrument in instrument_names:

    Instrument_AskPrice = market_data[instrument, 'AskPrice'][1:]
    Instrument_BidPrice = market_data[instrument, 'BidPrice'][1:]

    cashflow_dataframe[instrument] = np.where(trades_diff[instrument] >= 0,
                                              trades_diff[instrument] * -
                                              Instrument_AskPrice,
                                              trades_diff[instrument] * -Instrument_BidPrice)

# # Show DataFrame & PnL
cashflow_dataframe.to_excel(r'result\total_cashflow.xls')
# print(cashflow_dataframe)
total_cashflow = cashflow_dataframe.sum().sum()
#
print('The total Cashflow is: €', round(total_cashflow, 2))

# Cumulative Cashflows
cashflow_cumulative = {}

for column in cashflow_dataframe.columns:

    cashflow_cumulative[column] = cashflow_dataframe[column].cumsum()

cashflow_cumulative = pd.DataFrame(cashflow_cumulative)

# # Show Cumulative Cashflow
# print(cashflow_cumulative)
#
# # Checking for Match
# print('This number should match the above number: €',
#       round(cashflow_cumulative[-1:].sum().sum(), 2))

# Create a new dataframe of trades with most columns dropped
trades_minimal = trades.drop(['Call Delta', 'Put Delta', 'Time to Expiry', 'Total Option Delta',
                              'Remaining Option Delta'], axis=1)

trades_minimal.columns = trades_minimal.columns.droplevel(level=0)

# Create Dataframe with market_data as index
valuation_dataframe = pd.DataFrame(index=market_data.index)

# Forloop to calculate valuations on every timestamp
for instrument in instrument_names:

    if 'C' in instrument:

        Instrument_AskPrice = market_data[instrument, 'AskPrice']
        Instrument_BidPrice = market_data[instrument, 'BidPrice']

        valuation_dataframe[instrument] = np.where(trades_minimal[instrument] > 0,
                                                   trades_minimal[instrument] *
                                                   Instrument_BidPrice,
                                                   trades_minimal[instrument] * Instrument_AskPrice)

    if 'P' in instrument:

        Instrument_AskPrice = market_data[instrument, 'AskPrice']
        Instrument_BidPrice = market_data[instrument, 'BidPrice']

        valuation_dataframe[instrument] = np.where(trades_minimal[instrument] > 0,
                                                   trades_minimal[instrument] *
                                                   Instrument_BidPrice,
                                                   trades_minimal[instrument] * Instrument_AskPrice)

    if 'S' in instrument:

        Instrument_AskPrice = market_data[instrument, 'AskPrice']
        Instrument_BidPrice = market_data[instrument, 'BidPrice']

        valuation_dataframe[instrument] = np.where(trades_minimal[instrument] > 0,
                                                   trades_minimal[instrument] *
                                                   Instrument_BidPrice,
                                                   trades_minimal[instrument] * Instrument_AskPrice)

# # Show DataFrame & Calculate total valuation
valuation_dataframe.to_excel(r'result\total_valuation.xls')
# print(valuation_dataframe)
total_valuation = valuation_dataframe[-1:].sum().sum()
#
print("Total valuation of our Position is currently: €", round(total_valuation, 2))

# Create Empty DataFrame
blackscholes_dataframe = {}

# Create Columns based on Option Names
for option in option_names:

    if 'C' in option:
        blackscholes_dataframe[option] = []

    if 'P' in option:
        blackscholes_dataframe[option] = []

# Forloop that calculates the margins and thus profits
for time, data in market_data.iterrows():

    for option in option_names:

        if "C" in option:
            margin1 = data[option, 'BidPrice'] - data[option, 'Expected AskPrice']
            margin2 = data[option, 'Expected BidPrice'] - data[option, 'AskPrice']

            if margin1 > 0.10:
                blackscholes_dataframe[option].append(margin1)

            elif margin2 > 0.10:
                blackscholes_dataframe[option].append(margin2)

            else:
                blackscholes_dataframe[option].append(0)

        elif "P" in option:
            margin1 = data[option, 'BidPrice'] - data[option, 'Expected AskPrice']
            margin2 = data[option, 'Expected BidPrice'] - data[option, 'AskPrice']

            if margin1 > 0.10:
                blackscholes_dataframe[option].append(margin1)

            elif margin2 > 0.10:
                blackscholes_dataframe[option].append(margin2)

            else:
                blackscholes_dataframe[option].append(0)

# Create DataFrame with index of market_data
blackscholes_dataframe = pd.DataFrame(blackscholes_dataframe, index=market_data.index)
blackscholes_dataframe.to_excel(r'result\total_blackscholes.xls')

# Calculate Black_Scholes Profit
total_blackscholes = (abs(trades_diff).drop('Stock', axis=1) * blackscholes_dataframe).sum().sum()
print('The total profit from Black Scholes is: €', round(total_blackscholes, 2))

# Blackscholes Dataframe times Volumes
blackscholes_dataframe = pd.DataFrame(
    abs(trades_diff).drop('Stock', axis=1) * blackscholes_dataframe)

# Cumulative Blackscholes
blackscholes_cumulative = {}

for column in blackscholes_dataframe.columns:

    blackscholes_cumulative[column] = blackscholes_dataframe[column].cumsum()

blackscholes_cumulative = pd.DataFrame(blackscholes_cumulative)

# Show Dataframe
# print(blackscholes_cumulative)
# Last PnL in the Table should match the below value
print('The total profit generated from the Option Arbitrage strategy is: €',
      round(total_cashflow + total_valuation + total_blackscholes, 2))