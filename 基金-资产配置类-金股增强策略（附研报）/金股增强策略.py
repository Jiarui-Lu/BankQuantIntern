import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

gold_stock_frame = pd.read_excel(r'data\gold_stock_frame.xlsx')

stats_write_date = gold_stock_frame.groupby('end_date')['write_date'].max()
stats_write_date = stats_write_date.to_frame()
stats_write_date.columns = ['write_date']


def tradeday_of_month(watch_dt: str):
    """查询该交易日是当月的第N个交易日"""
    tradedays = pd.read_excel(r'data\tradedays.xlsx')
    watch_dt = datetime.datetime.strptime(watch_dt, '%Y-%m-%d')
    current_tradedays = []
    for d in tradedays['date']:
        d = datetime.datetime.strptime(str(d)[:10], '%Y-%m-%d')
        if d.year == watch_dt.year and d.month == watch_dt.month:
            current_tradedays.append(d)
    if watch_dt in current_tradedays:
        return current_tradedays.index(watch_dt) + 1
    else:
        day = watch_dt.day
        day_range = [1, -1, 2, -2, 3, -3]
        for i in day_range:
            day = day + i
            for j in current_tradedays:
                if day == j.day:
                    return current_tradedays.index(j) + 1


stats_write_date['DayOfMonth'] = stats_write_date['write_date'].apply(lambda x: tradeday_of_month(x))

fig, ax = plt.subplots(figsize=(18, 12))
bar_ax = ax.bar(stats_write_date.index, stats_write_date['DayOfMonth'])

max_idx = stats_write_date.index.get_loc(stats_write_date['DayOfMonth'].idxmax())
bar_ax[max_idx].set_color('red')
avg_num: float = stats_write_date['DayOfMonth'].mean()
ax.axhline(avg_num, color='darkgray', ls='--')
plt.xticks(rotation=90)
plt.savefig(r'result\平均发布日期.jpg')
plt.show()
print(f'平均在当月的第{round(avg_num, 2)}发布')

stock = pd.read_excel(r'data\monthlydata.xlsx', index_col=0, sheet_name=0).T
stock.index = [str(i)[:9] for i in stock.index]
sw = pd.read_excel(r'data\monthlydata.xlsx', index_col=0, sheet_name=1).T
stock = stock.sort_values(by=stock.columns[-1]).T
sw = sw.sort_values(by=sw.columns[-1]).T
stock_price = stock.iloc[:-1, :]
stock_industry = stock.iloc[-1, :]
stock_price = stock_price.pct_change().iloc[1:, :]
sw_price = sw.iloc[:-1, :]
sw_price = sw_price.pct_change().iloc[1:, :]
sw_industry = sw.iloc[-1, :]
excess_ret = pd.DataFrame(columns=stock_price.columns, index=stock_price.index)
i = 0
for col in stock_price.columns:
    col_index = list(stock_price.columns).index(col)
    sw_index = list(sw_industry).index(stock_industry[col_index])
    excess_ret.iloc[:, i] = stock_price.iloc[:, col_index] - sw_price.iloc[:, sw_index]
    i = i + 1

excess_ret.index = [str(i)[:10] for i in excess_ret.index]
stock_price.index = excess_ret.index
gold_stock_frame['next_ret'] = 0
gold_stock_frame['industry_excess'] = 0
for i in gold_stock_frame.index:
    try:
        code = gold_stock_frame.loc[i, 'ticker_symbol_map_sec_id']
        monthend = str(gold_stock_frame.loc[i, 'end_date'])
        if monthend == '2022-04-30':
            monthend = '2022-04-29'
        elif monthend == '2022-07-31':
            monthend = '2022-07-29'
        else:
            pass
        gold_stock_frame.loc[i, 'industry_excess'] = excess_ret.loc[monthend, code][0]
        gold_stock_frame.loc[i, 'next_ret'] = stock_price.loc[monthend, code][0]
    except:
        pass
gold_stock_frame.to_excel(r'result\金股数据库.xlsx')

gold_stock_frame = pd.read_excel(r'result\金股数据库.xlsx', index_col=0)
for i in gold_stock_frame.index:
    if gold_stock_frame.loc[i, 'next_ret'] == 0:
        gold_stock_frame = gold_stock_frame.drop(labels=i)
gold_stock_frame = gold_stock_frame.reset_index()
del gold_stock_frame['index']
gold_stock_frame = gold_stock_frame.set_index('author')
gold_stock_frame = gold_stock_frame.set_index('end_date', append=True)
beta_distribution = {}
for author, date in gold_stock_frame.index:
    author_df = gold_stock_frame.loc[author, :]
    alpha = 0
    beta = 0
    for i in author_df.index:
        try:
            if author_df.loc[i, 'industry_excess'] > 0:
                alpha = alpha + 1
            else:
                beta = beta + 1
        except:
            if author_df.loc[i, 'industry_excess'][0] > 0:
                alpha = alpha + 1
            else:
                beta = beta + 1
    miu = alpha / (alpha + beta)
    beta_distribution[author] = miu
gold_stock_frame = gold_stock_frame.reset_index()
gold_stock_frame['miu'] = [beta_distribution[a] for a in gold_stock_frame['author']]
gold_stock_frame['qcut'] = pd.qcut(gold_stock_frame['miu'], q=5, labels=[str(i) for i in range(1, 6)])
gold_stock_frame = gold_stock_frame.set_index('end_date')
gold_stock_frame = gold_stock_frame.set_index('qcut', append=True)
month_end = []
for date, q in gold_stock_frame.index:
    if date not in month_end:
        month_end.append(date)
factor_df = pd.DataFrame(index=month_end, columns=[str(i) for i in range(1, 6)])
for date, q in gold_stock_frame.index:
    date_df = gold_stock_frame.loc[date, :]
    for i in [str(i) for i in range(1, 6)]:
        factor_df.loc[date, i] = date_df.loc[i, 'next_ret'].mean()
factor_df.to_excel(r'result\factor_df.xlsx')

fig = plt.figure(figsize=(16, 8))
for col in factor_df.columns:
    cum_return = (1 + factor_df[col]).cumprod()
    plt.plot(factor_df.index, cum_return, label=col)
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('return')
plt.title('Strategy return')
plt.savefig(r'result\Strategy return.jpg')
plt.show()
