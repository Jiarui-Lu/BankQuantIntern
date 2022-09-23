import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')


# 画出蜡烛图
def plot_candlestick(df: pd.DataFrame, title: str = '', **kwargs):
    '''
    画出蜡烛图
    -----------
        price:index-date columns-OHLC
                index为datetime
        kwargs:为pathpatch时则画出需要标记的k线
    '''

    df = df.copy()
    data = df[['open', 'high', 'low', 'close']]

    # 生成横轴的刻度名字
    date_tickers = df.index

    day_quotes = [tuple([i] + list(quote[:])) for i, quote in enumerate(data.values)]
    fig, ax = plt.subplots(figsize=(18, 4))
    plt.title(title)

    def format_date(x, pos=None):
        if x < 0 or x > len(date_tickers) - 1:
            return ''
        return date_tickers[int(x)]

    candlestick_ohlc(ax, day_quotes, colordown='g', colorup='r', width=0.2)

    if 'pathpatch' in kwargs:
        ax.add_patch(kwargs['pathpatch'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date));
    ax.grid(True)
    plt.savefig(r'result\{}'.format(title))
    plt.show()


# 标记需要标记的K线
def get_mark_data(price: pd.DataFrame, target_date: list):
    '''
    标记出k线
    -----------
        price:index-date columns-OHLC
            index为datetime
        target_date:list 日期格式yyyy-mm-dd
    '''

    df = price[['open', 'high', 'low', 'close']].copy()

    if isinstance(target_date, list):
        target_data = [target_date]

    vertices = []
    codes = []

    idx = [df.index.get_loc(i) for i in target_date]

    for i in idx:
        low = df['low'].iloc[i] * (1 - 0.001)
        high = df['high'].iloc[i] * (1 + 0.001)

        codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        vertices += [(i - 0.5, low), (i - 0.5, high), (i + 0.5, high), (i + 0.5, low), (i - 0.5, low)]

    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor='None', edgecolor='black', lw=2)

    return pathpatch


def candlestick_ohlc(ax, quotes, width=0.2, colorup='k', colordown='r',
                     alpha=1.0):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        As long as the first 5 elements are these values,
        the record can be as long as you want (e.g., it may store volume).
        time must be in float days format - see date2num
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added
    """
    return _candlestick(ax, quotes, width=width, colorup=colorup,
                        colordown=colordown,
                        alpha=alpha, ochl=False)


def _candlestick(ax, quotes, width=0.2, colorup='k', colordown='r',
                 alpha=1.0, ochl=True):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes
    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added
    """

    OFFSET = width / 2.0

    lines = []
    patches = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


price = pd.read_excel(r'data\HS300.xlsx', index_col=0)

plot_candlestick(price, '沪深300指数蜡烛图',
                 pathpatch=get_mark_data(price, ['2021-09-06', '2022-01-05', '2022-05-26']))

low_data = pd.read_excel(r'data\stock.xlsx', index_col=0, sheet_name=0)
high_data = pd.read_excel(r'data\stock.xlsx', index_col=0, sheet_name=1)
open_data = pd.read_excel(r'data\stock.xlsx', index_col=0, sheet_name=2)
close_data = pd.read_excel(r'data\stock.xlsx', index_col=0, sheet_name=3)
#
month = pd.read_excel(r'data\monthly_data.xlsx', index_col=0)

uppershadow = high_data - np.maximum(close_data, open_data)
#
std_uppershadow = uppershadow.div(uppershadow.rolling(window=5).mean())
std_uppershadow = std_uppershadow.iloc[4:, :].dropna(axis=1)

lowershadow = np.minimum(close_data, open_data) - low_data
std_lowershadow = lowershadow / lowershadow.rolling(window=5).mean()
std_uppershadow = std_uppershadow.iloc[4:, :].dropna(axis=1)
std_lowershadow = std_lowershadow.iloc[4:, :].dropna(axis=1)

upper_shadow_mean = pd.DataFrame()
upper_shadow_std = pd.DataFrame()
lower_shadow_mean = pd.DataFrame()
lower_shadow_std = pd.DataFrame()
for m in std_uppershadow.index:
    if m in month.index:
        m_index = list(std_uppershadow.index).index(m)
        if m_index >= 20:
            df_tmp = std_uppershadow.iloc[m_index - 20:m_index, :]
        else:
            df_tmp = std_uppershadow.iloc[:m_index, :]
        row_mean = df_tmp.apply(lambda x: x.mean(), axis=0)
        row_std = df_tmp.apply(lambda x: x.std(), axis=0)
        upper_shadow_mean = pd.concat([upper_shadow_mean, pd.DataFrame(row_mean)], axis=0)
        upper_shadow_std = pd.concat([upper_shadow_std, pd.DataFrame(row_std)], axis=0)
for m in std_lowershadow.index:
    if m in month.index:
        m_index = list(std_lowershadow.index).index(m)
        if m_index >= 20:
            df_tmp = std_lowershadow.iloc[m_index - 20:m_index, :]
        else:
            df_tmp = std_lowershadow.iloc[:m_index, :]
        row_mean = df_tmp.apply(lambda x: x.mean(), axis=0)
        row_std = df_tmp.apply(lambda x: x.std(), axis=0)
        lower_shadow_mean = pd.concat([lower_shadow_mean, pd.DataFrame(row_mean)], axis=0)
        lower_shadow_std = pd.concat([lower_shadow_std, pd.DataFrame(row_std)], axis=0)

factor_normal = pd.DataFrame(0, index=[i for i in range(len(month.index) * len(std_uppershadow.columns))],
                             columns=['date', 'code'])
_index = 0
factor_normal['upper_shadow_mean'] = list(upper_shadow_mean[upper_shadow_mean.columns[0]])
factor_normal['upper_shadow_std'] = list(upper_shadow_std[upper_shadow_std.columns[0]])
factor_normal['lower_shadow_mean'] = list(lower_shadow_mean[lower_shadow_mean.columns[0]])
factor_normal['lower_shadow_std'] = list(lower_shadow_std[lower_shadow_std.columns[0]])
factor_normal['next_ret'] = 0
for m in month.index:
    for c in std_uppershadow.columns:
        factor_normal.loc[_index, 'next_ret'] = month.loc[m, c]
        factor_normal.loc[_index, 'date'] = m
        factor_normal.loc[_index, 'code'] = c
        _index = _index + 1
index = pd.MultiIndex.from_frame(factor_normal.loc[:, ['date', 'code']])
factor_normal.index = index
del factor_normal['date']
del factor_normal['code']
factor_normal.to_excel(r'result\factor normal.xlsx')


# 分组
def get_group(df: pd.DataFrame, target_factor: str, num_group: int = 5) -> pd.DataFrame:
    '''
    分组
    ----
        target_factor:目标列
        num_group:默认分5组
    '''
    df = df.copy()
    df = df.dropna(subset=[target_factor])
    label = [i for i in range(1, num_group + 1)]
    df['group'] = df.groupby(level='date')[target_factor].transform(
        lambda x: pd.qcut(x.dropna(), 5, labels=label))

    return df


# 计算分组收益率
def get_algorithm_return(factor_df: pd.DataFrame) -> pd.DataFrame:
    returns = pd.pivot_table(factor_df.reset_index(
    ), index='date', columns='group', values='next_ret')
    returns.columns = [str(i) for i in returns.columns]

    returns.index = pd.to_datetime(returns.index)
    label = ['G%s' % i for i in range(1, 6)]
    returns.columns = label

    return returns


def add_benchmark(factor: pd.DataFrame, col_name: str) -> pd.DataFrame:
    '''获取因子分组收益与基准收益'''

    group_df = get_group(factor, col_name)  # 因子升序排列
    returns = get_algorithm_return(group_df)

    benchmark = pd.read_excel(r'data\HS300.xlsx', index_col=0)
    benchmark = benchmark.reindex(returns.index)
    benchmark = benchmark['close'].pct_change().fillna(0)
    returns['benchmark'] = benchmark

    returns['excess_ret'] = returns['G1'] - returns['G5']
    cum_df = np.exp(np.log1p(returns).cumsum())

    return cum_df


# 画图
col_name = list(factor_normal.columns[:-1])
# 净值情况
factorCumList = [add_benchmark(factor_normal, col) for col in col_name]

for i in range(len(factorCumList)):
    nav = factorCumList[i]
    fig = plt.figure()
    plt.plot(nav.index, nav['G1'], color='Navy', label='G1')
    plt.plot(nav.index, nav['G2'], color='LightGrey', ls='-.', label='G2')
    plt.plot(nav.index, nav['G3'], color='DimGray', ls='-.', label='G3')
    plt.plot(nav.index, nav['G4'], color='DarkKhaki', ls='-.', label='G4')
    plt.plot(nav.index, nav['G5'], color='LightSteelBlue', label='G5')
    plt.axhline(1, color='black', lw=0.5)
    # 多空单独反应
    plt.plot(nav.index, nav['excess_ret'], color='r', ls='--', label='excess_ret')
    plt.plot(nav.index, nav['benchmark'], color='black', ls='--', label='benchmark')
    plt.legend(loc='best')
    plt.savefig(r'result\{}'.format(col_name[i]))
    plt.grid(True)
    plt.show()
