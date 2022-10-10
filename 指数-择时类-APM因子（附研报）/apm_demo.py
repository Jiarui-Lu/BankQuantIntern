import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


# 因子构造
class APM(object):
    def __init__(self, stock_30min_open, stock_30min_close, stock_1d_open, stock_1d_close):
        self.stock_30min_open = stock_30min_open
        self.stock_30min_close = stock_30min_close
        self.stock_1d_open = stock_1d_open
        self.stock_1d_close = stock_1d_close
        self.times = {
            '上午': ('10:00:00', '11:30:00'),  # 10:00-open,11:30-close
            '下午': ('13:30:00', '15:00:00'),  # 13:00-open,15:00-close
            'am1': ('10:00:00', '10:30:00'),  # 10:00-open,10:30-close
            'am2': ('11:00:00', '11:30:00'),  # 11:00-open,11:30-close
            'pm1': ('14:30:00', '15:00:00'),
            'pm2': ('13:30:00', '14:00:00')
        }
        self.stock_1d_ret = self.overnight_ret()
        self.daily_pct = np.log(self.stock_1d_close.iloc[:, 1:] / self.stock_1d_close.iloc[:, 1:].shift(1)).iloc[1:]

    def overnight_ret(self):
        stock_1d_ret = pd.DataFrame(index=self.stock_1d_close.index, columns=self.stock_1d_close.columns)
        for col in stock_1d_ret.columns:
            stock_1d_ret[col] = self.stock_1d_open[col] / self.stock_1d_close[col].shift(1) - 1
        return stock_1d_ret

    def _get_logret(self, start: str, end: str):
        '''
        获取收益率
        '''
        open_df = self.stock_30min_open.at_time(start)
        open_df.index = open_df.index.normalize()
        close_df = self.stock_30min_close.at_time(end)
        close_df.index = close_df.index.normalize()
        return np.log(close_df / open_df)

    def calc_resid(self, pos1, pos2, trade):
        trade_index = list(self.stock_1d_close.index).index(trade)
        if pos1 == '隔夜':
            am = self.overnight_ret().iloc[trade_index - 20:trade_index, :]
        else:
            am = self._get_logret(pos1[0], pos1[1]).iloc[trade_index - 20:trade_index, :]
        pm = self._get_logret(pos2[0], pos2[1]).iloc[trade_index - 20:trade_index, :]
        return self.regression(am), self.regression(pm)

    def regression(self, log_ret):
        x = log_ret[log_ret.columns[1:]].unstack().reset_index(
            level=0).sort_index()
        x.columns = ['code', 'log_ret']
        x['benchmark'] = log_ret['hs300']

        def _rls(df: pd.DataFrame) -> pd.Series:
            df = df.set_index('code')
            X = sm.add_constant(df['benchmark'])
            y = df['log_ret']
            mod = sm.OLS(y, X)
            res = mod.fit()

            return res.resid

        return x.groupby(level=0).apply(_rls)

    def calc_factor(self, method: str):
        factor_dic = {}
        method_dic = {'apm_raw': (self.times['上午'], self.times['下午']),
                      'apm_new': ('隔夜', self.times['下午']),
                      'apm_1': ('隔夜', self.times['pm1']),
                      'apm_2': (self.times['am1'], self.times['pm1']),
                      'apm_3': (self.times['am2'], self.times['pm2'])}
        for trade in self.stock_1d_close.index[21:]:
            trade_index = list(self.stock_1d_close.index).index(trade)
            am_resid, pm_resid = self.calc_resid(*method_dic[method], trade)
            dif = am_resid - pm_resid
            stat_ = dif.groupby(level='code').apply(
                lambda x: (x.mean() * np.sqrt(len(x))) / x.std())
            daily_pct_20 = self.daily_pct.iloc[trade_index - 20:trade_index, :].sum()
            stat_, daily_pct_20 = stat_.align(daily_pct_20)
            mod = sm.OLS(stat_, daily_pct_20)
            res = mod.fit()
            factor_df = pd.DataFrame(res.resid, index=res.resid.index)
            factor_df.columns = [method]
            factor_dic[trade] = factor_df
        df = pd.concat(factor_dic, names=['date', 'code'])
        return df


def get_next_returns(factor_df: pd.DataFrame) -> pd.DataFrame:
    '''
    获取下期收益率
    ------
    输入:
        factor_df:MuliIndex-level0-datetime.date level1-code columns - factors
        last_date:最后一期时间
    ------
    return pd.DataFrame
           MuliIndex level0-date level1-code value-next_ret
    '''
    days = pd.to_datetime(
        factor_df.index.get_level_values('date').unique().tolist())

    dic = {}
    for s, e in zip(days[:-1], days[1:]):
        stocks = factor_df.loc[s].index.get_level_values(
            'code').unique().tolist()
        a = stock_1d_close.loc[s, stocks]
        b = stock_1d_close.loc[e, stocks]
        dic[s] = b / a - 1

    df = pd.concat(dic).to_frame('next_ret')

    df.index.names = ['date', 'code']
    return df


def build_factor_data(factor_data: pd.DataFrame, returns: pd.DataFrame,
                      quantile: int):
    '''
    构造为alphalens通用的数据格式
    ------
    输入参数:
        factor_data:MuliIndex level0-date level1-code columns-因子名称
        returns:下期收益率get_next_returns的结果
    ------
    return Dict
           key-因子名称
           values-pd.DataFrame 其中 MuliIndex level0-date level1-code columns:factor|factor_quantile|1
           1就是next_returns
    '''

    def add_group(ser: pd.Series, quantile: int):
        factor_data = ser.to_frame('factor')
        grouper = [factor_data.index.get_level_values('date')]

        def quantile_calc(x, _quantiles, _bins=None, _zero_aware=False):
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()

        factor_quantile = factor_data.groupby(grouper)['factor'] \
            .apply(quantile_calc, quantile)
        factor_quantile.name = 'factor_quantile'
        return factor_quantile
        # return quantize_factor(factor_data, quantiles=quantile)

    def df_concat(df_list) -> pd.DataFrame:
        df = pd.concat(df_list, axis=1)
        df.columns = ['factor', 1, 'factor_quantile']
        df.index.names = ['date', 'asset']
        return df

    return {
        i: df_concat(
            (factor_data[i], returns, add_group(factor_data[i], quantile)))
        for i in factor_data.columns
    }


def get_factor_group_return(factor_df: pd.DataFrame) -> pd.DataFrame:
    '''
    获取因子的N分位收益
    ------
    输入参数:
        factor_df:MuliIndex level0-date level1-code columns:factor|factor_quantile|1
    '''

    return pd.pivot_table(factor_df.reset_index(),
                          index='date',
                          columns='factor_quantile',
                          values=1)


if __name__ == '__main__':
    stock_30min_open = pd.read_excel(r'data\stock_30min_data.xlsx', index_col=0, sheet_name=0)
    stock_30min_close = pd.read_excel(r'data\stock_30min_data.xlsx', index_col=0, sheet_name=1)
    stock_1d_open = pd.read_excel(r'data\stock_daily_data.xlsx', index_col=0, sheet_name=0)
    stock_1d_close = pd.read_excel(r'data\stock_daily_data.xlsx', index_col=0, sheet_name=1)
    apm = APM(stock_30min_open, stock_30min_close, stock_1d_open, stock_1d_close)
    factors = ['apm_raw', 'apm_new', 'apm_1', 'apm_2', 'apm_3']
    factor_df = apm.calc_factor('apm_raw')
    for f in factors[1:]:
        tmp_df = apm.calc_factor(f)
        tmp_col = tmp_df.columns[0]
        factor_df[tmp_col] = tmp_df[tmp_col]
    factor_df.to_excel(r'result\factor_df.xlsx')

    factor_df = pd.read_excel(r'result\factor_df.xlsx', index_col=[0, 1])
    # 获取未来一期收益
    returns = get_next_returns(factor_df)
    #
    # # 因子字典
    factor_dic = build_factor_data(factor_df, returns, 5)
    #
    # # 获取分组收益
    apm_ret_dic = {k: get_factor_group_return(v) for k, v in factor_dic.items()}

    # 分组收益画图
    rows = factor_df.shape[1]  # 组数

    color_map = ['#5D91A7', '#00516C', '#6BCFF6', '#00A4DC', '#6DBBBF',
                 '#008982']  # 设置颜色

    fig = plt.figure(figsize=(18, 5 * rows))

    benchmark = stock_1d_close[stock_1d_close.columns[0]].pct_change()[21:]
    cum_benchmark = (1 + benchmark).cumprod()[:-1]

    for factor_name, df in apm_ret_dic.items():
        mean = pd.DataFrame((df.sum() / len(df)) * 1000, columns=[factor_name])
        cum = (df + 1).cumprod()
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(2, 1, 1)
        ax1.bar(mean.index, mean[mean.columns[0]])
        ax1.set_xlabel('factor_quantile')
        ax1.set_ylabel('mean return(bps)')
        ax1.set_title('mean return per group')
        ax2 = plt.subplot(2, 1, 2)
        for i in range(len(color_map) - 1):
            ax2.plot(cum.index, cum[cum.columns[i]], label=cum.columns[i], color=color_map[i])
        ax2.plot(cum.index, cum_benchmark, label='benchmark', color=color_map[-1])
        ax2.set_xlabel('date')
        ax2.set_ylabel('cumulatiive return')
        ax2.set_title('{} cumulative return'.format(factor_name))
        plt.legend(loc='best')
        plt.savefig(r'result\{} return.jpg'.format(factor_name))
        plt.show()

    # 多空收益画图
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    for i, (factor_name, df) in enumerate(apm_ret_dic.items()):
        excess_return = (1 + df[df.columns[-1]] - df[df.columns[1]]).cumprod()
        ax.plot(df.index, excess_return, label=factor_name, color=color_map[i])
    ax.plot(cum_benchmark.index, cum_benchmark, label='benchmark', color=color_map[-1])
    ax.set_xlabel('date')
    ax.set_ylabel('cumulatiive return')
    ax.set_title('all factors cumulative return')
    plt.legend(loc='best')
    plt.savefig(r'result\all factors cumulative return.jpg')
    plt.show()
