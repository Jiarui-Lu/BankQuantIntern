import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from typing import (Callable)
from DE_algorithm import *
import matplotlib.pyplot as plt

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')

price = pd.read_excel(r'data\daily_ETFdata.xlsx', index_col=0)
hs300 = pd.read_excel(r'data\hs300_data.xlsx', index_col=0)
price = price.loc[hs300.index, :]
pct_chg = price.pct_change().iloc[1:]
'''使用sklearn接口用于超参'''


class DEPortfolioOpt(BaseEstimator):

    def __init__(self, func: Callable, size_pop: int, max_iter: int, F: float, proub_mut: float) -> None:
        self.func = func  # 目标函数
        self.size_pop = size_pop  # 种群大小
        self.max_iter = max_iter  # 迭代次数
        self.F = F  # 变异系数
        self.proub_mut = proub_mut  # 变异概率

    def fit(self, returns: pd.DataFrame) -> np.array:
        '''获取优化后的权重'''

        self.de = DE(func=lambda w: self.func(returns @ w),
                     n_dim=returns.shape[1],
                     size_pop=self.size_pop,
                     max_iter=self.max_iter,
                     F=self.F,
                     prob_mut=self.proub_mut,
                     lb=[0],
                     ub=[1],
                     constraint_eq=[lambda x: 1 - np.sum(x)])

        w, self.maxDrawdown = self.de.run()

        return pd.Series(w, index=returns.columns)

    def predict(self, returns) -> float:
        '''获取权重后的收益率'''
        w = self.fit(returns)
        return returns @ w

    def score(self, returns) -> float:
        '''评分根据优化后的回撤决定'''
        return -self.maxDrawdown


def cum_returns(returns, starting_value=0, out=None):
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out


def max_drawdown(returns, out=None):
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    np.nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


# 网格超参
param_grid = [{'F': np.arange(0.4, 0.6, 0.1),
               'proub_mut': np.arange(0.9, 1.1, 0.1)}]

de_portfolio = DEPortfolioOpt(max_drawdown, 50, 50, 0.5, 1)

# cv默认为3flod
grid_search = GridSearchCV(de_portfolio, param_grid)

grid_search.fit(pct_chg)

equal = -max_drawdown(pct_chg.mean(axis=1))

# 回撤情况
maxdrawdownList = [-max_drawdown(pct_chg @ w) for w in grid_search.best_estimator_.de.generation_best_X]

fig, axes = plt.subplots(1, 2, figsize=(6 * 2, 4))

axes[0].set_title('拟合曲线')
axes[0].plot(maxdrawdownList, label='拟合的回撤')
axes[0].axhline(equal, color='r', label='等权回撤')
plt.legend(loc='best')

axes[1].set_title('拟合度变化')
axes[1].plot(grid_search.best_estimator_.de.generation_best_Y)
plt.savefig(r'result\拟合曲线和拟合度变化.jpg')
plt.show()

best_x = grid_search.best_estimator_.de.best_x  # 最优组合

# 样本内
in_sample = pct_chg.loc[:pct_chg.index[1000]]
# 样本外
out_of_sample = pct_chg.loc[pct_chg.index[1000]:]

fig, ax = plt.subplots(figsize=(18, 4))
ax.set_title('权重优化对比')

cum_returns(in_sample @ best_x).plot(ax=ax, label='in_sample', color='r')
cum_returns(out_of_sample @ best_x).plot(ax=ax, label='out_of_sample', color='r')
cum_returns(out_of_sample.mean(axis=1)).plot(ax=ax, label='equal', color='darkgray')
cum_returns(in_sample.mean(axis=1)).plot(ax=ax, label='equal', color='darkgray')
plt.axvline(pct_chg.index[1000], color='black')

h1, l1 = ax.get_legend_handles_labels()
h1 = h1[:-1]
l1 = l1[:-1]
plt.legend(h1, l1)
plt.savefig(r'result\权重优化对比.jpg')
plt.show()


# # 计算权重
def get_weight(opt: DEPortfolioOpt, pct_df, tradedt) -> pd.Series:
    '''获取组合权重'''
    tradedt_index = list(pct_df.index).index(tradedt)
    pct_df = pct_df.iloc[tradedt_index - 126:tradedt_index, :]
    return opt.fit(pct_df)


#
#
# 用于回测
def BackTesting(hs300, quarter, pct_chg, weight_df) -> pd.DataFrame:
    '''
    timeRange:时间区间
    wieght:权重key-trade,values:pd.Series
    ----------------
        return 等权、优化权重、基准的收益率序列
    '''
    pct_chg = pct_chg.loc[hs300.index, :]
    # 缓存容器
    ret_df = pd.DataFrame(data=np.zeros((len(hs300.index), 2)),
                          index=hs300.index,
                          columns=['opt_ret', 'equal_ret'])
    # 模拟回测
    for tradedt in range(len(quarter)):
        if tradedt == len(quarter) - 1:
            break
        weight_ser = weight_df.loc[quarter[tradedt], :]
        pct_chg_tmp = pct_chg.loc[quarter[tradedt]:quarter[tradedt + 1], :]
        pct_chg_tmp.mul(weight_ser, axis=1)
        col_sum = pct_chg_tmp.apply(lambda x: x.sum(), axis=1)
        ret_df.loc[pct_chg_tmp.index, 'opt_ret'] = col_sum  # 计算组合权重
        ret_df.loc[pct_chg_tmp.index, 'equal_ret'] = pct_chg_tmp.mean(axis=1)
    #
    benchmark = hs300.pct_change()
    ret_df['benchmark'] = benchmark
    ret_df.to_excel(r'result\opt_price.xlsx')
    return ret_df.dropna()


# 计算每期的权重
'''
每季度获取前125日的数据进行组合优化
在观察日的基础上前移一天避免未来数据
'''
price = pd.read_excel(r'data\daily_ETFdata.xlsx', index_col=0)
pct_chg = price.pct_change().iloc[1:]
quarter = pd.read_excel(r'data\seasonly_ETFdata.xlsx', index_col=0).index

etf_weight_dic = {tradeDt: get_weight(DEPortfolioOpt(max_drawdown, 50, 50, 0.5, 1), pct_chg, tradeDt) for tradeDt in
                  quarter}
weight_df = pd.DataFrame(etf_weight_dic.values(), index=etf_weight_dic.keys())
weight_df.to_excel(r'result\ETF_weight.xlsx')

weight_df = pd.read_excel(r'result\ETF_weight.xlsx', index_col=0)

ret_df = BackTesting(hs300, quarter, pct_chg, weight_df)

opt_CUM = (1 + ret_df['opt_ret']).cumprod()
equal_CUM = (1 + ret_df['equal_ret']).cumprod()
benchmark_CUM = (1 + ret_df['benchmark']).cumprod()
plt.figure()
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(ret_df.index, opt_CUM, label='持仓优化')
ax1.plot(ret_df.index, benchmark_CUM, label='沪深300')
ax1.plot(ret_df.index, equal_CUM, label='等权回撤')

plt.legend(loc='best')
plt.xlabel('时间')
plt.ylabel('净值')
plt.title('各策略净值曲线对比')
plt.savefig(r'result\return backtest.jpg')
plt.show()
