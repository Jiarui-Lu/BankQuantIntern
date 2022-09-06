import functools
from typing import Callable, Dict, List, Tuple, Union

# import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TC import Company, Trader
# from sklearn import mixture
import warnings

warnings.filterwarnings("ignore")


def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x


def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.tanh(x)


def sign(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.where(x > 0.0, 1, 0)


def ReLU(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return sign(x) * x


def Exp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.exp(x)


def operators_max(x: Union[int, np.ndarray],
                  y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.maximum(x, y)


def operators_min(x: Union[int, np.ndarray],
                  y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.minimum(x, y)


def operators_add(x: Union[int, np.ndarray],
                  y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.add(x, y)


def operators_diff(x: Union[int, np.ndarray],
                   y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.subtract(x, y)


def operators_multiple(
        x: Union[int, np.ndarray],
        y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.multiply(x, y)


def get_x(x: Union[int, np.ndarray],
          y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return x


def get_y(x: Union[int, np.ndarray],
          y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return y


def x_is_greater_than_y(
        x: Union[int, np.ndarray],
        y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.greater(x, y) * 1.0


def Corr(x: Union[int, np.ndarray],
         y: Union[int, np.ndarray] = None) -> Union[int, np.ndarray]:
    return np.corrcoef(x, y, rowvar=False)[0][1]


# 基础参数
activation_funcs: List[Callable] = [identity, ReLU, sign, tanh]
binary_operators: List[Callable] = [
    operators_max,
    operators_min,
    operators_add,
    operators_diff,
    get_x,
    get_y,
    operators_multiple,
    x_is_greater_than_y,
]

# 读取储存数据
price = pd.read_csv('data\data.csv', index_col=[0], parse_dates=[0])


# print(price)

def plot_cum_returns(returns: pd.DataFrame, title: str = ''):
    plt.figure(figsize=(18, 6))
    returns['benchmark_cum'] = (1 + returns['benchmark']).cumprod()
    returns['000300_cum'] = (1 + returns['000300.SH']).cumprod()
    plt.plot(returns.index, returns['benchmark_cum'], label='benchmark')
    plt.plot(returns.index, returns['000300_cum'], label='000300.SH')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(r'result\{}.jpg'.format(title))
    plt.show()


def get_backtesting(params: Dict, data: pd.DataFrame, target_name: str, without_target: bool = False) -> pd.DataFrame:
    codes: List = data.columns.tolist()
    if without_target:
        test_col: List = codes
    else:
        test_col: List = [i for i in codes if i != target_name]

    train_data: np.ndarray = data[test_col]
    target: np.ndarray = data[target_name]

    company = Company(**params)

    company.fit(train_data.values, target.values)

    time_window: int = params['time_window'] - 1
    benchmark: pd.Series = target.iloc[time_window:]

    sign: np.ndarray = np.where(company.aggregate > 0, 1, 0)
    returns: pd.Series = benchmark.shift(-1) * sign
    df: pd.DataFrame = pd.concat((benchmark, returns), axis=1)
    df.columns = ['benchmark', target_name]

    return df


codes: List = price.columns.tolist()
target_name = '000300.SH'
params1 = {'trader_num': 100,
           'A': activation_funcs,
           'O': binary_operators,
           'stock_num': len(codes) - 1,
           'M': 10,
           'max_lag': 9,
           'l': 1,
           'time_window': 100,
           'Q': 0.5,
           'generate_method': 'BayesianGaussianMixture',
           'evaluation_method': 'ACC',
           'aggregate_method': 'Q',
           'seed': None}
params2 = {'trader_num': 100,
           'A': activation_funcs,
           'O': binary_operators,
           'stock_num': len(codes) - 1,
           'M': 10,
           'max_lag': 9,
           'l': 1,
           'time_window': 100,
           'Q': 0.5,
           'generate_method': 'GaussianMixture',
           'evaluation_method': 'ACC',
           'aggregate_method': 'Q',
           'seed': None}
params3 = {'trader_num': 100,
           'A': activation_funcs,
           'O': binary_operators,
           'stock_num': len(codes) - 1,
           'M': 10,
           'max_lag': 9,
           'l': 1,
           'time_window': 100,
           'Q': 0.5,
           'generate_method': 'Gaussian',
           'evaluation_method': 'ACC',
           'aggregate_method': 'Q',
           'seed': None}

# 5m-28.7
if __name__ == '__main__':
    ret1: pd.DataFrame = get_backtesting(params1, price, target_name, True)
    plot_cum_returns(ret1, 'BayesianGaussianMixture')
    ret1.to_excel(r'result\BayesianGaussianMixture.xlsx')
    ret2: pd.DataFrame = get_backtesting(params2, price, target_name, True)
    plot_cum_returns(ret2, 'GaussianMixture')
    ret2.to_excel(r'result\GaussianMixture.xlsx')
    ret3: pd.DataFrame = get_backtesting(params3, price, target_name, True)
    plot_cum_returns(ret3, 'Gaussian')
    ret3.to_excel(r'result\Gaussian.xlsx')
