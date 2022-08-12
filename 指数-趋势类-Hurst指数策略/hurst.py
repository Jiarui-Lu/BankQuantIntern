from math import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")


def Hurst(X, T, step):
    X = np.array(X)
    nX = X.shape[0]
    hurst = np.zeros(nX - T + 1)  # hurst值的序列
    eh = np.zeros(nX - T + 1)  # E(H)值的序列
    for i in range(0, nX - T + 1, step):
        XX = X[i:i + T]
        nMax = int(T / 2)  # 区间最大长度
        narray = []
        RS = []
        eRS = []
        for j in range(10, nMax):
            A = int(round(1.0 * T / j))  # 区间个数
            narray.append(j)
            RS.append(SingleRSn(A, XX, j))
            eRS.append(Peters(j))
        hurst[i] = Linear_Regression(RS, narray)  # 线性回归
        eh[i] = Linear_Regression(eRS, narray)
    return hurst, eh


# 求RSn
def SingleRSn(A, XX, n):
    RS = np.zeros(A)
    for i in range(A):
        XXX = XX[i * n:(i + 1) * n]
        Ma = XXX.mean()
        Sa = XXX.std()
        XXX = XXX - Ma
        cumXXX = XXX.cumsum()  # 累积离差
        Ra = cumXXX.max() - cumXXX.min()  # 极差

        RS[i] = 1.0 * Ra / Sa  # 重标极差
    return np.mean(RS)


# E[(R / S)n]的计算我们采用 Peters 的方法
def Peters(n):
    preRe = 1.0 * ((n - 0.5) / n) * (n * np.pi / 2) ** (-0.5)
    sumR = 0
    for i in range(1, n):
        sumR += sqrt((n - i) / i)
    return 1.0 * preRe * sumR


def Linear_Regression(RS, n):
    RSlog = np.log(RS)
    nlog = np.log(n)
    N = len(n)
    RSlog.shape = (N, 1)
    nlog.shape = (N, 1)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(nlog, RSlog)
    return lr.coef_

inter = 233 #滑动时间窗口
step = 1#移动步长

'''读取数据'''
data = pd.read_excel(r"data\price.xlsx")
leftdata = data['close'][inter-1:] #窗口区间之外的数据
leftdata.index = np.arange(len(leftdata))
# print(data)
hurst,eh = Hurst(data,inter,step) #返回hurst值和E（h）的值
# print(hurst,eh)
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hurst)
plt.plot(eh)
plt.subplot(2,1,2)
plt.plot(leftdata)

hurst = pd.DataFrame(hurst)
eh = pd.DataFrame(eh)

result =  pd.concat([hurst,eh,leftdata],axis=1,join='inner')
excelname = r'result\hurst.xlsx'
writer = pd.ExcelWriter(excelname)
result.to_excel(writer,'Sheet1')
writer.save()

plt.show()