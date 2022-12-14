# -*- coding: utf-8 -*-
'''
Created on 2016/2/14

@author: tang
'''

import csv
import os
import types
import math
import copy as cp
import numpy as np
import ITSCreator
from numpy.matlib import rand
import pandas as pd


# 获取文件路径
# def GetFileList(dir, fileList):
#     for s in os.listdir(dir):
#         newDir=os.path.join(dir,s)
#         GetFileList(newDir, fileList)
#     return fileList
#
def _recurrentFunc(tradeDateFileList):
    itsValue = []
    utsValue = []
    msdValue = []
    for index in tradeDateFileList:
        try:
            y = ITSCreator.itsFutSignal(index)
            itsValue.append(y.itsSig)
            utsValue.append(y.utsSig)
            msdValue.append(y.msdSig)
        except:
            pass
    return (itsValue, utsValue, msdValue)


def main():
    # 交易数据文件名获取
    tradeDateFileList = []
    for s in os.listdir('download_data'):
        tradeDateFileList.append(s)
    # tradeDateFileList = GetFileList('data', [])
    # print(tradeDateFileList)

    # 情绪指标值 ITS UTS MSD
    # itsValue = [ITSCreator.itsFutSignal(index).itsSig for index in tradeDateFileList]
    # utsValue = [ITSCreator.itsFutSignal(index).utsSig for index in tradeDateFileList]
    # msdValue = [ITSCreator.itsFutSignal(index).msdSig for index in tradeDateFileList]
    itsValue, utsValue, msdValue = _recurrentFunc(tradeDateFileList=tradeDateFileList)

    # 当前分析数据个数
    msdNum = len(msdValue)
    # 作为平均值最近的天数
    dayslen = 23
    msdAvgSum = [sum(msdValue[index:index + dayslen]) for index in range(msdNum - dayslen)]
    msdAvg = [1.0 * index / dayslen for index in msdAvgSum]
    msdleftValue = msdValue[dayslen:msdNum]

    msdleftNum = len(msdleftValue)
    # MSD与平均MSD的差值
    difflefttoavg = 0.0
    futSignal = []
    for index in range(msdleftNum):
        if (msdleftValue[index] - msdAvg[index]) > difflefttoavg:
            futSignal.append(1)
        elif (msdleftValue[index] - msdAvg[index]) < -difflefttoavg:
            futSignal.append(-1)
        else:
            futSignal.append(0)

    # 实际涨跌情况
    realSignal = [1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1,
                  1, -1, -1, -1, 1, -1, 0, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
                  -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1]

    # 实际数据取得是次日的涨跌指标数据
    realSignal = realSignal[dayslen + 1:msdNum]
    futSignal = np.array(futSignal)
    realSignal = np.array(realSignal)
    # 无效的记录个数
    cntinvalue = 0
    hit = 0
    hitRate = 0
    compNum = msdleftNum - 1
    for index in range(compNum):
        if futSignal[index] == realSignal[index]:
            hit += 1
        if futSignal[index] == 0:
            cntinvalue += 1
    hitRate = 1.0 * hit / (compNum - cntinvalue)

    print('testlen----')
    print(compNum - cntinvalue)
    print('hit--------')
    print(hit)
    print('Rate:------')
    print(hitRate)
    df1=pd.DataFrame([compNum - cntinvalue,hit,hitRate],columns=['testlen,hit,Rate'])
    df1.to_excel(r'result\result.xls')


if __name__ == '__main__':
    main()
