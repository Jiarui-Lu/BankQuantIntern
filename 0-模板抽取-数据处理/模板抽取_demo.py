'''
出现相关报错：
（1）在使用scipy时出现ImportError: cannot import name 'logsumexp'的问题
打开..\venv\lib\site-packages\gensim\models\ldamodel.py
将from scipy.misc import logsumexp
改为from scipy.special import logsumexp
（2）在调用h5模型时出现AttributeError: 'str' object has no attribute 'decode'的问题
python版本存在一定问题
打开..\venv\lib\site-packages\keras\engine\saving.py
删去各个错误行的.decode('utf-8')
'''

import pandas as pd
import numpy as np
import datetime
import re
from keras.models import load_model
from gensim.models.word2vec import Word2Vec
from pathlib import Path


class PrepareData(object):
    def __init__(self, df_collist, *df):
        self.df_collist = df_collist
        self.df = df  # 以tuple格式传入可变参数

    '''
        模板 - 日期处理方法：
        (1)
        日期统一格式化，有的通过csv读入的日期列格式显示float，有的是str，
        各种类型样式的输入20220105、2022 / 01 / 05、2022 - 01 - 05
        等全部规范成你给定参数样式，比如全部转为“ % Y - % m - % d”
        (2)
        日期升序、降序
    '''

    def TimeStandardlized(self, df, colname, ascending=True):  # 默认升序
        if type(df[colname][0]) == np.int64:
            df[colname] = df[colname].apply(lambda x: str(datetime.datetime.strptime(str(x), '%Y%m%d'))[:10])
        else:
            df[colname] = df[colname].apply(lambda x: str(x))

            def process(x):
                compile = re.compile(r'[^0-9]')
                symbol = compile.findall(x)[0]
                tmp_list = x.split(symbol)
                result = ''.join(tmp_list)
                return result

            df[colname] = df[colname].apply(lambda x: str(datetime.datetime.strptime(process(x), '%Y%m%d'))[:10])
        df = df.sort_values(by=colname, ascending=ascending)
        df = df.rename(columns={colname: 'date'})
        return df

    '''
        模板 - 股票基金代码处理方法：
        (1)
        去除多余字符，补足6位（防止港股等4位编码）；只保留6位纯数字。
        (2)股票代码格式有ASH123456、123456.SH、123456.ZH、123456.
        SHA、港股0700.HK等。
    '''

    def CodeStandardlized(self, df, colname):
        def process(x):
            compile = re.compile(r'[0-9]')
            tmp_list = compile.findall(x)
            result = ''.join(tmp_list)
            return result.rjust(6, '0')

        df[colname] = df[colname].apply(lambda x: process(x))
        df = df.rename(columns={colname: 'code'})
        return df

    '''
    将日期处理方法和股票基金代码处理相结合
    形成PipeLine处理方法
    与两者分别进行等价
    '''

    def PipeLine(self, df, date, code, ascending=True):
        new = self.TimeStandardlized(df, date, ascending=ascending)
        new = self.CodeStandardlized(new, code)
        new = new.rename(columns={date: 'date', code: 'code'})
        return new

    '''
    模板 - 拼接方法：
    对日期和代码标准化后进行拼接
    采取merge方法进行拼接
    选取不同的连接方式作为参数
    包括内连接、左连接和右连接
    拼接在date和code上
    '''

    def ConcatDf(self, df1, df2, col_list, how='inner'):
        col_list.extend(['date', 'code'])
        df1 = df1[col_list]
        if how == 'inner':
            result = df2.merge(df1, how=how, on=['date', 'code'])
        else:
            datelst = sorted(list(set(df1['date']).intersection(set(df2['date']))))
            result = pd.DataFrame(None)
            for date in datelst:
                df1_Date = df1[df1['date'] == date]
                df2_Date = df2[df2['date'] == date]
                temp = df1_Date.merge(df2_Date, how=how, left_on='code', right_on='code')
                result = pd.concat([result, temp], axis=0)
        return result


'''
模板 - 日期检测：
(1)每年是不是都是12个月？
①　把不是的年份输出来；
②　把这些年份中缺少的月份输出来；

(2)输出每月最后一个日期（自然日、工作日）
①　Df.groupby（‘日期列名’）.apply(lambda x:max(x))
②　单独输出一列作为结果；
③　可选参数是否将这一列插回到原始数据中；
'''


class DateTest(object):
    def IsFullMonth(self, df1):
        df = df1.copy()
        df['year'] = [date[:4] for date in df['date']]
        df['month'] = [date[5:7] for date in df['date']]
        df['day'] = [date[8:] for date in df['date']]
        df = df.set_index('year')
        df = df.set_index('month', append=True)
        null_year = []
        null_month = []
        full_month = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'}
        for year, month in df.index:
            dfyear = df.loc[year, :]
            if len(set(dfyear.index)) != 12 and year not in null_year:
                null_year.append(year)
                null_month.append(full_month.difference(set(dfyear.index)))
        return null_year, null_month

    def LastDayInMonth(self, df1, inplace=False):
        df = df1.copy()
        df['year'] = [date[:4] for date in df['date']]
        df['month'] = [date[5:7] for date in df['date']]
        df['day'] = [date[8:] for date in df['date']]
        df = df.set_index('year')
        df = df.set_index('month', append=True)
        LastDay = []
        for year, month in df.index:
            dfyear = df.loc[year, :]
            dfday = dfyear.loc[month, :]
            LastDay.append(dfday['day'][-1])
        if inplace == True:
            df1['LastDay'] = LastDay
            print(df1['LastDay'])
            return df1
        else:
            df['LastDay'] = LastDay
            print(df['LastDay'])
            return df1


'''
模板 - nan检测：
(1)输出全是nan的空行的数量、对应各个行索引结果；
(2)输出只有一个nan值的空行的数量，对应各个行索引结果；
(3)Nan值处理
(4)输出填充后结果（ffill、bfill、填0、其它填充方法）
(5)输出删除nan结果（整行为nan、任何一个nan）
'''


class NullTest(object):
    def AllNull(self, df):
        null_index = []
        null_count = 0
        for i in range(len(df.isnull().values)):
            if df.isnull().values[i].all() == True:
                null_count += 1
                null_index.append(i)
        return null_count, null_index

    def AnyNull(self, df):
        null_index = []
        null_count = 0
        for i in range(len(df.isnull().values)):
            if df.isnull().values[i].any() == True:
                null_count += 1
                null_index.append(i)
        return null_count, null_index

    def DropNull(self, df):
        df1 = df.copy()
        df1 = df1.dropna(axis=0)
        return df1

    def FillNull(self, df, method=0):
        df1 = df.copy()
        if method == 'ffill' or 'bfill':
            df1 = df1.fillna(method=method)
        else:
            df1 = df1.fillna(0)
        return df1


'''
模板 - 数据读入模板
（1）读取csv格式文件
参数包括：
①是否包含注释行header并去除注释行（默认为无）
②是否包含索引列（默认为无），一般可选择第0列为索引
③选择编码方式（默认为utf-8），部分csv文件可选ISO-8859-1
（2）读取excel格式文件
参数包括：
①选择工作表（默认为第0张表），可选
②是否包含注释行header并去除注释行（默认为无）
③是否包含索引列（默认为无），一般可选择第0列为索引
④选择编码方式（默认为utf-8），部分csv文件可选ISO-8859-1
（3）读取h5模型和pkl模型
（4）读取txt文件
默认按行读取
'''


class DataLoad(object):
    def Loadcsv(self, csv, header=None, index_col=None, encoding='utf-8'):
        df = pd.read_csv(csv, index_col=index_col, encoding=encoding)
        if header != None and type(header) == int:
            return df.iloc[header:, :]
        elif header != None and type(header) == str:
            return df[header:]
        else:
            return df

    def Loadexcel(self, xls, sheet_name=0, header=None, index_col=0, encoding='utf-8'):
        df = pd.read_excel(xls, sheet_name=sheet_name, index_col=index_col, encoding=encoding)
        if header != None and type(header) == int:
            return df.iloc[header:, :]
        elif header != None and type(header) == str:
            return df[header:]
        else:
            return df

    def Loadh5(self, h5):
        model = load_model(h5)
        return model

    def Loadpkl(self, pkl):
        model = Word2Vec.load(pkl)
        return model

    def Loadtxt(self, txt):
        text_list = []
        with Path(txt).open() as f:
            for line in f:
                text_list.append(line.strip())
        return text_list


'''
模板 - 数据导出模板
（1）导出csv文件
参数包括：
①分隔符，默认为无
②替换空值，默认为空值
③格式，数字保留几位小数，如float_format='%.2f'
④是否保留列名，默认保留列名
⑤是否保留行索引，默认保留索引
（2）导出excel文件
参数包括：
①选择工作表（默认为第0张表），可选
②分隔符，默认为无
③替换空值，默认为空值
④格式，数字保留几位小数，如float_format='%.2f'
⑤是否保留列名，默认保留列名
⑥是否保留行索引，默认保留索引
（3）导出h5和pkl模型
（4）导出txt文件
写入文件
'''


class DataSave(object):
    def Savecsv(self, csv, csv_path, sep=',', na_rep='',
                float_format=None, header=True, index=True):
        csv.to_csv(csv_path, sep=sep, na_rep=na_rep,
                   float_format=float_format, header=header, index=index)

    def Saveexcel(self, xls, xls_path, sheet_name=0, sep=None, na_rep=None,
                  float_format=None, header=True, index=True):
        xls.to_excel(xls_path, sheet_name=sheet_name, sep=sep, na_rep=na_rep,
                     float_format=float_format, header=header, index=index)

    def Saveh5(self, h5, model_path):
        h5.save(model_path)
        return

    def Savepkl(self, pkl, model_path):
        pkl.save(model_path)
        return

    def Savetxt(self, txt, txt_path):
        with open(txt_path, 'w') as f:
            f.write(txt)
        return


if __name__ == '__main__':
    # eodprices = pd.read_csv(r'wind-1\ashareeodprices.csv')[:100]     # , index_col=2
    # eodprices.to_csv('ashareeodprices_100,csv')
    l = DataLoad()
    eodprices = l.Loadcsv('ashareeodprices_100.csv', header=None, index_col=0)
    model1 = l.Loadh5('LSTM_model.h5')
    model2 = l.Loadpkl('Word2vec_model.pkl')
    model3 = l.Loadtxt('tsinghua.negative.txt')
    # indIndex = pd.read_csv(r'industry_zx_CN_STOCK_A\industry_zx_CN_STOCK_A.csv') [:100]   # , index_col=0
    # indIndex.to_csv('industry_zx_CN_STOCK_A_100.csv.csv')
    indIndex = l.Loadcsv('industry_zx_CN_STOCK_A_100.csv.csv', header=None, index_col=0, encoding='ISO-8859-1')
    eodprices_col_list = ["S_DQ_PRECLOSE", "S_DQ_OPEN"]
    z = NullTest()
    all_null_count, all_null_index = z.AllNull(eodprices)
    print('全为空值的行数量', all_null_count)
    print('全为空值的行索引', all_null_index)
    any_null_count, any_null_index = z.AnyNull(eodprices)
    print('存在任一空值的行数量', any_null_count)
    print('存在任一空值的行索引', any_null_index)
    drop_null_df = z.DropNull(eodprices)
    print(drop_null_df)
    fill_null_df = z.FillNull(eodprices, method='ffill')
    print(fill_null_df)
    eodprices = eodprices[:-1]
    x = PrepareData(eodprices_col_list, eodprices, indIndex)
    df1 = x.PipeLine(x.df[0], 'TRADE_DT', 'S_INFO_WINDCODE', ascending=True)
    df2 = x.PipeLine(x.df[1], 'date', 'instrument', ascending=True)
    concat_df = x.ConcatDf(df1, df2, x.df_collist, how='inner')
    y = DateTest()
    null_year, null_month = y.IsFullMonth(df1)
    for i in range(len(null_year)):
        print('不是12月份的年：', null_year[i])
        print('缺少的月份为：', null_month[i])
    LastDay = y.LastDayInMonth(df1, inplace=False)
