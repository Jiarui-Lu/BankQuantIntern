import pandas as pd
# 画图
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')

# 忽略报错
import warnings
warnings.filterwarnings("ignore")

close_df=pd.read_excel(r'data\HS300.xlsx',index_col=0)
close_df['pre_close']=close_df['close'].shift(1)
close_df['pct']=close_df['close']/close_df['pre_close']-1
# print(close_df)
# 画图
plt.figure(figsize=(22, 8))
# 计算n日波动率  未年化处理
for periods in [60, 120, 200, 250]:

    col = 'std_'+str(periods)
    close_df[col] = close_df['pct'].rolling(periods).std()
    plt.plot(close_df[col], label=col)

plt.legend(loc='best')
plt.xlabel('时间')
plt.ylabel('波动率')
plt.title('沪深300不同参数下的历史波动率对比（日波动率，未年化）')
plt.savefig(r'result\volatility comparison.jpg')
plt.show()
#
# 图表3 沪深300及其250日波动率

y1 = close_df['close']  # 获取收盘数据
y2 = close_df['std_250']  # 获取250日波动率

fig = plt.figure(figsize=(18, 8))  # 图表大小设置

ax1 = fig.add_subplot(111)

ax1.plot(y1, label='close')
ax1.set_ylabel('收盘价')
plt.legend(loc='upper left')


ax2 = ax1.twinx()  # 设置双Y轴 关键function
ax2.plot(y2, '#87CEFA')
ax2.set_ylabel('波动率')
plt.legend(loc='best')

ax1.set_title("沪深300及其250日波动率")
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # x轴显示Y-m
plt.xlabel('时间')
plt.savefig(r'result\close versus volatility.jpg')
plt.show()

#  图表4:上证综指与波动率的滚动相关系数
#
y1 = close_df['close'] # 获取收盘数据
#
# 获取250日波动率与收盘价滚动1年相关系数corr
y2 = close_df['close'].rolling(250).corr(close_df['std_250'])

fig,ax = plt.subplots(1,1,figsize=(18,8)) # 图表大小设置


ax.plot(y1,label='close')
ax.set_ylabel('收盘价')
plt.legend(loc='upper left')

ax1 = ax.twinx()  # 设置双Y轴 关键function
ax1.plot(y2,'#87CEFA',label='corr')
ax1.set_ylabel('相关系数')
plt.legend(loc='upper right')

ax1.set_title("沪深300及其250日波动率")
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m')) # x轴显示Y-m
plt.xlabel('时间')
plt.savefig(r'result\correlation.jpg')
plt.show()

# 图5，6 上证综指日换手率(这里调用了tushare数据，需要tushare 400积分才能调用)

# tushare=>float_share流通股本 （万股），free_share自由流通股本 （万）
# tushare 指数数据从2004年开始提供....有单次查询限制


index_daily_df=pd.read_excel(r'data\turnover.xlsx',index_col=0)
# print(index_daily_df)
# 画图
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('沪深300日换手率')

x = index_daily_df.index
ax[0].bar(x, index_daily_df['turnover_rate'])
ax[0].set_title('基于总流通股本的换手率')

ax[1].bar(x, index_daily_df['turnover_rate_f'])
ax[1].set_title('基于自由流通股本的换手率')

plt.savefig(r'result\turnover.jpg')
plt.show()


# 图表7  沪深300不同参数下的日均换手率

# 画图
plt.figure(figsize=(18, 8))

for periods in [60, 120, 200, 250]:

    col = 'turnover_rate_'+str(periods)
    index_daily_df[col] = index_daily_df['turnover_rate_f'].rolling(
        periods).mean()
    plt.plot(index_daily_df[col], label=col)

plt.legend(loc='best')
plt.xlabel('时间')
plt.ylabel('换手率')
plt.title('沪深300不同参数下的日均换手率')
plt.savefig(r'result\daily turnover.jpg')
plt.show()

# 图表8 沪深300与 250 日换手率

close_df['turnover_rate_250'] = index_daily_df['turnover_rate_250']
y1 = close_df['close']  # 获取收盘数据
y2 = close_df['turnover_rate_250']  # 获取250日波动率

fig = plt.figure(figsize=(18, 8))  # 图表大小设置

ax1 = fig.add_subplot(111)

ax1.plot(y1, label='close')
ax1.set_ylabel('收盘价')
plt.legend(loc='upper left')


ax2 = ax1.twinx()  # 设置双Y轴 关键function
ax2.plot(y2, '#87CEFA')
ax2.set_ylabel('换手率')
plt.legend(loc='best')

ax1.set_title("沪深300与 250 日换手率")
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # x轴显示Y-m
plt.xlabel('时间')
plt.savefig(r'result\turnover comparision.jpg')
plt.show()

print('这里换手率单位为百分位,我们将波动率*100是单位保持一直，为了使两个参数都在次轴反应对波动率*2\n')

print('波动率:\n', (close_df['std_250']*100*2).describe(), '\n')

print('换手率\n', close_df['turnover_rate_250'].describe())

#  图表9 波动率与换手率对沪深300走势状态的划分

y1 = close_df['close']  # 获取收盘数据
y2 = close_df['turnover_rate_250']  # 获取250日波动率
y3 = close_df['std_250']*100*2  # 换手率为%，所以这里要*100

fig = plt.figure(figsize=(18, 8))  # 图表大小设置

ax1 = fig.add_subplot(111)

ax1.plot(y1, linestyle='-.', label='close')
ax1.set_ylabel('收盘价')
plt.legend(loc='upper left')


ax2 = ax1.twinx()  # 设置双Y轴 关键function
ax2.plot(y2, '#00FA9A')
ax2.plot(y3, '#00CED1')
ax2.set_ylabel('换手率\波动率')
plt.legend(loc='best')

ax1.set_title("波动率与换手率对沪深300走势状态的划分")
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # x轴显示Y-m
plt.xlabel('时间')
plt.savefig(r'result\volatility versus turnover.jpg')
plt.show()

# 图表11

# 计算牛熊指标

close_df['kernel_index'] = close_df['std_200'] / \
    index_daily_df['turnover_rate_200']

y1 = close_df['close']  # 获取收盘数据
y2 = close_df['kernel_index']  # 获取250日波动率

fig = plt.figure(figsize=(18, 8))  # 图表大小设置

ax1 = fig.add_subplot(111)

ax1.plot(y1, label='close')
ax1.set_ylabel('收盘价')
plt.legend(loc='upper left')


ax2 = ax1.twinx()  # 设置双Y轴 关键function
ax2.plot(y2, '#87CEFA')
ax2.set_ylabel('牛熊指标')
plt.legend(loc='best')

ax1.set_title("沪深300收盘价与其对应的牛熊指标")
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # x轴显示Y-m
plt.xlabel('时间')
plt.savefig(r'result\close versus bullishbearindex.jpg')
plt.show()

# 牛熊指标与收盘数据的相关系数
corr = close_df[['kernel_index']].corrwith(close_df['close']).values[0]
df=pd.DataFrame([corr],columns=['corr'])
df.to_excel(r'result\corr.xls')
# print('牛熊指标与收盘数据的相关系数:%.2f' % corr)
# 牛熊指标与收盘数据的相关系数:-0.42

