# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *


"""
示例策略仅供参考，不建议直接实盘使用。

菲阿里四价策略是一种简单趋势型日内交易策略。昨天最高点、昨天最低点、昨日收盘价、今天开盘价,可并称为菲阿里四价。
没有持仓下，当现价突破上轨时做多，当现价跌穿下轨时做空；以开盘价作为止损价，尾盘平仓，其中
上轨=昨日最高点；
下轨=昨日最低点；
止损=今日开盘价。
注：受目前回测机制限制，期货主力合约只能回测最近三年的数据，连续合约不受影响
"""


def init(context):
    # 设置标的
    context.symbol = 'DCE.JM'
    # 订阅一分钟线
    subscribe(symbols = context.symbol,frequency = '60s',count = 1)
    # 记录开仓次数，当前设置夜盘和日盘各最多一次
    context.count = 0
    # 定时任务：夜盘21点开始，日盘9点开始
    schedule(schedule_func=algo, date_rule='1d', time_rule='21:00:00')
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:00:00')


def algo(context):
    if context.now.hour>=20:
        # 当天夜盘和次日日盘属于同一天数据，为此当天夜盘的开盘价调用第二天的开盘价
        next_date = get_next_trading_date(exchange='DCE', date=context.now)
        # 获取历史的n条信息
        context.history_data = history_n(symbol=context.symbol, frequency='1d', end_time=next_date,
                                        fields='symbol,open,high,low,eob', count=2, adjust_end_time=context.now, df=True)
    else:
        # 获取历史的n条信息
        context.history_data = history_n(symbol=context.symbol, frequency='1d', end_time=context.now,
                                        fields='symbol,open,high,low,eob', count=2, adjust_end_time=context.now, df=True)


def on_bar(context,bars):
    # 现有持仓情况
    position_long = context.account().position(symbol=context.symbol, side=PositionSide_Long)# 多头仓位
    position_short = context.account().position(symbol=context.symbol, side=PositionSide_Short)# 空头仓位
    # 尾盘平仓
    if context.now.hour == 14 and context.now.minute >= 59 or context.now.hour == 15:
        # 有持仓时才触发平仓操作
        if position_long or position_short:
            order_close_all()
            print('{}:尾盘平仓'.format(context.now))
        context.count = 0

    # 非尾盘交易时间
    else:
        ## 数据获取
        bar = bars[0]
        data = context.history_data
        # 如果是回测模式
        if context.mode == 2:
            # 开盘价直接在data最后一个数据里取到,前一交易日的最高和最低价为history_data里面的倒数第二条中取到
            open = data[ 'open'].iloc[-1]
            high = data['high'].iloc[-2]
            low = data['low'].iloc[-2]
        # 如果是实时模式
        else:
            # 开盘价通过current取到,实时模式不会返回当天的数据，所以history_data里面的最后一条数据是前一交易日的数据
            open = current(context.symbol)[0]['open']
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]

        ## 交易逻辑部分
        if position_long:  
            if bar.close < open:# 平多仓：最新价小于开盘价时止损。
                order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
                            order_type=OrderType_Market, position_effect=PositionEffect_Close)
        elif position_short:
            if bar.close > open:# 平空仓：最新价大于开盘价时止损。
                order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
                            order_type=OrderType_Market, position_effect=PositionEffect_Close)
        else:  # 没有持仓
            if bar.close > high and not context.count:  # 开多仓：最新价大于了前一天的最高价
                order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
                            order_type=OrderType_Market, position_effect=PositionEffect_Open)
                context.count = 1
            elif bar.close < low and not context.count:  # 开空仓：最新价小于了前一天的最低价
                order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
                            order_type=OrderType_Market, position_effect=PositionEffect_Open)
                context.count = 1


def on_order_status(context, order):
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1:
            if side == 1:
                side_effect = '开多仓'
            elif side == 2:
                side_effect = '开空仓'
        else:
            if side == 1:
                side_effect = '平空仓'
            elif side == 2:
                side_effect = '平多仓'
        order_type_word = '限价' if order_type==1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now,symbol,order_type_word,side_effect,price,volume))
       

def on_backtest_finished(context, indicator):
    print('*'*50)
    print('回测已完成，请通过右上角“回测历史”功能查询详情。')
    

if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2021-01-01 20:50:00',
        backtest_end_time='2021-02-28 15:30:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)