from __future__ import print_function
import datetime
# from yahoo_finance import Share
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# START =  "2011-01-01" # Data start date
startDay = '2015-07-14'
# endDay = '2016-08-19'
_ID = 2330 # By default, TSMC (2330)

# stock = Share(str(_ID)+'.TW')
today = datetime.date.today()
# stock_data = stock.get_historical(START, str(today))
stock_data=pd.read_excel(r'data\2330.xls',index_col=0)[:100]
# print("Historical data since", startDay,": ", len(stock_data))


K = []
D = []
util = []
for i in range(len(stock_data.index)):
        util.append(float(stock_data['Close'][stock_data.index[i]]))
        if i >= 8:
                assert len(util) == 9

                #----RSV----            
                if max(util) == min(util):
                        RSV = 0.0
                else:
                        RSV = (util[len(util)-1] - min(util))/(max(util)-min(util))
                #----RSV----

                #----K----
                if i == 8:
                        temp_K = 0.5*0.6667 + RSV*0.3333
                        K.append(temp_K)
                else:
                        temp_K = K[-1]*0.6667 + RSV*0.3333
                        K.append(temp_K)
                #----K----

                #----D----
                if i == 8:
                        D.append(0.5*0.6667 + temp_K*0.3333)
                else:
                        D.append(D[-1]*0.6667 + temp_K*0.3333)
                #----D----
                util.pop(0)
                assert len(util) == 8


def draw(arrlist,name):
    new = np.zeros((len(arrlist)-15), dtype=np.float)
    print(len(arrlist))
    for x in range(0,len(arrlist)-15):#save file
        for y in range(0,15):#save file
            print(arrlist[x+y])
            new[y]=arrlist[x+y]
            plt.plot(new,label='K',linewidth=5)
            plt.axis([0, 14, 0, 1])
            plt.savefig('/'+name+'/'+str(x)+'.png')
            plt.close()

def drawkd(k,d):
    newk = np.zeros((len(k)-15), dtype=np.float)
    newd = np.zeros((len(d)-15), dtype=np.float)
    for x in range(0,len(k)-15):#save file
        for y in range(0,15):#save file
            newk[y]=k[x+y]
            newd[y]=d[x+y]
            plt.plot(newk,label='K',linewidth=5,color=[0,0,1])
            plt.plot(newd,label='D',linewidth=5,color=[0,1,0])
            plt.axis([0, 14, 0, 1])
            plt.axis('off')
            plt.legend()
            plt.savefig(r'result\KD_{}.png'.format(str(x)))
            plt.close()
# #draw(K,"k")
# #draw(D,"d")
drawkd(K,D)
