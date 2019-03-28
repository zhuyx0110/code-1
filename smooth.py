import pandas as pd
import datetime
from statsmodels.tsa.api import VAR, DynamicVAR
import statsmodels as sm
import numpy as np
from math import log,exp,sqrt
pd.set_option('display.max_columns', 20, 'display.max_rows', 20)

def SMA(data,n):
    out=data.rolling(n).mean()
    return out.values

def WMA(data,n):#权重为n,n-1,n-2.n-3....
    a=pd.DataFrame()
    for i in range(n):
        a[str(i)]=data.shift(i)
    out=a.apply(lambda x:np.average(x,weights=[i for i in range(n,0,-1)]),axis=1)
    return out.values

def EWMA(data,n):#指数平滑,权重为1,1-alpha,(1-alpha)^2...或者alphaXt+（1-alpha）EWMAt-1
    out= data.ewm(span=n,ignore_na=True,adjust=True).mean()#数据有限的情况下adjust=True,false实在数据很多亲情况下的简化；可以指定span=n，会自动计算alpha，使得n区间之外的权重影响极小。可以以指定权重alpha
    return out.values

def KAMA(data,fast=2,slow=30):
    # 一般取10天来计算ER
    day=10
    a=pd.DataFrame()
    a['diff1']=abs(data.diff(1))
    a['vol']=a['diff1'].rolling(day).sum()
    a['change'] = abs(data.diff(day))
    a['ER']=a['change']/a['vol']
    a['SC']=(a['ER']*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2#平滑系数
    m=data.shape[0]
    out=np.zeros([1,m])
    out[0,0:day-1]=np.nan
    out[0,day-1]=data.values[day-1]
    for i in range(day,m):
        out[0,i]=(1-a['SC'].values[i])*out[0,i-1]+a['SC'].values[i]*data.values[i]
    return out[0]

# return_list = [12, 12, 21, 15, 27, 16, 21, 22, 25, 20, 16, 17]
# MaxDrawdown(return_list)#输入为list、arrat均可
ETF_price = pd.read_csv(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\50ETF_DAILY.csv''')
ETF_price['Date'] = pd.to_datetime(ETF_price['Date'])
ETF_price['earning'] = (ETF_price['close_not_adj'].shift(-1) - ETF_price['open_not_adj'].shift(-1)) / ETF_price['open_not_adj'].shift(-1)
data=ETF_price['close_not_adj']
x1=SMA(data,20)
x2=WMA(data,20)
x3=EWMA(data,20)
x4=KAMA(data)
out=np.vstack([x2,x3,x4]).T
MAdata=pd.DataFrame(out,columns=[['WMA','EWMA','KAMA']])
MAdata['Date']=ETF_price['Date']
