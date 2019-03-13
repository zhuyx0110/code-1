import pandas as pd
import datetime
from statsmodels.tsa.api import VAR, DynamicVAR
import statsmodels as sm
import numpy as np


def MaxDrawdown(return_list):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return j,i,(return_list[j] - return_list[i]) / (return_list[j])


# return_list = [12, 12, 21, 15, 27, 16, 21, 22, 25, 20, 16, 17]
# MaxDrawdown(return_list)#输入为list、arrat均可
ETF_price=pd.read_csv(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\50ETF_DAILY.csv''' )
ETF_price['Date']=pd.to_datetime(ETF_price['Date'])
ETF_price['earning']=(ETF_price['close_not_adj'].shift(-1)-ETF_price['open_not_adj'].shift(-1))/ETF_price['open_not_adj'].shift(-1)

data=pd.read_csv('f:/backtestdata.csv')
data['skew1']=data['skew']+data['linear_skew']+data['tweight_skew']+data['cboe_skew']+data['skew_c'] \
               +data['linear_skew_c']+data['tweight_skew_c']+data['cboe_skew_c']- \
               data['skew_p']-data['linear_skew_p']-data['tweight_skew_p']-data['cboe_skew_p']
data['skew2']=data['skew_c']+data['linear_skew_c']+data['tweight_skew_c']+data['cboe_skew_c']+ \
               data['skew_p']+data['linear_skew_p']+data['tweight_skew_p']+data['cboe_skew_p']
data['skew3']=1/3*(data['linear_skew']+data['tweight_skew']+data['cboe_skew'])-data['skew']+ \
               1/3*(data['linear_skew_c']+data['tweight_skew_c']+data['cboe_skew_c'])-data['skew_c']- \
               1/3*(data['linear_skew_p']+data['tweight_skew_p']+data['cboe_skew_p'])+data['skew_p']
data['iv']=data['vol']+data['real_vol']
data['diff_iv']=data['vol']-data['real_vol']
data['pcr1']=data['pcr_trade']-data['pcr_hold']
data['pcr2']=data['pcr_trade']+data['pcr_hold']
data['date']=data['date'].astype('str')
data['Date']=pd.to_datetime(data['date'])

data2=pd.merge(data,ETF_price[['Date','earning']])

data2=data2[['date','earning','skew1','skew2','skew3','iv','diff_iv','pcr1','pcr2','pcr_trade']]


##反转指标skew1
backtest=data2[['date','earning','skew1']]
backtest=backtest.copy()
backtest['l2']=backtest['skew1'].rolling(60).quantile(0.4)
backtest['l8']=backtest['skew1'].rolling(60).quantile(0.8)
backtest['rank']=backtest.apply(lambda x: 2 if x['skew1']>x['l8'] else 0 if x['skew1']<x['l2'] else 1,axis=1)
backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else -x['earning'] if x['rank']==2 else 0),axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
# backtest['cum1']=(backtest['invest1']+1).cumprod()
# backtest['cum2']=(backtest['invest2']+1).cumprod()
# backtest['cum3']=(backtest['earning']+1).cumprod()

#
backtest=data2[['date','earning','skew2']]
backtest=backtest.copy()
backtest['l2']=backtest['skew2'].rolling(60).quantile(0.4)
backtest['l8']=backtest['skew2'].rolling(60).quantile(0.6)
backtest['rank']=backtest.apply(lambda x: 2 if x['skew2']>x['l8'] else 0 if x['skew2']<x['l2'] else 1,axis=1)
backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else -x['earning'] if x['rank']==2 else 0),axis=1)
backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
backtest['invest3']=backtest.apply(lambda x :( x['earning'] if x['rank']==1 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest['cum1']=(backtest['invest1']+1).cumprod()
backtest['cum2']=(backtest['invest2']+1).cumprod()
backtest['cum3']=(backtest['earning']+1).cumprod()
backtest['cum4']=(backtest['invest3']+1).cumprod()

#趋势指标skew3
backtest=data2[['date','earning','skew3']]
backtest=backtest.copy()
backtest['l2']=backtest['skew3'].rolling(60).quantile(0.4)
backtest['l8']=backtest['skew3'].rolling(60).quantile(0.8)
backtest['rank']=backtest.apply(lambda x: 2 if x['skew3']>x['l8'] else 0 if x['skew3']<x['l2'] else 1,axis=1)
backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( -x['earning'] if x['rank']==0 else x['earning'] if x['rank']==2 else 0),axis=1)
# backtest['invest1']=backtest.apply(lambda x :( -x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
# backtest['cum1']=(backtest['invest1']+1).cumprod()
# backtest['cum2']=(backtest['invest2']+1).cumprod()
# backtest['cum3']=(backtest['earning']+1).cumprod()

#反转指标iv
backtest=data2[['date','earning','iv']]
backtest=backtest.copy()
backtest['l2']=backtest['iv'].rolling(60).quantile(0.4)
backtest['l8']=backtest['iv'].rolling(60).quantile(0.8)
backtest['rank']=backtest.apply(lambda x: 2 if x['iv']>x['l8'] else 0 if x['iv']<x['l2'] else 1,axis=1)

backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else -x['earning'] if x['rank']==2 else 0),axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
# backtest['cum1']=(backtest['invest1']+1).cumprod()
# backtest['cum2']=(backtest['invest2']+1).cumprod()
# backtest['cum3']=(backtest['earning']+1).cumprod()

#反转指标diff_iv 有效性优于单纯波动率
backtest=data2[['date','earning','diff_iv']]
backtest=backtest.copy()
backtest['l2']=backtest['diff_iv'].rolling(60).quantile(0.4)
backtest['l8']=backtest['diff_iv'].rolling(60).quantile(0.8)
backtest['rank']=backtest.apply(lambda x: 2 if x['diff_iv']>x['l8'] else 0 if x['diff_iv']<x['l2'] else 1,axis=1)
backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else -x['earning'] if x['rank']==2 else 0),axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
# backtest['cum1']=(backtest['invest1']+1).cumprod()
# backtest['cum2']=(backtest['invest2']+1).cumprod()
# backtest['cum3']=(backtest['earning']+1).cumprod()


# 趋势指标 pcr之差
backtest=data2[['date','earning','pcr1']]
backtest=backtest.copy()
backtest['l2']=backtest['pcr1'].rolling(60).quantile(0.4)
backtest['l8']=backtest['pcr1'].rolling(60).quantile(0.8)
backtest['rank']=backtest.apply(lambda x: 2 if x['pcr1']>x['l8'] else 0 if x['pcr1']<x['l2'] else 1,axis=1)
backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else -x['earning'] if x['rank']==2 else 0),axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
# backtest['cum1']=(backtest['invest1']+1).cumprod()
# backtest['cum2']=(backtest['invest2']+1).cumprod()
# backtest['cum3']=(backtest['earning']+1).cumprod()


# 趋势指标 pcr之和
backtest=data2[['date','earning','pcr2']]
backtest=backtest.copy()
backtest['l2']=backtest['pcr2'].rolling(60).quantile(0.4)
backtest['l8']=backtest['pcr2'].rolling(60).quantile(0.8)
backtest['rank']=backtest.apply(lambda x: 2 if x['pcr2']>x['l8'] else 0 if x['pcr2']<x['l2'] else 1,axis=1)
backtest=backtest.dropna()
backtest=backtest.drop(columns=['l2','l8'])
backtest['earning'].groupby(backtest['rank']).describe()
backtest['invest']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else-x['earning'] if -x['rank']==2 else 0),axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
# backtest['cum1']=(backtest['invest1']+1).cumprod()
# backtest['cum2']=(backtest['invest2']+1).cumprod()
# backtest['cum3']=(backtest['earning']+1).cumprod()



## 组合打分策略
backtest=data2[['date','earning','skew1','skew3','iv','diff_iv','pcr_trade']]
backtest=backtest.copy()
m,n=backtest.shape
for i in range(2,n):
    column=backtest.columns[i]
    backtest['l4']=backtest[column].rolling(60).quantile(0.4)
    backtest['l8']=backtest[column].rolling(60).quantile(0.8)
    backtest['rank'+str(i-1)] = backtest.apply(lambda x: 2 if x[column] > x['l8'] else 0 if x[column] < x['l4'] else 1,
                                          axis=1)
backtest['rank2']=abs(backtest['rank2']-2)
backtest=backtest.dropna()
backtest['rank']=backtest['rank2']+backtest['rank5']+backtest['rank1']+backtest['rank3']+backtest['rank4']
backtest['averank']=backtest['rank']/5
backtest=backtest.dropna()
backtest['invest']=backtest.apply(lambda x : x['earning'] if (x['averank']<=0.6) else -x['earning'] if (x['averank']>=1.2)  else 0,axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)

MaxDrawdown(backtest['cum'].values)
backtest['hold']=backtest.apply(lambda x:1 if (x['averank']<=0.6) else -1 if (x['averank']>=1.2)  else 0,axis=1)
backtest['diff']=backtest['hold'].diff(1)
a=len(backtest[backtest['diff']!=0])


#分别计算趋势和翻转指标
backtest=data[['date','earning','skew1','skew3','iv','diff_iv','pcr_trade']]
backtest=backtest.copy()
m,n=backtest.shape
for i in range(2,n):
    column=backtest.columns[i]
    backtest['l4']=backtest[column].rolling(60).quantile(0.4)
    backtest['l8']=backtest[column].rolling(60).quantile(0.8)
    backtest['rank'+str(i-1)] = backtest.apply(lambda x: 2 if x[column] > x['l8'] else 0 if x[column] < x['l4'] else 1,
                                          axis=1)

backtest['rank2']=abs(backtest['rank2']-2)
backtest=backtest.dropna()
# backtest['rank']=backtest['rank2']+backtest['rank5']+backtest['rank1']+backtest['rank3']+backtest['rank4']
# backtest['averank']=backtest['rank']/5
# backtest['rank']=backtest['rank2']+backtest['rank5']+backtest['rank6']
# backtest['reverserank']=backtest['rank1']+backtest['rank3']+backtest['rank4']
backtest['freq2']=backtest.apply(lambda x: np.size(np.where(x==2)),axis=1)
backtest['freq0']=backtest.apply(lambda x: np.size(np.where(x==0)),axis=1)
backtest=backtest.dropna()
backtest['invest']=backtest.apply(lambda x : x['earning'] if (x['freq0']>=7) else -x['earning'] if (x['freq2']>=3)  else 0,axis=1)
# backtest['invest1']=backtest.apply(lambda x :( x['earning'] if x['rank']==0 else 0),axis=1)
# backtest['invest2']=backtest.apply(lambda x :( -x['earning'] if x['rank']==2 else 0),axis=1)
win=backtest[backtest['invest']!=0]
wr=len(win[win['invest']>0])/len(win)
backtest['cum']=(backtest['invest']+1).cumprod()
backtest=backtest.reset_index(drop=True)
MaxDrawdown(backtest['cum'].values)
#
# a=backtest[(backtest['reverserank']<=2) &(backtest['rank']<=2)]
# a=backtest[(backtest['reverserank']>=4) &(backtest['rank']>=4)]