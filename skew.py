import pandas as pd
from math import log,exp,sqrt
from scipy import stats
import numpy as np
pd.set_option('display.max_columns', 20, 'display.max_rows', 20)

#读取所有交易日日期
import os
filename=os.listdir(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\Mapping''')
days=[filename[i][0:8] for i in range(np.shape(filename)[0])]
global tradeday
tradeday=np.array(days)
# global ETF_price
# ETF_price=pd.read_csv(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\50ETF_DAILY.csv''' )

def deltatradeday(NOWADAY,MATURITY): #计算剩余交易日天数，一年近似250个交易日;数据格式问题20150901和2015-09-01
    m=''.join(list(filter(str.isalnum, MATURITY)))
    p1=np.argwhere(tradeday==NOWADAY)[0][0]
    p2=np.argwhere(tradeday==m)[0][0]
    T=p2-p1
    return T/250

#计算期权delta
def delta(s,k,t,r,sigma,type):
    d1 = (log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    if type=='C':
        d=stats.norm.cdf(d1)
    else:
        d=stats.norm.cdf(d1)-1
    return d


def bsm_value(s,k,t,r,sigma,type):  #bsm计算期权价值
    d1=(log(s/k)+(r+0.5*sigma**2)*t)/(sigma*sqrt(t))
    d2=(log(s/k)+(r-0.5*sigma**2)*t)/(sigma*sqrt(t))
    if type=='C':
        Optvalue=(s*stats.norm.cdf(d1))-k*exp(-r*t)*stats.norm.cdf(d2)
    elif type=='P':
        Optvalue=-s*stats.norm.cdf(-d1)+k*exp(-r*t)*stats.norm.cdf(-d2)
    else:
        print('wrong type')
    return Optvalue

def iv_dichotomy(s,k,t,r,c,type,iv_floor=0,iv_top=5):#二分法计算隐含波动率，上下界默认为0,5
    error=1
    iv=0.5*(iv_top+iv_floor)
    while abs(error)>1e-8 and iv>1e-10:
        c_est=bsm_value(s,k,t,r,iv,type)
        error=c_est-c
        if error>0:
            iv_top=iv
            iv=0.5*(iv_top+iv_floor)
        else:
            iv_floor=iv
            iv = 0.5 * (iv_top + iv_floor)
    return iv


def Realiv(date,samplenum,r): #计算给定日期的真实隐含波动率，行权价的采样区间为[S-interval,S+interval]

    mapping = pd.read_csv(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\Mapping\%s.csv''' % (date))
    Mkt_price=pd.read_csv(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\Daily\%s.csv''' % (date))
    S = Mkt_price[Mkt_price['CODE'] == '510050.SH']['CLOSE'].values[0]
    data=pd.merge(mapping[['CODE','OPT_TYPE','MATURITY','STRIKE']],Mkt_price[['CLOSE','CODE']],on='CODE')
    #计算T
    data['T'] = data.apply(lambda x: deltatradeday( date, x['MATURITY']), axis=1)

    #选择适合的样本计算FS
    if S>3 :
        interval=samplenum*0.5
    else:
        interval = samplenum  * 0.025
    sample=data[(abs(data['STRIKE']-S)<=interval) &( data['T']>0.015) & (data['T']<0.12)] #选择合适的期权样本
    sample = sample.sort_values(by=['STRIKE', 'MATURITY']) #将同一行权价和到期日的c，p期权排列在一起
    sample=sample.reset_index(drop=True)

    #计算FS
    FS=[]
    for i in range(0,sample.shape[0]-1,2):
        T=sample['T'][i]
        K=sample['STRIKE'][i]
        Price_c=sample['CLOSE'].values[i]
        Price_p=sample['CLOSE'].values[i+1]
        if sample['STRIKE'][i]!= sample['STRIKE'][i+1]:
            print(date,i)
        FS.append(Price_c+K*exp(-1*r*T)-Price_p)#平价公式隐含的期货价值
    Ave_FS=np.mean(FS)
    #计算adj_iv和delta
    T = sample['T'].values[0]
    data_T=data[data['T']==T]
    data_T=data_T.copy()
    data_T['adj_iv'] = data_T.apply(lambda x: iv_dichotomy(Ave_FS,x['STRIKE'],x['T'],r,x['CLOSE'],x['OPT_TYPE']), axis=1)
    data_T['delta'] = data_T.apply(lambda x: delta(Ave_FS, x['STRIKE'], x['T'], r, x['adj_iv'], x['OPT_TYPE']), axis=1) #调整过的iv计算delta
    data_T = data_T.sort_values(by=['OPT_TYPE','delta']) #将同一行权价和到期日的c，p期权排列在一起
    data_T=data_T.reset_index(drop=True)
    #线性插值
    delta_c = [0.25, 0.5]
    delta_p = [-0.25, -0.5]
    data_c=data_T[data_T['OPT_TYPE']=='C']
    iv_c=np.interp(delta_c, data_c['delta'], data_c['adj_iv'])
    data_p=data_T[data_T['OPT_TYPE']=='P']
    iv_p=np.interp(delta_p, data_p['delta'], data_p['adj_iv'])
    skew_c=iv_c[0]/iv_c[1]-1
    skew_p=iv_p[0]/iv_p[1]-1
    return pd.DataFrame([[date,skew_c,skew_p]],columns=['date','skew_c','skew_p'])


samplenum=4
r=0.03
#计算skew序列
skew = pd.DataFrame([], columns=['date','skew_c', 'skew_p'])

for date in tradeday:
    if int(date)>20160530:
        break
    try:
        data = Realiv(date, samplenum, r)
    except:
        continue
    print(date)
    skew=skew.append(data)
skew=skew.reset_index(drop=True)
skew[skew['skew_c']>0.1].shape[0]/skew.shape[0],skew[skew['skew_c']<0].shape[0]/skew.shape[0]
skew[skew['skew_p']>0.1].shape[0]/skew.shape[0],skew[skew['skew_p']<0].shape[0]/skew.shape[0]
skew.to_csv('f://skew.csv')
