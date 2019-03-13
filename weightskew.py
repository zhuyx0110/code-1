import pandas as pd
from math import log,exp,sqrt
from scipy import stats
import numpy as np
pd.set_option('display.max_columns', 20, 'display.max_rows', 20)

# #读取所有交易日日期
# import os
# filename=os.listdir(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\Mapping''')
# days=[filename[i][0:8] for i in range(np.shape(filename)[0])]
global tradeday
calendar=pd.read_csv('F:\calendar.csv')
day=calendar[calendar['a']==1].astype('str')
tradeday=day[day['date']>'20150208']
tradeday=tradeday.reset_index(drop=True)
global ETF_price
ETF_price=pd.read_csv(r'''\\win-g12\ResearchWorks\Interns\yuxiao.zhu\Data\50ETF\50ETF_DAILY.csv''' )
ETF_price['Date']=pd.to_datetime(ETF_price['Date'])

def deltatradeday(NOWADAY,MATURITY): #计算剩余交易日天数，一年近似250个交易日;数据格式问题20150901和2015-09-01
    m=''.join(list(filter(str.isalnum, MATURITY)))
    p1=tradeday[tradeday['date']==NOWADAY].index[0]
    p2=tradeday[tradeday['date']==m].index[0]
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
    # date1=pd.to_datetime(date)
    S =ETF_price[ETF_price['Date']==pd.to_datetime(date)]['close_not_adj'].values[0]
        # Mkt_price[Mkt_price['CODE'] == '510050.SH']['CLOSE'].values[0]
    data=pd.merge(mapping[['CODE','OPT_TYPE','MATURITY','STRIKE']],Mkt_price[['CLOSE','CODE']],on='CODE')
    #计算T
    data['T'] = data.apply(lambda x: deltatradeday( date, x['MATURITY']), axis=1)
    data = data.dropna()
    a=data.groupby('T').count()
    T1=a.index[np.where(a.index>0.02)[0][0]]  #当月T值
    T2=a.index[np.where(a.index>0.02)[0][1]]  #次月T值

    # 选择适合的样本计算FS
    data1=data[( data['T']==T1)]
    datap=data1[data1['OPT_TYPE'] == 'P']
    datap=datap.copy()
    datap.rename(columns={'CODE':'CODE1','CLOSE':'CLOSE1','OPT_TYPE':'OPR_TYPE1'}, inplace = True)
    data2=pd.merge(data1[data1['OPT_TYPE']=='C'],datap,on=['MATURITY','T','STRIKE'])
    data2['ATM']=abs(data2['STRIKE']-S)
    data2 = data2.dropna()
    data2=data2.sort_values(by=['ATM','STRIKE','OPT_TYPE'])
    data2=data2.reset_index(drop=True)
    sample=data2[data2.index<samplenum]
    #计算FS
    FS=[]
    for i in range(0,sample.shape[0]):
        T=sample['T'][i]
        K=sample['STRIKE'][i]
        try:
            Price_c=sample['CLOSE'].values[i]
            Price_p=sample['CLOSE1'].values[i]
        except:
            continue
        FS.append(Price_c+K*exp(-1*r*T)-Price_p)#平价公式隐含的期货价值
    Ave_FS=np.mean(FS)
    #计算adj_iv和delta
    T = sample['T'].values[0]
    data_T=data[data['T']==T]
    data_T=data_T.copy()
    data_T['adj_iv'] = data_T.apply(lambda x: iv_dichotomy(Ave_FS,x['STRIKE'],x['T'],r,x['CLOSE'],x['OPT_TYPE']), axis=1)
    data_T['delta'] = data_T.apply(lambda x: delta(Ave_FS, x['STRIKE'], x['T'], r, x['adj_iv'], x['OPT_TYPE']), axis=1) #调整过的iv计算delta
    data_T = data_T.sort_values(by=['delta']) #将同一行权价和到期日的c，p期权排列在一起
    data_T = data_T.sort_values(by=['STRIKE'])
    data_T=data_T.reset_index(drop=True)
    #线性拟合
    data_cp=data_T[(abs(data_T['delta'])<=0.5) & (abs(data_T['delta'])>=0.05)]
    z1 = np.polyfit(data_cp['delta'], data_cp['adj_iv'], 1)[0]
    # a=np.poly1d(z)
    # print(a)
    data_c=data_cp[(data_cp['OPT_TYPE']=='C')]
    if len(data_c)>1:
        z11 = np.polyfit(data_c['delta'],data_c['adj_iv'] , 1)[0]
    else:
        z11=np.nan
    data_p=data_cp[(data_cp['OPT_TYPE']=='P')]
    if len(data_p)>1:
        z12 = np.polyfit(data_p['delta'], data_p['adj_iv'], 1)[0]
    else:
        z12=np.nan

    if a.index[0]>0.08:
        z2=z1
        z21=z11
        z22=z12
    else:
        #次月合约
        data1 = data[(data['T'] == T2)]
        datap = data1[data1['OPT_TYPE'] == 'P']
        datap = datap.copy()
        datap.rename(columns={'CODE': 'CODE1', 'CLOSE': 'CLOSE1', 'OPT_TYPE': 'OPR_TYPE1'}, inplace=True)
        data2 = pd.merge(data1[data1['OPT_TYPE'] == 'C'], datap, on=['MATURITY', 'T', 'STRIKE'])
        data2['ATM'] = abs(data2['STRIKE'] - S)
        data2=data2.dropna()
        data2 = data2.sort_values(by=['ATM', 'STRIKE', 'OPT_TYPE'])
        data2 = data2.reset_index(drop=True)
        sample = data2[data2.index < samplenum]
        # 计算FS
        FS = []
        for i in range(0, sample.shape[0]):
            T = sample['T'][i]
            K = sample['STRIKE'][i]
            try:
                Price_c = sample['CLOSE'].values[i]
                Price_p = sample['CLOSE1'].values[i]
            except:
                continue
            FS.append(Price_c + K * exp(-1 * r * T) - Price_p)  # 平价公式隐含的期货价值
        Ave_FS = np.mean(FS)
        # 计算adj_iv和delta
        T = sample['T'].values[0]
        data_T = data[data['T'] == T]
        data_T = data_T.copy()
        data_T['adj_iv'] = data_T.apply(lambda x: iv_dichotomy(Ave_FS, x['STRIKE'], x['T'], r, x['CLOSE'], x['OPT_TYPE']),
                                        axis=1)
        data_T['delta'] = data_T.apply(lambda x: delta(Ave_FS, x['STRIKE'], x['T'], r, x['adj_iv'], x['OPT_TYPE']),
                                       axis=1)  # 调整过的iv计算delta
        # data_T = data_T.sort_values(by=['delta'])  # 将同一行权价和到期日的c，p期权排列在一起
        data_T = data_T.sort_values(by=['STRIKE'])
        data_T = data_T.reset_index(drop=True)
        # 线性拟合
        data_cp = data_T[(abs(data_T['delta'])<=0.5) & (abs(data_T['delta'])>=0.05)]
        z2 = np.polyfit(data_cp['delta'], data_cp['adj_iv'], 1)[0]
        # a=np.poly1d(z)
        # print(a)
        data_c = data_cp[(data_cp['OPT_TYPE'] == 'C')]
        if len(data_c) > 1:
            z21 = np.polyfit(data_c['delta'], data_c['adj_iv'], 1)[0]
        else:
            z21 =np.nan
        data_p = data_cp[(data_cp['OPT_TYPE'] == 'P')]
        if len(data_p) > 1:
            z22 = np.polyfit(data_p['delta'], data_p['adj_iv'], 1)[0]
        else:
            z22 = np.nan
    w1=T1*(T2-20/250)/(T2-T1)
    w2=20/250-w1
    return pd.DataFrame([[date,(w1*z1+w2*z2)*250/20,(w1*z11+w2*z21)*250/20,(w1*z12+w2*z22)*250/20]],columns=['date','tweight_skew','tweight_skew_c','tweight_skew_p'])
samplenum=4
r=0.03
#计算skew序列
skew = pd.DataFrame([], columns=['date','tweight_skew','tweight_skew_c','tweight_skew_p'])

for date in tradeday['date']:
    # if int(date)>20150229:
    #      break
    try:
        data = Realiv(date, samplenum, r)
    except:
        print(date)
        continue
    skew=skew.append(data)
skew=skew.reset_index(drop=True)


skew.to_csv('f:/weightskew1.csv')