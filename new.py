import pandas as pd
from math import log,exp,sqrt
from scipy import stats
import numpy as np
import scipy.optimize as sco
import statsmodels.api as sm #统计运算

import scipy.stats as scs #科学计算
pd.set_option('display.max_columns', 20, 'display.max_rows', 20)

def weightdata(data,weight):
    m,n=data.shape
    out=np.dot(data.values,weight)
    out1=np.reshape(out,(m,1))
    return out1


def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(train.mean()*weights)*252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(train.cov()*252,weights)))
    return np.array([port_returns, port_variance, port_returns/port_variance])

#最优化投资组合的推导是一个约束最优化问题

#最小化夏普指数的负值
def min_sharpe(weights):
    return -statistics(weights)[2]
#最小化风险
def min_variance(weights):
    return statistics(weights)[1]

def Markowitz(train):
    m1,n1=train.shape
    #约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    #我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0,1) for x in range(n1))
    #优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    #最大化sharp值
    opts = sco.minimize(min_sharpe, n1*[1./n1,], method = 'SLSQP', bounds = bnds, constraints = cons)
    #最小化风险
    optv = sco.minimize(min_variance, n1 * [1./ n1, ], method='SLSQP', bounds=bnds, constraints=cons)
    return opts['x'].round(3),optv['x'].round(3)

def min_risk(weights):
    weights = np.array(weights)
    var = train.var(0)*weights*252
    return np.std(var)


def riskequal(train):
    m1,n1=train.shape
    #约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    #我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0,1) for x in range(n1))
    #优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    opts = sco.minimize(min_risk, n1*[1./n1,], method = 'SLSQP', bounds = bnds, constraints = cons)
    return opts['x'].round(3)

from sklearn.decomposition import PCA


def min_pcarisk(weights):
    weights = np.array(weights)
    data=train*weights
    a = PCA(n_components=num)
    a.fit(data)
    return np.std(a.explained_variance_ratio_)


def pcaequal(train):
    m1,n1=train.shape
    #约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    #我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0,1) for x in range(n1))
    #优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    #最大化sharp值
    opts = sco.minimize(min_pcarisk, n1*[1./n1,], method = 'SLSQP', bounds = bnds, constraints = cons)
    return opts['x'].round(3)

basisdata=pd.read_excel('h://data.xlsx')
basisdata['return1']=basisdata['x1']/basisdata['x1'].shift(1)-1
basisdata['return2']=basisdata['x2']/basisdata['x2'].shift(1)-1
basisdata['return3']=basisdata['x3']/basisdata['x3'].shift(1)-1
basisdata['return4']=basisdata['x4']/basisdata['x4'].shift(1)-1
basisdata=basisdata.dropna()
basisdata=basisdata.reset_index(drop=True)

data1=basisdata[['date','return1','return2','return3','return4']]
m,n=data1.shape
#用多长时间计算均值和方差
window=120
#多长时间换仓一次
timehold=20
#主成分个数
global num
num=4
#加权结果为out
out=pd.DataFrame([],columns=['date','return_equal','return_Mark','return_Minvol','return_Riskequal','return_pca'])
weight_Mark=pd.DataFrame([],columns=['date','w1','w2','w3','w4'])
weight_Minvol=pd.DataFrame([],columns=['date','w1','w2','w3','w4'])
weight_riskequal=pd.DataFrame([],columns=['date','w1','w2','w3','w4'])
weight_pca=pd.DataFrame([],columns=['date','w1','w2','w3','w4'])
title=[]
for i in range(num):
    for j in range(4):
        name='x'+str(i)+str(j)
        title.append(name)
#风险平价主成分权重
name=title.copy()
name.append('date')
w=pd.DataFrame([],columns=name)

for i in range(window,m,timehold):
    #等权重收益率
    train=data1[['return1','return2','return3','return4']][i-window:i]
    train = train.reset_index(drop=True)
    predict=data1[['return1','return2','return3','return4']][i:i+timehold]
    predict=predict.reset_index(drop=True)
    return_equal=weightdata(predict,(n-1)*[1./(n-1)])
    #Markowitz和最小方差
    weight1,weight2=Markowitz(train)
    return_Mark = weightdata(predict,weight1)
    return_Minvol = weightdata(predict, weight2)
    #风险平价
    weight3=riskequal(train)
    return_Riskequal = weightdata(predict, weight3)
    #pca风险平价
    weight4=pcaequal(train)
    return_pca = weightdata(predict, weight4)
    #return数据
    outcome=pd.DataFrame(np.hstack((return_equal, return_Mark, return_Minvol, return_Riskequal, return_pca)),columns=['return_equal','return_Mark','return_Minvol','return_Riskequal','return_pca'])
    outcome['date']=data1['date'].values[i:i+timehold]
    out=out.append(outcome)
    # #权重数据
    a=pd.DataFrame([weight1],columns=['w1','w2','w3','w4'])
    a['date']=data1['date'].values[i]
    weight_Mark=weight_Mark.append(a,sort=False)
    b=pd.DataFrame([weight2],columns=['w1','w2','w3','w4'])
    b['date']=data1['date'].values[i]
    weight_Minvol=weight_Minvol.append(b,sort=False)
    c=pd.DataFrame([weight3],columns=['w1','w2','w3','w4'])
    c['date']=data1['date'].values[i]
    weight_riskequal=weight_riskequal.append(c,sort=False)
    d=pd.DataFrame([weight4],columns=['w1','w2','w3','w4'])
    d['date']=data1['date'].values[i]
    weight_pca=weight_pca.append(d,sort=False)
    e=train*weight4
    e1 = PCA(n_components=num)
    e1.fit(e)
    e2=pd.DataFrame([e1.components_.flatten()],columns=title)
    e2['date']=data1['date'].values[i]
    w=w.append(e2,sort=False)
out=out.reset_index(drop=True)
weight_Mark=weight_Mark.reset_index(drop=True)
weight_Minvol=weight_Minvol.reset_index(drop=True)
weight_riskequal=weight_riskequal.reset_index(drop=True)
weight_pca=weight_pca.reset_index(drop=True)
w=w.reset_index(drop=True)

##计算累计收益率
out['l1']=(out['return_equal']+1).cumprod()
out['l2']=(out['return_Mark']+1).cumprod()
out['l3']=(out['return_Minvol']+1).cumprod()
out['l4']=(out['return_Riskequal']+1).cumprod()
out['l5']=(out['return_pca']+1).cumprod()

writer=pd.ExcelWriter('h:/outcome.xlsx')
out.to_excel(writer,'out')
weight_Mark.to_excel(writer,'weight_Mark')
weight_Minvol.to_excel(writer,'weight_Minvol')
weight_riskequal.to_excel(writer,'weight_riskequal')
weight_pca.to_excel(writer,'weight_pca')
w.to_excel(writer,'w')