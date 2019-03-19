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
    m,n=train.shape
    #约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    #我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0,1) for x in range(n))
    #优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    #最大化sharp值
    opts = sco.minimize(min_sharpe, n*[1./n,], method = 'SLSQP', bounds = bnds, constraints = cons)
    #最小化风险
    optv = sco.minimize(min_variance, n * [1./ n, ], method='SLSQP', bounds=bnds, constraints=cons)
    return opts['x'].round(3),optv['x'].round(3)

def min_risk(weights):
    weights = np.array(weights)
    var = train.var(0)*weights*252
    return np.std(var)


def riskequal(train):
    m,n=train.shape
    #约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    #我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0,1) for x in range(n))
    #优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    #最大化sharp值
    opts = sco.minimize(min_risk, n*[1./n,], method = 'SLSQP', bounds = bnds, constraints = cons)
    return opts['x'].round(3)

from sklearn.decomposition import PCA

def pcaequal(train):
    a = PCA(n_components=2)
    a.fit(train)
    y = a.transform(train)
    var=np.var(y,0)
    weight=[var[0]/(var[0]+var[1]),var[1]/(var[0]+var[1])]
    out=np.dot(weight, a.components_)
    return out/np.sum(out)

basisdata=pd.read_csv('f://data.csv')
basisdata['return1']=basisdata['x1'].shift(-1)/basisdata['x1']-1
basisdata['return2']=basisdata['x2'].shift(-1)/basisdata['x2']-1
basisdata['return3']=basisdata['x3'].shift(-1)/basisdata['x3']-1
basisdata['return4']=basisdata['x4'].shift(-1)/basisdata['x4']-1
basisdata=basisdata.dropna()
basisdata=basisdata.reset_index(drop=True)

data1=basisdata[['date','return1','return2','return3','return4']]
m,n=data1.shape
window=120
timehold=20
out=pd.DataFrame([],columns=['date','return_equal','return_Mark','return_Minvol','return_Riskequal','return_pca'])
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
    np.hstack((return_equal, return_Mark, return_Minvol, return_Riskequal, return_pca))
    outcome=pd.DataFrame(np.hstack((return_equal, return_Mark, return_Minvol, return_Riskequal, return_pca)),columns=['return_equal','return_Mark','return_Minvol','return_Riskequal','return_pca'])
    outcome['date']=data1['date'].values[i:i+timehold]
    out=out.append(outcome)


