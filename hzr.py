from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#导入数据
data = pd.read_csv("2.csv")
data.drop(["Unnamed: 0"],axis=1,inplace=True)

#划分下特征和预测量

X = data.iloc[:,1:8]
y = data.iloc[:,-1]
#划分训练集和测试集
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.2,random_state=240)
#弄进XGBOOST模型训练
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain)
#预测
reg.predict(Xtest)
#打分看看效果
print(reg.score(Xtest,Ytest))

#看一些预测效果的参数

print(MSE(Ytest,reg.predict(Xtest)))
print(reg.feature_importances_)
#交叉验证
reg = XGBR(n_estimators=100)#没有训练过的模型

CVS(reg,Xtrain,Ytrain,cv=5).mean()
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()


#和随机森林做一个比对
rfr = RFR(n_estimators=100)
CVS(rfr,Xtrain,Ytrain,cv=5).mean()
CVS(rfr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
#线性回归对比
lr = LinearR()
CVS(lr,Xtrain,Ytrain,cv=5).mean()
CVS(lr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()

#画一下学习率的图 ，测试集和训练集都可以看出来
def plot_learning_curve(estimator,title, X, y,
                        ax=None, #选择子图
                        ylim=None, #设置纵坐标的取值范围
                        cv=None, #交叉验证
                        n_jobs=None #设定索要使用的线程
                       ):

    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            ,shuffle=True
                                                            ,cv=cv
                                                            ,random_state=420
                                                            ,n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid() #绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g",label="Test score")
    ax.legend(loc="best")
    return ax


plot_learning_curve(XGBR(n_estimators=170,random_state=240),"XGB",Xtrain,Ytrain)
plt.show()


###看看树的多少对结果的影响
axisx = range(10,1010,20)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=240)
    rs.append(CVS(reg,Xtrain,Ytrain).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
