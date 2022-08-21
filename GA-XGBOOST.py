import geneticXGboost
import numpy as np
import pandas as pd
import xgboost as xgb
import  random
import numpy as np
import pandas as pd
import geneticXGboost  # this is the module we crated above
import xgboost as xgb

np.random.seed(723)

# Importing the dataset
dataset = pd.read_csv('特征工程.csv')
dataset.drop(["Unnamed: 0"],axis=1,inplace=True)
X = dataset.iloc[:, 1:9]  # discard first two coloums as these are molecule's name and conformation's name

y = dataset.iloc[:, -1]  # extrtact last coloum as class (1 =&gt; desired odor, 0 =&gt; undesired odor)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=97)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# XGboost Classifier

# model xgboost
# use xgboost API now
xgDMatrix = xgb.DMatrix(X_train, y_train)  # create Dmatrix
xgbDMatrixTest = xgb.DMatrix(X_test, y_test)
# np.random.seed(723)
# # Importing the dataset
# dataset = pd.read_csv('特征工程.csv')
# dataset.drop(["Unnamed: 0"],axis=1,inplace=True)
# # X = dataset.iloc[:, 2:168].values #discard first two coloums as these are molecule's name and conformation's name
# # y = dataset.iloc[:, 168].values #extrtact last coloum as class (1 => desired odor, 0 => undesired odor)
# X = dataset.iloc[:,1:9]
# y = dataset.iloc[:,-1]
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 97)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# # Splitting the dataset into the Training set and Test set
#
# #XGboost Classifier
#
# #model xgboost
# #use xgboost API now
# xgDMatrix = xgb.DMatrix(X_train, y_train) #create Dmatrix
# xgbDMatrixTest = xgb.DMatrix(X_test, y_test)

numberOfParents = 20  # number of parents to start
numberOfParentsMating = 10  # number of parents that will 交配
numberOfParameters = 7  # number of parameters that will be optimized
numberOfGenerations =10  # number of genration that will be created
#定义总群大小
populationSize = (numberOfParents, numberOfParameters)
#使用随机生成的参数初始化总体
population = geneticXGboost.initilialize_poplulation(numberOfParents)
# 定义一个数组来存储适应度记录
fitnessHistory = np.empty([numberOfGenerations + 1, numberOfParents])
# 定义一个数组来存储每个父代和子代代的每个参数的值
populationHistory = np.empty([(numberOfGenerations + 1) * numberOfParents, numberOfParameters])
# 在历史记录中插入初始参数的值
populationHistory[0:numberOfParents, :] = population
for generation in range(numberOfGenerations):
    print("This is number %s generation" % (generation))

    # 训练数据集并获得适应度
    fitnessValue = geneticXGboost.train_population(population=population, dMatrixTrain=xgDMatrix,
                                                   dMatrixtest=xgbDMatrixTest, y_test=y_test)
    fitnessHistory[generation, :] = fitnessValue

    # 当前迭代中的最佳得分
    print('Best F1 score in the this iteration = {}'.format(np.max(fitnessHistory[generation, :])))
    #适者生存 - 根据适合度值和需要选择的父母数量，选择最好的父母
    parents = geneticXGboost.new_parents_selection(population=population, fitness=fitnessValue,
                                                   numParents=numberOfParentsMating)

    # 匹配这些父母以创建具有这些父母参数的孩子（我们使用统一交叉）
    children = geneticXGboost.crossover_uniform(parents=parents,
                                                childrenSize=(populationSize[0] - parents.shape[0], numberOfParameters))

    # 添加变异以创造遗传多样性
    children_mutated = geneticXGboost.mutation(children, numberOfParameters)

    '''
    我们将创建一个新的群体，其中将包含以前根据健康评分选择的父母，其余的将是孩子
    '''
    population[0:parents.shape[0], :] = parents  # 最佳父代
    population[parents.shape[0]:, :] = children_mutated  # 子代

    populationHistory[(generation + 1) * numberOfParents: (generation + 1) * numberOfParents + numberOfParents,
    :] = population  # srore parent information
    # 来自最终迭代的最佳解决方案
fitness = geneticXGboost.train_population(population=population, dMatrixTrain=xgDMatrix, dMatrixtest=xgbDMatrixTest,
                                          y_test=y_test)
fitnessHistory[generation + 1, :] = fitness
# 最佳解指数
bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]
# Best fitness
print("Best fitness is =", fitness[bestFitnessIndex])
# Best parameters
print("Best parameters are:")
print('learning_rate=', population[bestFitnessIndex][0])
print('n_estimators=', population[bestFitnessIndex][1])
print('max_depth=', int(population[bestFitnessIndex][2]))
print('min_child_weight=', population[bestFitnessIndex][3])
print('gamma=', population[bestFitnessIndex][4])
print('subsample=', population[bestFitnessIndex][5])
print('colsample_bytree=', population[bestFitnessIndex][6])