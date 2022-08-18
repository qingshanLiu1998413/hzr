# Importing the libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import  random
from sklearn.metrics import r2_score
from xgboost import train

#初始化遗传算法 生成一个包含7个参数的向量
def initilialize_poplulation(numberOfParents):
    learningRate = np.empty([numberOfParents, 1])
    nEstimators = np.empty([numberOfParents, 1], dtype=np.uint8)
    maxDepth = np.empty([numberOfParents, 1], dtype=np.uint8)
    minChildWeight = np.empty([numberOfParents, 1])
    gammaValue = np.empty([numberOfParents, 1])
    subSample = np.empty([numberOfParents, 1])
    colSampleByTree = np.empty([numberOfParents, 1])
    for i in range(numberOfParents):
        print(i)
        learningRate[i] = round(random.uniform(0.01, 1), 2)
        nEstimators[i] = random.randrange(10, 1500, step=25)
        maxDepth[i] = int(random.randrange(1, 10, step=1))
        minChildWeight[i] = round(random.uniform(0.01, 10.0), 2)
        gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
        subSample[i] = round(random.uniform(0.01, 1.0), 2)
        colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)

    population = np.concatenate(
        (learningRate, nEstimators, maxDepth, minChildWeight, gammaValue, subSample, colSampleByTree), axis=1)
    return population
#使用初始种群训练我们的模型并计算适应度值
def fitness_f1score(y_true, y_pred):
    fitness = round((r2_score(y_true, y_pred)), 4)
    return fitness

#训练数据，找到适应度
def train_population(population, dMatrixTrain, dMatrixtest, y_test):
    fScore = []
    for i in range(population.shape[0]):
        param = {
            # 'objective':'binary:logistic',
              'learning_rate': population[i][0],
              # 'n_estimators': population[i][1],
              'max_depth': int(population[i][2]),
              'min_child_weight': population[i][3],
              'gamma': population[i][4],
              'subsample': population[i][5],
              'colsample_bytree': population[i][6],
              'seed': 24}
        num_round = 200
        xgbT = xgb.train(param, dMatrixTrain, num_round)
        preds = xgbT.predict(dMatrixtest)
        preds = preds>0.5
        fScore.append(fitness_f1score(y_test, preds))
    return fScore


#
def new_parents_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1]))  # 创建一个数组来存储最适合的父母对象

    # 寻找表现最好的父母
    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = 1  # 如果是F1分数，则将此值设置为负值，以便不再选择此父项
    return selectedParents


'''
匹配这些父母以创建具有来自这些父母的参数的孩子（我们使用统一交叉方法）
'''


def crossover_uniform(parents, childrenSize):
    crossoverPointIndex = np.arange(0, np.uint8(childrenSize[1]), 1, dtype=np.uint8)  # 获取所有索引
    crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]),
                                             np.uint8(childrenSize[1] / 2))  #随机选择一半索引
    crossoverPointIndex2 = np.array(
        list(set(crossoverPointIndex) - set(crossoverPointIndex1)))  # 选择剩余索引

    children = np.empty(childrenSize)

    '''
    通过从使用new_parent_selection函数选择的两个父项中选择参数来创建子项。参数值
将从以上随机选择的索引中选取。
    '''
    for i in range(childrenSize[0]):
        # 查找父母索引1
        parent1_index = i % parents.shape[0]
        # 查找父母索引2
        parent2_index = (i + 1) % parents.shape[0]
        # 基于父1中随机选择的索引插入参数
        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
        # 基于父1中随机选择的索引插入参数
        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]
    return children


def mutation(crossover, numberOfParameters):
    #定义每个参数允许的最小值和最大值
    minMaxValue = np.zeros((numberOfParameters, 2))
    minMaxValue[0:] = [0.01, 1.0]  # min/max learning rate
    minMaxValue[1, :] = [10, 2000]  # min/max n_estimator
    minMaxValue[2, :] = [1, 15]  # min/max depth
    minMaxValue[3, :] = [0, 10.0]  # min/max child_weight
    minMaxValue[4, :] = [0.01, 10.0]  # min/max gamma
    minMaxValue[5, :] = [0.01, 1.0]  # min/maxsubsample
    minMaxValue[6, :] = [0.01, 1.0]  # min/maxcolsample_bytree

    # 突变会随机改变每个后代的一个基因
    mutationValue = 0
    parameterSelect = np.random.randint(0, 7, 1)
    print(parameterSelect)
    if parameterSelect == 0:  # learning_rate
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 1:  # n_estimators
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 2:  # max_depth
        mutationValue = np.random.randint(-5, 5, 1)
    if parameterSelect == 3:  # min_child_weight
        mutationValue = round(np.random.uniform(5, 5), 2)
    if parameterSelect == 4:  # gamma
        mutationValue = round(np.random.uniform(-2, 2), 2)
    if parameterSelect == 5:  # subsample
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 6:  # colsample
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)

    # 通过改变一个参数引入变异，如果超出范围，则设置为最大或最小
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if (crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
        if (crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]
        return crossover

