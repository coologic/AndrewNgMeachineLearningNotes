# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 注意Jupyter Nodebooks需要使用绝对路径，
# 因为默认所有指令执行是由Anaconda3\lib\site-packages\ipykernel_launcher.py文件调用的
# 同时__file__也无法获取此文件路径
# Pycharm可以使用相对路径
from mpl_toolkits.mplot3d import Axes3D


def loadDataFile(dataFilePath):
    return pd.read_csv(dataFilePath, header=None, names=['x1', 'x2', 'y'])


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iterations_max_number):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.size  # 先规整输入矩阵为行向量再求大小
    cost_history = np.zeros(iterations_max_number)
    m = X.shape[0]
    for i in range(iterations_max_number):
        temp = theta - (alpha / m) * (X * theta.T - y).T * X
        theta = temp
        cost_history[i] = computeCost(X, y, temp)
        # 可以再增加其他终止条件，比如连续n次迭代结果之间的差小于一定值认为趋于稳定提前跳出
    final_theta = theta
    return final_theta, cost_history


if __name__ == "__main__":
    dataFilePath = r'./ex1/ex1data2.txt'  # path in windows or mac are not same
    sourcePdData = loadDataFile(dataFilePath)
    sourcePdData = (sourcePdData - sourcePdData.mean()) / sourcePdData.std() # 特征归一化
    # 创建子图图像，fig为整体图像，axs存储子图二维矩阵，每一个子图是一个坐标轴axes所以用ax，参数指图是1行2列
    # squeeze为False永远返回二维数组否则根据实际行列情况返回值、向量、二维数组
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.scatter3D(sourcePdData['x1'], sourcePdData['x2'], sourcePdData['y'], s=30, c='r', marker='x')
    # jnb可以直接再plt.scatter后显示，pycharm第一次使用plt需要加载一些库（可能是pyqt的)，并且需要用plt.show()才显示
    # show是阻塞函数，在pycharm中建议放在最后调用，中途的刷新显示可以用draw

    m=sourcePdData.shape[0]
    n=sourcePdData.shape[1]
    X = np.ones((m, 1)) # h=th0 * 1 + a2 * x1 + a3 * x2  需要先有系数为1的一列
    # 矩阵行列的拼接下面两种均可 r_是行拼接， axis=0是第一轴拼接，多维矩阵应该只能用insert未测试
    # X = np.insert(X, 1, values=sourcePdData['Population'], axis=1)
    X = np.c_[X, sourcePdData['x1'], sourcePdData['x2']]
    y = sourcePdData.iloc[:, n - 1:n] # 此方法可直接从pd数据中截取出矩阵，但和np构成的数据类型不同

    # 将类型一致化都使用矩阵
    X = np.matrix(X)
    y = np.matrix(y.values)
    #构建theta
    theta = np.matrix(np.array([0, 0, 0]))
    # theta = np.matrix(X.T * X).I * X.T * y
    print(X.shape, theta.shape, y.shape, X.shape[0])
    print("优化前", computeCost(X, y, theta), theta)
    # # 调用梯度下降
    iterations_max_number = 1000
    final_theta, cost_history = gradientDescent(X, y, theta, 0.01, iterations_max_number)

    print("优化后", computeCost(X, y, final_theta), final_theta)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(np.arange(iterations_max_number), cost_history)
    plt.show()
