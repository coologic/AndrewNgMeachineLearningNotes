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
    return pd.read_csv(dataFilePath, header=None, names=['Population', 'Profit'])


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
    dataFilePath = r'./ex1/ex1data1.txt'  # path in windows or mac are not same
    sourcePdData = loadDataFile(dataFilePath)
    # 创建子图图像，fig为整体图像，axs存储子图二维矩阵，每一个子图是一个坐标轴axes所以用ax，参数指图是1行2列
    # squeeze为False永远返回二维数组否则根据实际行列情况返回值、向量、二维数组
    fig, axs = plt.subplots(2, 2, squeeze = False)

    axs[0, 0].scatter(sourcePdData['Population'], sourcePdData['Profit'], s=30, c='r', marker='x')
    axs[0, 0].grid()  # 打开网络
    axs[0, 0].set_title('Profit of Population')
    # jnb可以直接再plt.scatter后显示，pycharm第一次使用plt需要加载一些库（可能是pyqt的)，并且需要用plt.show()才显示
    # show是阻塞函数，在pycharm中建议放在最后调用，中途的刷新显示可以用draw

    m=sourcePdData.shape[0]
    n=sourcePdData.shape[1]
    X = np.ones((m, 1))
    # 矩阵行列的拼接下面两种均可 r_是行拼接， axis=0是第一轴拼接，多维矩阵应该只能用insert未测试
    # X = np.insert(X, 1, values=sourcePdData['Population'], axis=1)
    X = np.c_[X, sourcePdData['Population']]
    y = sourcePdData.iloc[:, n - 1:n] # 此方法可直接从pd数据中截取出矩阵，但和np构成的数据类型不同

    # 将类型一致化都使用矩阵
    X = np.matrix(X)
    y = np.matrix(y.values)
    #构建theta
    theta = np.matrix([0, 0])

    # 初始值的计算结果 与作业对比32.07
    assert(np.sum(computeCost(X, y, theta), axis=0) - 32.7 < 0.01)

    # 调用梯度下降
    iterations_max_number = 1000
    final_theta, cost_history = gradientDescent(X, y, theta, 0.01, iterations_max_number)
    # print(final_theta, cost_history)

    # 绘图-训练数据和回归曲线（预测线）
    regression_line_x = np.linspace(sourcePdData.Population.min(), sourcePdData.Population.max(), 20) # 类似于matlab x=0:0.1:20,20是取点的总数量
    regression_line_y = final_theta[0, 0] + (final_theta[0, 1] * regression_line_x)
    axs[0, 1].plot(regression_line_x, regression_line_y, 'r', label='Prediction')
    axs[0, 1].scatter(sourcePdData.Population, sourcePdData.Profit, label='Training Data')
    axs[0, 1].legend(loc=2)  # 2表示在左上角
    axs[0, 1].grid()
    axs[0, 1].set_title('Predicted Profit by Training Data')

    # 绘图-迭代收敛速度
    axs[1, 0].plot(np.arange(iterations_max_number), cost_history)
    axs[1, 0].grid()
    axs[1, 0].set_title('Iterations Cost Result')

    # 绘图-目标函数值域内的云图
    theta1UnitformPoint = np.linspace(-10, 10, 100)
    theta2UnitformPoint = np.linspace(-1, 4, 100)

    J_vals = np.zeros((np.size(theta1UnitformPoint), np.size(theta2UnitformPoint))) # zeros第一个参数是shape，应该是一个元组
    print(J_vals[1,1])
    for i in range(len(theta1UnitformPoint)):
        for j in range(len(theta2UnitformPoint)):
            tempTheta = np.matrix([theta1UnitformPoint[i], theta2UnitformPoint[j]])
            J_vals[i, j] = computeCost(X, y, tempTheta)

    axs[1, 1].contour(theta1UnitformPoint, theta2UnitformPoint, J_vals, 20)
    # axes3d = Axes3D(axs[1, 2])
    # axs[1,2].
    # axs[1, 2].plot_surface(theta1UnitformPoint, theta2UnitformPoint, J_vals)
    fig2 = plt.figure()
    ax3D = Axes3D(fig2)

    X, Y = np.meshgrid(theta1UnitformPoint, theta2UnitformPoint)
    ax3D.plot_surface(X, Y, J_vals, cmap='rainbow')

    # 最后不能用fig.show需要用plt否则无法阻塞
    plt.show()
