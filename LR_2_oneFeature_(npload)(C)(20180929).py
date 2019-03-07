import numpy as np
from numpy import *
import matplotlib.pyplot as plt

#显示中文
from pylab import *                             #显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']  #显示中文

#画图中显示负号
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False

#加载数据
data = np.loadtxt('ex1data1.txt', delimiter=',');print(data.shape) #;print(data)

#提取数据X,y
X = data[:, 0]#;print(X)
y = data[:, -1]#;print(y)

# #数据标准化/初始化X,y
m = X.shape[0]  #样本个数m
X = np.c_[np.ones((m)), X] #;print(X)
y = np.c_[y]; print(y.shape)#;print(y)

# #定义代价函数costFunction
def costFunction(X, y, theta):
    m = X.shape[0]    #样本个数m
    h = np.dot(X, theta)   #计算预测值h
    J = 1.0/(2*m)*np.dot((h-y).T, (h-y))   #计算代价函数值
    return J

#定义梯度下降算法函数
def gradDec(X, y, alpha=0.005, iter_num=15000):
    m, n = X.shape   #样本个数m,列数n
    theta = np.zeros((n, 1))   #初始化theta值
    J_history = np.zeros(iter_num)  #初始化代价历史值

    #开始梯度下降算法
    for i in range(iter_num):
        J_history[i] = costFunction(X, y, theta)  #计算代价函数值
        h = np.dot(X, theta)  #计算预测值
        deltatheta = 1.0/m*np.dot(X.T, (h-y))  #计算deltatheta
        theta -= alpha*deltatheta   #更新theta

    return J_history , theta

#执行梯度下降算法
J_history, theta=gradDec(X, y, iter_num=30000)

# #画代价函数图
plt.figure('代价函数图')
plt.title('代价函数')
plt.xlabel('迭代步数')
plt.ylabel('代价')
plt.plot(J_history)
plt.show()
#
# #画样本数据及回归线
h = np.dot(X, theta)  #计算预测值
plt.figure('样本数据及回归线')
plt.title('样本数据及回归线')
plt.xlabel('X')
plt.ylabel('y')
plt.scatter(X[:, 1], y, c='r', marker='x', label='样本')
plt.plot(X[:, 1], h[:, 0], label='回归线')
plt.legend(loc='upper left')
plt.show()
