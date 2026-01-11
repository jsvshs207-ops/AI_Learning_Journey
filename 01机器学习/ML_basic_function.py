import numpy as np
import pandas as pd

def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数
    :param num_examples: 样本数
    :param w: 线性方程系数
    :param bias: 是否有偏置项
    :param delta: 扰动项
    :param deg: 多项式的次数
    :return: 特征和标签
    """

    if bias:
        num_inputs = len(w) - 1
        features_true = np.random.randn(num_examples, num_inputs) # 特征
        w_true = np.array(w[:-1]).reshape(-1, 1) # 线性方程系数
        b_true = np.array(w[-1])
        labels_true = np.dot(np.power(features_true, deg), w_true) + b_true    # 严格满足人造规律的标签
        features = np.concatenate((features_true, np.ones((num_examples, 1))), axis=1)      # 加上全为1的一列之后的特征
    else:
        num_inputs = len(w)
        features_true = np.random.randn(num_examples, num_inputs) # 特征个数
        w_true = np.array(w).reshape(-1, 1)
        labels_true = np.dot(np.power(features_true, deg), w_true)
        features = features_true
    
    labels = labels_true + np.random.normal(0, size=labels_true.shape) * delta   # 扰动项
    return features, labels


def SSELoss(x, w, y):
    """SSE计算函数
    :param x: 特征
    :param w: 权重
    :param y: 标签
    """
    y_hat = np.dot(x, w)
    return (y - y_hat).T.dot(y - y_hat)

def MSE	(x, w, y):
    """MSE计算函数
    :param x: 特征
    :param w: 权重
    :param y: 标签
    """
    y_hat = np.dot(x, w)
    return np.mean((y - y_hat) ** 2)


def array_split(features, labels, rate = 0.7, random_state = 24):
    '''
    将数据集按照rate比例切分为训练集和测试集
    '''
    # 保证数据集的数据类型为numpy数组
    features = np.array(features)
    labels = np.array(labels)
    
    # 随机打乱数据集
    np.random.seed(random_state)
    np.random.shuffle(features)
    np.random.seed(random_state)
    np.random.shuffle(labels)
    
    # 切分数据集
    num_inputs = len(features)
    split_indices = int(num_inputs * rate)
    Xtrain, Xtest = np.vsplit(features, [split_indices,])
    ytrain, ytest = np.vsplit(labels, [split_indices,])

    return Xtrain, Xtest, ytrain, ytest

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def logit_cla(yhat, thr = 0.5):
    """逻辑回归类别判别函数
    :param yhat: 预测值
    :param thr: 阈值
    :return: 类别判别结果
    """
    ycla = np.zeros_like(yhat)
    ycla[yhat > thr] = 1
    return ycla

def lr_gd(X, w, y):
    """线性回归梯度计算公式
    """
    m = X.shape[0]
    grad = 2 * X.T.dot(X.dot(w) - y) / m
    return grad


def w_cal(X, w, y, gd_cal, lr = 0.02, itera_times = 20):
    """梯度下降中参数更新函数
    :param X: 特征矩阵
    :param w: 权重
    :param y: 标签
    :param gd_cal: 梯度计算函数
    :param lr: 学习率
    :param itera_times: 迭代次数
    :return: 每一轮迭代的参数计算结果列表
    """
    for i in range(itera_times):
        w -= lr * gd_cal(X, w, y)
    return w

def w_cal_rec(X, w, y, gd_cal, lr = 0.02, itera_times = 20):
    w_res = [np.copy(w)]
    for i in range(itera_times):
        w -= lr * gd_cal(X, w, y)
        w_res.append(np.copy(w))
    return w, w_res


def sgd_cal(X, w, y, gd_cal, epoch, batch_size=1, lr=0.02, shuffle=True, random_state=24):
    """随机梯度下降和小批量梯度下降计算函数
    :param X: 训练数据
    :param w: 权重
    :param y: 标签
    :param gd_cal: 梯度计算函数 
    :param epoch: 迭代次数
    :param batch_size: 批量大小
    :param lr: 学习率
    :param shuffle: 是否打乱数据
    :param random_state: 随机数种子
    """
    m = X.shape[0]
    n = X.shape[1]
    batch_num = np.ceil(m / batch_size)
    X = np.copy(X)
    y = np.copy(y)
    for j in range(epoch):
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(X)
            np.random.seed(random_state)
            np.random.shuffle(y)
        for i in range(np.int64(batch_num)):
            w = w_cal(X[i * batch_size: np.min([(i+1)*batch_size, m])], w, 
                      y[i * batch_size: np.min([(i+1)*batch_size, m])], 
                      gd_cal= gd_cal, lr = lr, itera_times = 1)
    return w


def logit_gd(X, w, y):
    """逻辑回归梯度计算公式
    :param X: 输入数据
    :param w: 权重
    :param y: 标签"""
    m = X.shape[0]
    grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y) / m
    return grad

def logit_acc(x,  w, y, thr=0.5):
    """逻辑回归准确率计算公式
    :param x: 输入数据
    :param w: 权重
    :param y: 标签
    :param thr: 阈值
    :return: 准确率"""
    yhat = sigmoid(x.dot(w))
    yhat = logit_cla(yhat, thr)
    acc = (yhat == y).mean()
    return acc