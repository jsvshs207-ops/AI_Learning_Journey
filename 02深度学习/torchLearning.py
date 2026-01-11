# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# numpy
import numpy as np

# pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchLearning import *
from torch.utils.data import random_split

from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

# 一个cell输出多个结果
writer = SummaryWriter(log_dir= 'reg_loss')



def tensorGenReg(num_examples = 1000, w = [2, -1, -1], bias = True, delta = 0.01, deg = 1):
    """ 回归类数据集创建函数

    num_examples: 创建数据集的数据量
    w: 包括截距的特征系数张量
    bias: 是否需要截距
    delta: 扰动项取值
    deg: 方程次数
    return: 生成的特征张量和标签张量
    
    """

    if bias == True:
        num_inputs = len(w) - 1                                 
        features_true = torch.randn(num_examples, num_inputs)   # 特征张量
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()    # 自变量系数
        b_true = torch.tensor(w[-1]).float()                    # 截距
        if num_inputs == 1:                                     # 若输入特征只有一个，则不能使用矩阵的乘法
            labels_true = torch.pow(features_true, deg)*w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1)     # 在特征张量2的最后一列添加一列全是1的列
        labels = labels_true + torch.randn(size = labels_true.shape)*delta
    
    else:
        num_inputs = len(w)
        features_true = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_inputs == 1:                                 # 若输入特征只有一个，则不能使用矩阵的乘法
            labels_true = torch.pow(features_true, deg)*w_true 
        else:
            labels_true = torch.mm(torch.pow(features_true, deg), w_true)
        labels = labels_true + torch.randn(size = labels_true.shape)*delta
        features = features_true    # 在特征张量2的最后一列添加一列全是1的列
    return features,labels


def tensorGenCla(num_examples = 500, num_inputs = 2, num_class = 3, deg_dispersion = [4, 2], bias = False):
    """ 分类数据集创建函数
    num_examples: 每个类别的数据数量
    num_inputs: 数据集特征数量
    num_class: 数据集标签类别总数
    deg_dispersion: 数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考，第二个参数表示随机数组标准差。
    bias: 建立模型逻辑回归模型时是否带入截距
    return: 生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组
    """

    cluster_1 = torch.empty(num_examples, 1)        # 每一类标签张量的形状
    mean_ = deg_dispersion[0]                       # 每一类特征张量的均值的参考值
    std_ = deg_dispersion[1]                        # 每一类特征张量的方差
    lf = []                                         # 用于存储每一类特征张量的列表容器
    ll = []                                         # 用于存储每一类标签张量的列表容器
    k = mean_*(num_class - 1) / 2                 # 每一类特征张量的惩罚因子

    for i in range(num_class):
        data_temp = torch.normal(i*mean_-k, std_, size=(num_examples, num_inputs))      # 生成每一类张量
        lf.append(data_temp)                                                            # 将每一类张量添加到Lf中
        labels_temp = torch.full_like(cluster_1, i)                                     # 生成标签
        ll.append(labels_temp)
    
    features = torch.cat(lf).float()
    labels = torch.cat(ll).long()

    if bias == True:
        features = torch.cat((features, torch.ones(len(features), 1)), 1)
    return features, labels


def data_iter(batch_size, features, labels):
    """ 数据切分函数
    batch_size : 每个子数据集包含多少数据
    features: 输入的特征张量
    labels: 输入的标签张量
    return l: 包含batch_size个列表,每个列表由切分后的特征和标签所构成
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    l = []
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        l.append([torch.index_select(features, 0, j), torch.index_select(labels, 0, j)])
    return l

# 创建一个针对手动创建数据的数据类
class GenData(Dataset):
    def __init__(self, features, labels) :      # 创建该类时需要输入的数据集
        self.features = features                # features属性返回数据集特征
        self.labels = labels                    # labels 属性返回数据集标签
        self.lens = len(features)               # lens属性返回数据集大小
    
    def __getitem__(self, index) :
        # 调用该方法时需要输入 index数值，方法最终返回index对应的特征和标签
        return self.features[index, :], self.labels[index]
    
    def __len__(self):
         # 调用该方法不需要输入额外参数，方法最终返回数据集大小
        return self.lens



def split_loader(features, labels, batch_size = 10, rate = 0.7):
    """数据封装，切分，和加载函数：
    
    param features: 输入的特征
    param labels: 数据集标签张量
    param batch_size: 数据加载时的每一个小批数据量
    param rate: 训练集数据占比
    return: 加载好的训练集和测试集启发式算法学习课程
    """

    data = GenData(features, labels)
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    data_train, data_test = random_split(data, [num_train, num_test])
    train_loader = DataLoader(data_train, batch_size= batch_size, shuffle= True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle= False)
    return (train_loader, test_loader)




def fit(net, criterion, optimizer, batchdata, epochs = 3, cla = False):
    """模型训练函数
    
    param net: 待训练模型
    param criterion: 损失函数
    param optimizer: 优化算法
    param batchdata: 训练数据集
    param cla: 是否是分类问题
    param epochs: 遍历数据次数
    """

    for epoch in range(epochs):
        for X, y in batchdata:
            if cla == True:
                y = y.flatten().long()      # 如果是分类问题，需要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



def mse_cal(data_loader, net):
    """ mse计算函数
    
    param data_loader: 加载好的数据
    param net: 模型
    return: 根据输入的数据, 输出其MSE计算结果
    """
    data = data_loader.dataset      # 还原Dataset类
    X = data[:][0]                  # 还原数据的特征
    y = data[:][1]                  # 还原数据的标签
    yhat = net(X)
    return F.mse_loss(yhat, y)


def accuracy_cal(data_loader, net):
    """ 准确率
    data_loader: 加载好的数据
    net: 模型
    return: 根据输入的数据，输出其准确率计算结果
    """

    data = data_loader.dataset      # 还原Dataset类
    x = data[:][0]                  # 还原数据的特征
    y = data[:][1]                  # 还原数据的标签
    zhat = net(x)                   # 默认是分类问题，且输出结果是未经过softmax转化的结果
    soft_z = F.softmax(zhat, 1)
    acc_bool = torch.argmax(soft_z, 1).flatten() == y.flatten()
    acc = torch.mean(acc_bool.float())
    return acc


# 构建一个三个隐藏层的神经网络
class Relu_class3(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, in_hidden3 = 4, out_features = 1, bias =True) :
        super(Relu_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.linear3 = nn.Linear(in_hidden2, in_hidden3, bias= bias)
        self.out = nn.Linear(in_hidden3, out_features, bias= bias)
     
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.relu(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.relu(z2)
        z3 = self.linear3(sigma2)
        sigma3 = torch.relu(z3)
        zhat = self.out(sigma3)
        return zhat

# 构建一个四个隐藏层的神经网络
class Relu_class4(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, in_hidden3 = 4, in_hidden4 =4, out_features = 1, bias =True) :
        super(Relu_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.linear3 = nn.Linear(in_hidden2, in_hidden3, bias= bias)
        self.linear4 = nn.Linear(in_hidden3, in_hidden4, bias= bias)
        self.out = nn.Linear(in_hidden4, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.relu(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.relu(z2)
        z3 = self.linear3(sigma2)
        sigma3 = torch.relu(z3)
        z4 = self.linear4(sigma3)
        sigma4 = torch.relu(z4)
        zhat = self.out(sigma4)
        return zhat

# sigmoid激活函数
class tanh_class1(nn.Module):
    def __init__(self, in_features= 2, n_hidden = 4, out_features =1, bias =True) :
        super(tanh_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias= bias)
        self.linear2 = nn.Linear(n_hidden, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.tanh(z1)
        out = self.linear2(sigma1)
        return out


class tanh_class2(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, out_features = 1, bias =True) :
        super(tanh_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.out = nn.Linear(in_hidden2, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.tanh(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.tanh(z2)
        zhat = self.out(sigma2)
        return zhat

# 构建一个三个隐藏层的神经网络
class tanh_class3(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, in_hidden3 = 4, out_features = 1, bias =True) :
        super(tanh_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.linear3 = nn.Linear(in_hidden2, in_hidden3, bias= bias)
        self.out = nn.Linear(in_hidden3, out_features, bias= bias)
     
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.tanh(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.tanh(z2)
        z3 = self.linear3(sigma2)
        sigma3 = torch.tanh(z3)
        zhat = self.out(sigma3)
        return zhat

# 构建一个四个隐藏层的神经网络
class tanh_class4(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, in_hidden3 = 4, in_hidden4 =4, out_features = 1, bias =True) :
        super(tanh_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.linear3 = nn.Linear(in_hidden2, in_hidden3, bias= bias)
        self.linear4 = nn.Linear(in_hidden3, in_hidden4, bias= bias)
        self.out = nn.Linear(in_hidden4, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.tanh(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.tanh(z2)
        z3 = self.linear3(sigma2)
        sigma3 = torch.tanh(z3)
        z4 = self.linear4(sigma3)
        sigma4 = torch.tanh(z4)
        zhat = self.out(sigma4)
        return zhat
    

# sigmoid激活函数
class Sigmoid_class1(nn.Module):
    def __init__(self, in_features= 2, n_hidden = 4, out_features =1, bias =True) :
        super(Sigmoid_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias= bias)
        self.linear2 = nn.Linear(n_hidden, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.sigmoid(z1)
        out = self.linear2(sigma1)
        return out


class Sigmoid_class2(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, out_features = 1, bias =True) :
        super(Sigmoid_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.out = nn.Linear(in_hidden2, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.sigmoid(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.sigmoid(z2)
        zhat = self.out(sigma2)
        return zhat

# 构建一个三个隐藏层的神经网络
class Sigmoid_class3(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, in_hidden3 = 4, out_features = 1, bias =True) :
        super(Sigmoid_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.linear3 = nn.Linear(in_hidden2, in_hidden3, bias= bias)
        self.out = nn.Linear(in_hidden3, out_features, bias= bias)
     
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.sigmoid(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.sigmoid(z2)
        z3 = self.linear3(sigma2)
        sigma3 = torch.sigmoid(z3)
        zhat = self.out(sigma3)
        return zhat

# 构建一个四个隐藏层的神经网络
class Sigmoid_class4(nn.Module):
    def __init__(self, in_features=2, in_hidden1 = 4, in_hidden2 = 4, in_hidden3 = 4, in_hidden4 =4, out_features = 1, bias =True) :
        super(Sigmoid_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, in_hidden1, bias= bias)
        self.linear2 = nn.Linear(in_hidden1, in_hidden2, bias= bias)
        self.linear3 = nn.Linear(in_hidden2, in_hidden3, bias= bias)
        self.linear4 = nn.Linear(in_hidden3, in_hidden4, bias= bias)
        self.out = nn.Linear(in_hidden4, out_features, bias= bias)
    
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.sigmoid(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.sigmoid(z2)
        z3 = self.linear3(sigma2)
        sigma3 = torch.sigmoid(z3)
        z4 = self.linear4(sigma3)
        sigma4 = torch.sigmoid(z4)
        zhat = self.out(sigma4)
        return zhat
    

def model_comparison(model_1,
                     name_1,
                     train_data,
                     test_data, 
                     num_epochs = 20,
                     criterion = nn.MSELoss(),
                     optimizer = optim.SGD,
                     lr = 0.03,
                     cla = False,
                     eva = mse_cal):
    """模型对比函数
    
    param model_1 :模型序列
    param name_1: 模型名称序列
    param train_data: 训练数据
    param test_data: 测试数据
    param num_epochs: 迭代论数
    param criterion: 损失函数
    param lr :学习率
    param cal: 是否是分类模型
    return: MSE张量矩阵
    """

    # 模型评估指标矩阵
    train_1 = torch.zeros(len(model_1,), num_epochs)
    test_1 = torch.zeros(len(model_1), num_epochs)

    # 模型训练过程
    for epochs in range(num_epochs):
        for i, model in enumerate(model_1):
            model.train()
            fit(net= model,
                criterion= criterion,
                optimizer= optimizer(model.parameters(), lr= lr),
                batchdata= train_data,
                epochs= epochs,
                cla= cla)
            model.eval()
            train_1[i][epochs] = eva(train_data, model).detach()
            test_1[i][epochs] = eva(test_data, model).detach()
    return train_1, test_1


def weights_vp(model, att = 'grad'):
    """ 观察各层参数取值和梯度的小提琴图绘图函数。
    
    param model: 观察对象（模型）
    param att: 选择参数梯度(grad)还是参数取值(weights)进行观察
    return: 对应att的小提琴图
    """

    vp = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            if att == 'grad':
                vp_x = m.weight.grad.detach().reshape(-1, 1).numpy()
            else:
                vp_x = m.weight.detach().reshape(-1, 1).numpy()
            vp_y = np.full_like(vp_x, i)
            vp_a = np.concatenate((vp_x, vp_y), 1)
            vp.append(vp_a)
    vp_r = np.concatenate((vp), 0)
    ax = sns.violinplot(y = vp_r[:, 0], x = vp_r[:, 1])
    ax.set(xlabel= 'num_hidden', title= att)


def model_train_test(model,
                     train_data,
                     test_data,
                     num_epochs = 20,
                     criterion = nn.MSELoss(),
                     optimizer = optim.SGD,
                     lr = 0.03,
                     cla = False,
                     eva = mse_cal):
    """ 模型误差测试函数：
    
    param mldel: 模型
    param train_data: 训练数据
    param test_data: 测试数据
    param num_epochs: 迭代论数
    param lr: 学习率
    param cal: 是否是分类模型
    return: MSE列表
    """

    # 模型评估指标列表
    train_1 = []
    test_1 = []
    
    # 模型训练过程
    for epochs in range(num_epochs):
        model.train()
        fit(net= model,
            criterion= criterion,
            optimizer= optimizer(model.parameters(), lr = lr),
            batchdata= train_data,
            epochs= epochs,
            cla = cla)
        model.eval()
        train_1.append(eva(train_data, model).detach())
        test_1.append(eva(test_data, model).detach())
    return train_1, test_1


# 有BN层的神经网络的构建


class net_class1(nn.Module):
    def __init__(self, act_fun = torch.relu, in_features = 2, n_hidden= 4, out_features = 1, bias = True, BN_model = None, momentum = 0.1) :
        super(net_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias= bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden, momentum= momentum)
        self.linear2 = nn.Linear(n_hidden, out_features, bias= bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
    
    def forward(self, x):
        out = None
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            out = self.linear2(p1)
        
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            out = self.linear2(p1)
        
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            out = self.linear2(self.normalize1(p1))
        return out


class net_class2(nn.Module):
    def __init__(self, act_fun = torch.relu, in_features = 2, n_hidden1= 4, n_hidden2= 4,  out_features = 1, bias = True, BN_model = None, momentum = 0.1) :
        super(net_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias= bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum= momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias= bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, out_features, momentum= momentum)
        self.linear3 = nn.Linear(n_hidden2, out_features, bias= bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
    
    def forward(self, x):
        out = None
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            out = self.linear3(p2)
        
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            out = self.linear3(p2)
        
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            out = self.linear3(self.normalize1(p2))
        return out


class net_class3(nn.Module):
    def __init__(self, act_fun = torch.relu, in_features = 2, n_hidden1= 4, n_hidden2= 4,  n_hidden3=4, out_features = 1, bias = True, BN_model = None, momentum = 0.1) :
        super(net_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias= bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum= momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias= bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, momentum= momentum)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias= bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3, momentum= momentum)
        self.linear4 = nn.Linear(n_hidden3, out_features, bias= bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
    
    def forward(self, x):
        out = None
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            z3 = self.linear3(p2)
            p3 = self.act_fun(z3)
            out = self.linear4(p3)
        
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = self.act_fun(z3)
            out = self.linear4(p3)
        
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = self.act_fun(z3)
            out = self.linear4(self.normalize3(p3))
        return out


class net_class4(nn.Module):
    def __init__(self, act_fun = torch.relu, in_features = 2, n_hidden1= 4, n_hidden2= 4,  n_hidden3=4, n_hidden4 = 4,  out_features = 1, bias = True, BN_model = None, momentum = 0.1) :
        super(net_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias= bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum= momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias= bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, momentum= momentum)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias= bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3, momentum= momentum)
        self.linear4 = nn.Linear(n_hidden3, n_hidden4, bias= bias)
        self.normalize4 = nn.BatchNorm1d(n_hidden4, momentum= momentum)
        self.linear5 = nn.Linear(n_hidden4, out_features, bias= bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
    
    def forward(self, x):
        out = None
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            z3 = self.linear3(p2)
            p3 = self.act_fun(z3)
            z4 = self.linear4(p3)
            p4 = self.act_fun(z4)
            out = self.linear5(p4)
        
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = self.act_fun(z3)
            z4 = self.normalize4(self.linear4(p3))
            p4 = self.act_fun(z4)
            out = self.linear4(p4)
        
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = self.act_fun(z3)
            z4 = self.linear4(self.normalize3(p3))
            p4 = self.act_fun(z4)
            out = self.linear5(self.normalize4(p4))
        return out

def fit_rec(net, criterion, optimizer, train_data, test_data, epochs = 3, cla = False, eva= mse_cal):
    """模型训练函数(记录每一次遍历后模型评估指标)
    
    param net: 待训练模型
    param criterion: 损失函数
    param optimizer: 优化算法
    param batchdata: 训练数据集
    param cla: 是否是分类问题
    param epochs: 遍历数据次数
    """
    train_1 = []
    test_1 = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_data:
            if cla == True:
                y = y.flatten().long()      # 如果是分类问题，需要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        train_1.append(eva(train_data, net).detach())
        test_1.append(eva(test_data, net).detach())
    return train_1, test_1


def fit_rec_sc(net, 
            criterion, 
            optimizer, 
            train_data, 
            test_data,
            scheduler, 
            epochs = 3, 
            cla = False, 
            eva= mse_cal):
    """加入学习率调度后的模型训练函数(记录每一次遍历后模型评估指标)
    
    param net: 待训练模型
    param criterion: 损失函数
    param optimizer: 优化算法
    param batchdata: 训练数据集
    param scheduler: 学习率调度器
    param cla: 是否是分类问题
    param epochs: 遍历数据次数
    param eva: 模型评估方法
    """
    train_1 = []
    test_1 = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_data:
            if cla == True:
                y = y.flatten().long()      # 如果是分类问题，需要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        train_1.append(eva(train_data, net).detach())
        test_1.append(eva(test_data, net).detach())
    return train_1, test_1
