import pandas
import numpy as np
import h5py
from scipy.spatial.distance import pdist, squareform

train_data_size = 7291
test_data_size = 2007


def knn(train_data, test_data, train_label, test_label):
    accuracy = 0  # 准确率
    for i in range(test_data_size):
        result = -1  # 定义数据类型
        count = np.zeros(10)  # 手写数字识别，定义计数数组（对应从0到9）一维
        dist1 = np.zeros((train_data_size, 2))
        dist2 = np.zeros((train_data_size, 2))
        dist3 = np.zeros((train_data_size, 2))
        dist4 = np.zeros((train_data_size, 2))
        dist5 = np.zeros((train_data_size, 2))
        # 距离共五类，一列存储类型，一列存储距离
        for j in range(train_data_size):
            #dist1[j, 1] = np.linalg.norm(test_data[i]-train_data[j], ord=2)  # 欧式距离
            #dist2[j, 1] = np.linalg.norm(test_data[i]-train_data[j], ord=1)  # 曼哈顿距离
            #dist3[j, 1] = np.linalg.norm(test_data[i]-train_data[j], ord=np.inf)  # 切比雪夫距离
            #dist4[j, 1] = np.linalg.norm(test_data[i]-train_data[j], ord=2) / np.std(train_data[j])  # 标准化欧式距离
            dist5[j, 1] = np.dot(test_data[i], train_data[j].T) / (np.linalg.norm(test_data[i]) * np.linalg.norm(train_data[j]))
            # 余弦距离
            #dist1[j, 0] = train_label[j]
            #dist2[j, 0] = train_label[j]
            #dist3[j, 0] = train_label[j]
            #dist4[j, 0] = train_label[j]
            dist5[j, 0] = train_label[j]
            # print(train_data[j])
        #order1 = dist1[np.lexsort(dist1.T)]
        #order2 = dist2[np.lexsort(dist2.T)]
        #order3 = dist3[np.lexsort(dist3.T)]
        #order4 = dist4[np.lexsort(dist4.T)]
        order5 = dist5[np.lexsort(dist5.T)]
        # print(i, order5)
        # 按第二列的距离数据对整个distance二维数组升序排序
        for loop in range(k):  # k代表"K聚类"中的"K"是几
            maxdist_label = order5[loop, 0].astype(int)
            # print(maxdist_label)
            # 获取k个最小距离对应的数据标签(即这个数是几)，并转换类型为int
            count[maxdist_label] += 1
            # count计数数组内对应的格子计数+1
        result = count.argmax()
        # 找到count数组内最大计数的格子，并将这个格子对应的数值作为预测结果
        # print(test_label[i])
        if result == test_label[i]:
            accuracy += 1
    accuracy = accuracy/test_data_size
    if k == 1:
        print("usps手写体数字数据集的", "最", "近邻准确率为：", accuracy)
    else:
        print("usps手写体数字数据集的", k, "近邻准确率为：", accuracy)


with h5py.File('usps.h5') as hf:
    train = hf.get('train')
    x_train = train.get('data')[:]
    y_train = train.get('target')[:]
    test = hf.get('test')
    x_test = test.get('data')[:]
    y_test = test.get('target')[:]
# 数据预处理（固定类型固定代码，网上直接摘取的）

train_data = np.mat(pandas.DataFrame(x_train))  # 提取训练特征数据并转为矩阵
test_data = np.mat(pandas.DataFrame(x_test))  # 提取测试特征数据并转为矩阵
train_label = np.mat(pandas.DataFrame(y_train)).astype(int)  # 提取训练数据的label（这个数是几）并转为矩阵，转换为int类型
test_label = np.mat(pandas.DataFrame(y_test)).astype(int)  # 提取训练数据的label（这个数是几）并转为矩阵，转换为int类型
# print(train_label)
# print(test_label)
for Loop in range(3):
    k = Loop + 1
    knn(train_data, test_data, train_label, test_label)