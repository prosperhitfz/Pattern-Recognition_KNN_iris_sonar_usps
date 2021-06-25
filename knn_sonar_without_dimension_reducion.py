import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform


number = 208  # 总样本个数
feature_number = 60  # 每个样本的特征个数


def knn(array):
    distance1 = pdist(array, 'euclidean')  # 欧式距离（矩阵
    distance2 = pdist(array, 'minkowski', p=2.)  # 明氏距离（矩阵
    distance3 = pdist(array, 'cityblock')  # 曼哈顿距离（矩阵
    distance4 = pdist(array, 'seuclidean', V=None)  # 标准欧几里得距离（向量
    distance5 = pdist(array, 'cosine')  # 余弦距离（向量
    accuracy = 0  # 准确率
    for i in range(number):
        classifier1 = 0
        classifier2 = 0
        # 分类器2类
        result = 0  # 数据类型
        train = np.delete(array, i, axis=0)  # 定义训练集是按行删除第i行数据后的数据集
        test = array[i]  # 定义测试集由删除的第i行数据组成
        # print(train[i, feature_number])
        # print(test[feature_number])
        train_data = np.zeros((number - 1, feature_number))
        train_data[:, 0:feature_number] = train[:, 0:feature_number]
        # print(train_data)
        test_data = np.zeros((1, feature_number))
        # print(test_data)
        for loop1 in range(feature_number):
            test_data[0, loop1] = test[loop1]
        # print(test_data)
        # print(np.shape(test))
        # print(test)
        # print("*",test[4])
        dist1 = np.zeros((number-1, 2))
        dist2 = np.zeros((number-1, 2))
        dist3 = np.zeros((number-1, 2))
        dist4 = np.zeros((number-1, 2))
        dist5 = np.zeros((number-1, 2))
        # 距离共五类，一列存储类型，一列存储距离
        # print(len(distance1))
        for j in range(number-1):
            dist1[j, 1] = np.linalg.norm(test_data-train_data[j], ord=2)  # 欧式距离
            dist2[j, 1] = np.linalg.norm(test_data-train_data[j], ord=1)  # 曼哈顿距离
            dist3[j, 1] = np.linalg.norm(test_data-train_data[j], ord=np.inf)  # 切比雪夫距离
            dist4[j, 1] = np.linalg.norm(test_data-train_data[j], ord=2) / np.std(train_data[j])  # 标准化欧式距离
            dist5[j, 1] = np.dot(test_data, train_data[j].T) / (np.linalg.norm(test_data) * np.linalg.norm(train_data[j]))
            # 余弦距离
            dist1[j, 0] = train[j, feature_number]
            dist2[j, 0] = train[j, feature_number]
            dist3[j, 0] = train[j, feature_number]
            dist4[j, 0] = train[j, feature_number]
            dist5[j, 0] = train[j, feature_number]
        # print(dist5[: 1])
        order1 = dist1[np.lexsort(dist1.T)]
        order2 = dist2[np.lexsort(dist2.T)]
        order3 = dist3[np.lexsort(dist3.T)]
        order4 = dist4[np.lexsort(dist4.T)]
        order5 = dist5[np.lexsort(dist5.T)]
        # print(i, order5)
        # 按第二列的距离数据对整个distance二维数组升序排序
        for loop in range(k):  # k代表"K聚类"中的"K"是几
            if order5[loop, 0] == 10:
                classifier1 += 1
            elif order5[loop, 0] == 100:
                classifier2 += 1
        if classifier1 >= classifier2:
            result = 10
        if classifier2 >= classifier1:
            result = 100
        # print(result)
        if result == test[feature_number]:
            # print(test[feature_number])
            # print("***", result)
            accuracy += 1
    accuracy = accuracy/208
    if k == 1:
        print("sonar数据集的", "最", "近邻准确率为：", accuracy)
    else:
        print("sonar数据集的", k, "近邻准确率为：", accuracy)


iris = pandas.read_csv('sonar.all-data.csv', header=None, sep=',')
iris1 = iris.iloc[0:208, 0:61]
data1 = np.mat(iris1)
data = data1.A  # 由矩阵化成数组
# print(data)
for Loop in range(10):
    k = Loop + 1
    knn(data)
