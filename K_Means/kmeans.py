from dataEEG import Manual
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def data_preprocess(data_path):
    data = Manual(data_path).forward()
    data_113 = data['113']
    data_114 = data['114']
    return data_113, data_114

def data_split(data):
    data = np.array([data['mean'], data['min'], data['max'], data['median'], data['Sample_Entropy'], data['cross_zero']])
    for i, value in enumerate(data[4]):
        if value == 'inf':
            data[4][i] = data[4][i-1]
    data = data.transpose(1,0).astype(float)
    return data

def generate_knn(features):
    features = features.transpose(1,0)
    cla_max = features.sum(0).max()
    cla_min = features.sum(0).min()
    labels = np.zeros(15)
    flag1, flag2, flag3 = 0,0,0
    for i, v in enumerate(features.sum(0)):
        if v < (cla_min + (cla_max - cla_min)/3):
            flag1 += 1
            labels[i] = 1
        elif v >= (cla_min + (cla_max - cla_min)/3) and v <(cla_min + 2 * (cla_max - cla_min)/3):
            labels[i] = 2
            flag2 += 1
        elif v > (cla_min + 2 * (cla_max - cla_min)/3):
            labels[i] = 3
            flag3 += 1
    
    print(flag1, flag2, flag3)
    features = features.transpose(1,0)
    labels = labels.reshape(-1,)
    # features = features.reshape(15, -1)
    train_features = features[0:10]
    train_labels = labels[0:10]
    test_features = features[10:]
    test_labels = labels[10:]

    knn.fit(train_features, train_labels)
    predict = knn.predict(test_features)
    print('predict:', predict,'gt:', test_labels)

if __name__ == '__main__':
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', leaf_size=10, p=2, metric='minkowski', metric_params=None, n_jobs=4)
    # iris = datasets.load_iris()
    # iris_x = iris.data
    # iris_y = iris.target
    # x_train, x_test , y_train, y_test = train_test_split(iris_x, iris_y, test_size = 0.3)
    # print(x_train.shape, y_train.shape)
    # print(x_train)
    # print(y_train)
    # knn.fit(x_train, y_train)      #放入训练数据进行训练
    # print(knn.predict(x_test))           #打印预测内容
    # print(y_test)     #实际标签

    path = '/home/yangry/MachineLearning/feature_extract/data.json'
    data_113, data_114 = data_preprocess(path)
    # data_mean, data_min, data_max, data_median, data_se, data_cz = data_split(data_113)
    data113 = data_split(data_113)
    data114 = data_split(data_114)
    features113 = data113 / data113.max(0)
    features114 = data114 / data114.max(0)
    for n in range(1, 10):
        kmeans113 = KMeans(n_clusters=n, random_state=666)
        kmeans114 = KMeans(n_clusters=n, random_state=666)
        y_pred113 = kmeans113.fit_predict(features113)
        y_pred114 = kmeans114.fit_predict(features114)
        # print("113, n_clusters: {}\tinertia: {}".format(n, kmeans113.inertia_), y_pred113)
        # print("114, n_clusters: {}\tinertia: {}".format(n, kmeans114.inertia_), y_pred114)
        print(kmeans114.inertia_)


    
    
    