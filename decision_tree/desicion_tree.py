import numpy as np
from scipy import signal
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt #绘图
#from numpy import *
import os
import gc   #gc模块提供一个接口给开发者设置垃圾回收的选项
import time
from tqdm import tqdm
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
#读取文件第一列，保存在s1列表中
###########################################################################################################
start = 114 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
N = 1                                                      #设立变量N，作为循环读取文件的增量
for e in range(start,start+N):                            #循环2次，读取113&114两个文件

    data = open(r'sharp_waves_raw/20151026_%d'% (e)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
    data = data.split( )                                  #以空格为分隔符，返回数值列表data
    data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型
    s1=data
    s1=data[0:45000*4:4]                          #list切片L[n1:n2:n3]  n1代表开始元素下标；n2代表结束元素下标
                                                    #n3代表切片步长，可以不提供，默认值是1，步长值不能为0
    
####################################################################################################################

#滤波
##################################################################################################################
    fs = 150                                           #设立频率变量fs
    lowcut = 1
    highcut = 30
    order = 2                                           #设立滤波器阶次变量
    nyq = 0.5*fs                                        #设立采样频率变量nyq，采样频率=fs/2。
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter(order,[low,high],btype='band') #设计巴特沃斯带通滤波器 “band”
    s1_filter1 = signal.lfilter(b,a,s1)                #将s1带入滤波器，滤波结果保存在s1_filter1中
###################################################################################################################
lenth = int(len(s1_filter1)/fs)
split_data = np.zeros([lenth, fs])
for i in range(lenth):
    split_data[i] = s1_filter1[i*fs:(i+1)*fs]

def get_basics(data): # dimension = 2
    embed_num = len(data.shape)
    basics = {'mean': np.mean(data, embed_num - 1),
            'max': np.max(data, embed_num - 1),
            'min': np.min(data, embed_num - 1),
            'median': np.median(data, embed_num - 1)}
    rms = np.sqrt(basics['mean'] ** 2 + np.std(data) ** 2)
    shape_data = basics['max'] / rms
    basics['shape'] = shape_data
    return basics


def cross_zero(data):
    embed_num = len(data.shape)
    length = data.shape[1]
    dim = data.shape[0]
    flag = np.zeros(dim)
    for i in range(dim):
        tmp_data = data[i]
        for j in range(len(tmp_data) - 1):
            if tmp_data[j] < 0 and tmp_data[j+1] > 0:
                flag[i] += 1
            elif tmp_data[j] > 0 and tmp_data[j+1] < 0:
                flag[i] += 1
            else:
              pass
    return {'cross_zero': flag.astype(int)}




if __name__ == '__main__':
    basics = get_basics(split_data)
    cz = cross_zero(split_data)
    data_113 = {'mean': basics['mean'],
                'min': basics['min'],
                'max': basics['max'],
                'median': basics['median'],
                'get_shape': basics['shape'],
                'cross_zero': cz['cross_zero']}

    np.save('data114_150.npy', data_113)
    data113 = np.load('data113_150.npy', allow_pickle=True).item()
    data114 = np.load('data114_150.npy', allow_pickle=True).item()
    
    data = []
    for i in data114.values():
        data.append(i.tolist())
    
    features = np.array(data)
    features = features / features.max(0)
    

    cla_max = features.sum(0).max()
    cla_min = features.sum(0).min()
    labels = np.zeros(300)
    flag1, flag2, flag3 = 0,0,0
    for i, v in enumerate(features.sum(0)):
        if v < (cla_min + (cla_max - cla_min)/30):
            flag1 += 1
            labels[i] = 1
        elif v >= (cla_min + (cla_max - cla_min)/30) and v <(cla_min + 1 * (cla_max - cla_min)/3):
            labels[i] = 2
            flag2 += 1
        elif v > (cla_min + 1 * (cla_max - cla_min)/3):
            labels[i] = 3
            flag3 += 1
    
    print(flag1, flag2, flag3)
    features = features.transpose(1,0)
    feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.2, random_state=100, stratify=labels)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(feature_train,label_train)
    print(clf.score(feature_test, label_test))
    
    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
    tree.plot_tree(clf,
                filled=True
                )
    plt.savefig("img_tree114.png")