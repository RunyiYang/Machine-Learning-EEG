# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################

"""
Created on Fri Jan 06 10:08:42 2017

@author: Yuyangyou

代码功能描述：（1）读取Sharp_waves文件，
              （2）采用巴特沃斯滤波器，进行60-240Hz滤波
              （3）画图
              （4）....

"""
#####################################################################

import numpy as np
from scipy import signal
import math
import matplotlib
import matplotlib.pylab as plt #绘图
#from numpy import *
import os
import gc   #gc模块提供一个接口给开发者设置垃圾回收的选项
import time
from tqdm import tqdm
import pywt
#读取文件第一列，保存在s1列表中
###########################################################################################################
start = 114 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
N = 1                                                      #设立变量N，作为循环读取文件的增量
for e in range(start,start+N):                            #循环2次，读取113&114两个文件

    data = open(r'20151026_%d'% (e)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
    data = data.split( )                                  #以空格为分隔符，返回数值列表data
    data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型
    s1=data
    s1=data[0:45000*4:4]                          #list切片L[n1:n2:n3]  n1代表开始元素下标；n2代表结束元素下标
                                                    #n3代表切片步长，可以不提供，默认值是1，步长值不能为0
    
####################################################################################################################

#滤波
##################################################################################################################
    fs = 3000                                           #设立频率变量fs
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
    return {'mean': np.mean(data, embed_num - 1),
            'max': np.max(data, embed_num - 1),
            'min': np.min(data, embed_num - 1),
            'median': np.median(data, embed_num - 1)}

def get_sample_entropy(data, fs=15):
    def Sample_Entropy(x, m, r=0.15):
        """
        样本熵
        m 滑动时窗的长度
        r 阈值系数 取值范围一般为：0.1~0.25
        """
        # 将x转化为数组
        x = np.array(x)
        # 检查x是否为一维数据
        if x.ndim != 1:
            raise ValueError("x的维度不是一维")
        # 计算x的行数是否小于m+1
        if len(x) < m+1:
            raise ValueError("len(x)小于m+1")
        # 将x以m为窗口进行划分
        entropy = 0  # 近似熵
        for temp in range(2):
            X = []
            for i in range(len(x)-m+1-temp):
                X.append(x[i:i+m+temp])
            X = np.array(X)
            # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
            D_value = []  # 存储差值
            for index1, i in enumerate(tqdm(X)):
                sub = []
                for index2, j in enumerate(X):
                    if index1 != index2:
                        sub.append(max(np.abs(i-j)))
                D_value.append(sub)
            # 计算阈值
            F = r*np.std(x, ddof=1)
            print("计算阈值")
            # 判断D_value中的每一行中的值比阈值小的个数除以len(x)-m+1的比例
            num = np.sum(D_value<F, axis=1)/(len(X)-m+1-temp)
            print("判断比例")
            # 计算num的对数平均值
            Lm = np.average(np.log(num))
            entropy = abs(entropy) - Lm
        return entropy
    sample_entropy = []
    for x in data:
        entropy = Sample_Entropy(x, 2)
        sample_entropy.append(entropy)
    return {'Sample_Entropy' : np.array(sample_entropy)}

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


# print(get_basics(split_data), get_sample_entropy(split_data), cross_zero(split_data))




#画图
###################################################################################################################
def figure(data):
    fig1 = plt.figure()                             #创建画图对象，开始画图
    ax1 = fig1.add_subplot(211)
                    #在一张figure里面生成多张子图，将画布分割成2行1列， 图像画在从左到右从上到下的第1块
                    #例如，fig1.add_subplot(349)  将画布分割成3行4列，图像画在从左到右从上到下的第9块

    plt.plot(s1,color='r')                          #在选定的画布位置上，画未经滤波的s1图像，设定颜色为红色
    ax1.set_title('Denoised Signal')               #设定子图211的title为denoised signal
    plt.ylabel('Amplitude')                         #设定子图211的Y轴lable为amplitude

    ax2 = fig1.add_subplot(212)
                    # 在一张figure里面生成多张子图，将画布分割成2行1列， 图像画在从左到右从上到下的第2块

    plt.plot(s1_filter1,color='r')                  #在选定的画布位置上，画经过滤波的s1_filter1图像，设定颜色为红色
    ax2.set_title('Denoised Signal')               #设定子图212的title为denoised signal
    plt.ylabel('Amplitude')                         #设定子图212的Y轴lable为amplitude
    plt.savefig(r'c:/data/20151026_%d.png' % (e))  #保存图像，设定保存路径并统一命名，%d处，十进制替换为e值
    plt.close('all')                                 #关闭绘图对象，释放绘图资源
##################################################################################################################







































































