# -- coding: utf-8 --

import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import tqdm

path = '/home/yangry/MachineLearning/feature_extract/data.json'

class Manual():
    def __init__(self, feature_path):
        
        super(Manual, self).__init__()
        self.path = feature_path
        with open(self.path,"r") as data:
            self.data = json.load(data)
            print(self.path, 'data_name:', self.data.keys())
    def forward(self): 
        return self.data

class EEGdata(Dataset):
    '''
        dataset of EEG signal
    '''
    def __init__(self, data_path, file):
        super(EEGdata, self).__init__()
        start = 113 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
        N = 2                                                 #设立变量N，作为循环读取文件的增量
        self.results = []
        for e in range(start,start+N):                        #循环2次，读取113&114两个文件
            dp = data_path + '/20151026_%d'% (e)
            data = open(r(dp)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
            data = data.split( )                                  #以空格为分隔符，返回数值列表data
            data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型
            s1= data
            s1 = data[0:45000*4:4]                          #list切片L[n1:n2:n3]  n1代表开始元素下标；n2代表结束元素下标
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
            s1_filter1 = signal.lfilter(b,a,s1)                 #将s1带入滤波器，滤波结果保存在s1_filter1中
            results.append(s1_filter1)
    
        
        