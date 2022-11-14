from dataEEG import Manual
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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

def plot(data_16s, name_save):
    name = ['mean', 'min', 'max', 'median', 'Sample_Entropy', 'cross_zeros']
    data_16s = data_16s.transpose(1,0)
    time = np.arange(16)
    for i in range(1, 7):
        plt.subplot(2,3,i)
        plt.title(name[i-1])
        plt.plot(time, data_16s[i-1], 'o--', color='c', markersize=4)
        plt.plot(time[-1], data_16s[i-1][-1], 'o--', color='r', markersize=4) 
    plt.tight_layout()
    name = 'linear'+ name_save +'.png'
    plt.savefig(name)
    plt.close()

def generate_linear(data, name):
    x = np.arange(15).reshape(-1,1)
    test = np.array([15]).reshape(-1,1)
    cls = LinearRegression(fit_intercept=True, n_jobs=4)
    cls.fit(x, data)
    
    predict = cls.predict(test)
    final_16s = np.concatenate((data, predict),axis=0)
    plot(final_16s, name)
    return predict, cls
if __name__ == '__main__':
    path = '/home/yangry/MachineLearning/feature_extract/data.json'
    data_113, data_114 = data_preprocess(path)
    # data_mean, data_min, data_max, data_median, data_se, data_cz = data_split(data_113)
    data113 = data_split(data_113)
    data114 = data_split(data_114)
    result113, cls113 = generate_linear(data113, '113')
    print('cls113 parameters: ', cls113.coef_, cls113.intercept_)
    print('predict_113', result113)
    result114, cls114 = generate_linear(data114, '114')
    print('cls114_parameters: ', cls114.coef_, cls114.intercept_)
    print('predict_114: ', result114)
    
    
    
    
    