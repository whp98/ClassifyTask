import numpy as np
import pandas as pd
import sys
import os

sys.path.append("libsvm-324\python")
import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA
from svmutil import *

## 降维的维度
weidu = 25
## 中间文件的名称
tempfilename = "all.txt"
##是否使用pca降维
usepca = True


# 读取数据
train = pd.read_csv('data/pfm_train.csv')
test = pd.read_csv('data/pfm_test.csv')
print('train size:{}'.format(train.shape))  # train size:(1100, 31)
print('test size:{}'.format(test.shape))  #test size:(350, 30)
print(train)
# 查看数据集中是否含有缺失值：无缺失值
train.isnull().mean()
# EmployeeNumber为员工ID，将其删除
train.drop(['EmployeeNumber'], axis = 1, inplace = True)
test.drop(['EmployeeNumber'],axis=1,inplace=True)
print(train)
# label
Attrition = train['Attrition']
data_label = Attrition.values
train.drop(['Attrition'], axis = 1, inplace = True)
#train.insert(0, 'Attrition', Attrition)
source = train
train = pd.get_dummies(train)
data_train = train.values
print(data_train.shape)
print(data_label.shape)
data = data_train
label = data_label

#数据方差滤波
def lvbofangcha(nparray):
    fangcha = np.var(nparray,axis=0)
    print(len(fangcha))
    de=[]
    for i in range(0,nparray.shape[1]):
        if(fangcha[i]<0.25):
            de.append(i)
    print(len(de))
    print(len(fangcha)-len(de))
    return np.delete(nparray,de,axis=1)
#data=lvbofangcha(data)

# 归一化
def guiyihuanp(nparray):
    x_normed = nparray / nparray.max(axis=0)
    return x_normed


#data = guiyihuanp(data)

pca = PCA(n_components=weidu)

if (usepca == True):
    data = pca.fit_transform(data)


# 清空数据文件
def removefile(filename):
    if os.path.exists(filename):
        os.remove(filename)


# 获取全部数据svm格式文件
def getallfile():
    removefile(tempfilename)
    removefile(tempfilename + "_scale")
    file = open(tempfilename, 'a')
    for i in range(0, data.shape[0]):
        # 拼接一行字符串
        line = str(int(label[i])) + " "
        for j in range(1, data.shape[1] + 1):
            line = line + " " + str(j) + ":" + str(data[i][j - 1])
        line = line + "\n"
        file.write(line)
    file.close()
    os.system("libsvm-324\\windows\\svm-scale.exe " + tempfilename + " > " + tempfilename + "_scale")


def liuyiyanzhengNew():
    vals = []
    a, b = svm_read_problem(tempfilename + "_scale")
    for i in range(0, len(a)):
        if (i == 0):
            m = svm_train(a[1:], b[1:], '-c 4')
            e = [a[i]]
            r = [b[i]]
            p_label, p_acc, p_val = svm_predict(e, r, m)
        elif (i == len(a) - 1):
            e = [a[len(a) - 1]]
            r = [b[len(a) - 1]]
            m = svm_train(a[0:len(a) - 2], b[0:len(a) - 2], '-c 4')
            p_label, p_acc, p_val = svm_predict(e, r, m)
        else:
            e = [a[i]]
            r = [b[i]]
            m = svm_train(a[0:i - 1] + a[i + 1:], b[0:i - 1] + b[i + 1:], '-c 4')
            p_label, p_acc, p_val = svm_predict(e, r, m)
        vals.append(p_acc[0])
    sum = 0
    for b in vals:
        sum = sum + b
    return sum / (len(vals))

getallfile()
print("误差值：",liuyiyanzhengNew())