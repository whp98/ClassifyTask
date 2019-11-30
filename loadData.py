import sys
sys.path.append("libsvm-324\python")
import scipy.io as scio
import numpy as np
import os
from svmutil import *
from sklearn.decomposition import PCA
pca=PCA(n_components=6)

# 数据预处理
dataFile = '作业2_Class_4_data_label.mat'
load_data = scio.loadmat(dataFile)
# print(type(load_data))
# print(load_data.keys())
# print(load_data.values())
# print(type(load_data['data']))
# print(type(load_data['label']))
# 将数据分组，分为训练组和测试组
# 输出数据的行和列
# print(load_data["data"].shape)
# print(load_data["label"].shape)
data=load_data["data"]
label=load_data["label"]
# 将两个numpy和到一起
# print(np.concatenate((data, label), axis=1)[0])
# for a in np.concatenate((data, label), axis=1):
#     print(a)
#归一化
def guiyihuanp(nparray):
    x_normed = nparray / nparray.max(axis=0)
    return x_normed
data=guiyihuanp(data)
#print(data)

#数据方法滤波
def lvbofangcha(nparray):
    fangcha = np.var(nparray,axis=0)
    print(len(fangcha))
    de=[]
    for i in range(0,nparray.shape[1]):
        if(fangcha[i]<0.02):
            de.append(i)
    print(len(de))
    return np.delete(nparray,de,axis=1)

# def pca(dataMat, topNfeat=1000):
#
#     # 1.对所有样本进行中心化（所有样本属性减去属性的平均值）
#     meanVals = np.mean(dataMat, axis=0)
#     meanRemoved = dataMat - meanVals
#
#     # 2.计算样本的协方差矩阵 XXT
#     covmat = np.cov(meanRemoved, rowvar=0)
#     print(covmat)
#
#     # 3.对协方差矩阵做特征值分解，求得其特征值和特征向量，并将特征值从大到小排序，筛选出前topNfeat个
#     eigVals, eigVects = np.linalg.eig(np.mat(covmat))
#     eigValInd = np.argsort(eigVals)
#     eigValInd = eigValInd[:-(topNfeat+1):-1]    # 取前topNfeat大的特征值的索引
#     redEigVects = eigVects[:, eigValInd]        # 取前topNfeat大的特征值所对应的特征向量
#
#     # 4.将数据转换到新的低维空间中
#     lowDDataMat = meanRemoved * redEigVects     # 降维之后的数据
#     reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构数据，可在原数据维度下进行对比查看
#     return np.array(lowDDataMat), np.array(reconMat)

# data,sss=pca(data,1000)
# print(data.shape)
# print(sss.shape)
data=pca.fit_transform(data)
#数据文件名
trainfilename="trainout.txt"
testfilename="testout.txt"


# 清空文件
def clear(a,b):
    if os.path.exists(a):
        os.remove(a)
    if os.path.exists(b):
        os.remove(b)

#一键清空
def easyclear():
    clear(trainfilename, testfilename)



def getlabelnum():
    # 按照数据行来处理文件
    # 先统计各个类别的数据数目
    label1 = 0
    label2 = 0
    label3 = 0
    label4 = 0
    for i in range(0, data.shape[0]):
        if (label[i] - 1) < 0.001:
            label1 = label1 + 1
        if (label[i] - 2) < 0.001:
            label2 = label2 + 1
        if (label[i] - 3) < 0.001:
            label3 = label3 + 1
        if (label[i] - 4) < 0.001:
            label4 = label4 + 1

    return label1,label2,label3,label4
# 第二次循环直接将数据写入文件中
# 定义四个数来计数
def oldgenFile():
    la1 = 0
    la2 = 0
    la3 = 0
    la4 = 0
    label1, label2, label3, label4 = getlabelnum()
    easyclear()
    trainfile = open(trainfilename, 'a')
    testfile = open(testfilename, 'a')
    for i in range(0, data.shape[0]):
        # 拼接一行字符串
        line = str(int(label[i])) + " "
        for j in range(1, data.shape[1] + 1):
            line = line + " " + str(j) + ":" + str(data[i][j - 1])
        line = line + "\n"
        # 判断标签是否超过计数器，这将影响写入位置
        if (label[i] - 1) < 0.001:
            la1 = la1 + 1
            if (la1 < label1 * 0.8):
                trainfile.write(line)
            else:
                testfile.write(line)
        if (label[i] - 2) < 0.001:
            la2 = la2 + 1
            if (la2 < label2 * 0.8):
                trainfile.write(line)
            else:
                testfile.write(line)
        if (label[i] - 3) < 0.001:
            la3 = la3 + 1
            if (la3 < label3 * 0.8):
                trainfile.write(line)
            else:
                testfile.write(line)
        if (label[i] - 4) < 0.001:
            la4 = la4 + 1
            if (la4 < label4 * 0.8):
                trainfile.write(line)
            else:
                testfile.write(line)
    trainfile.close()
    testfile.close()
#将所有数据存储到一个文件中
def getallfile():
    if os.path.exists("all.txt"):
        os.remove("all.txt")
    file = open("all.txt",'a')
    for i in range(0, data.shape[0]):
        # 拼接一行字符串
        line = str(int(label[i])) + " "
        for j in range(1, data.shape[1] + 1):
            line = line + " " + str(j) + ":" + str(data[i][j - 1])
        line = line + "\n"
        file.write(line)
    file.close()
    os.system("libsvm-324\\windows\\svm-scale.exe " + "all.txt" + " > " + "all.txt" + "_scale")

# 留一法交叉验证不归一
def liuyiyanzheng():
    # 用于存储每次迭代训练序列的值
    vals = []
    for a in range(0, data.shape[0]):
        easyclear()
        trainfile = open(trainfilename, 'a')
        testfile = open(testfilename, 'a')
        for i in range(0,data.shape[0]):
            # 每一行的数据
            line = str(int(label[i])) + " "
            for j in range(1, data.shape[1] + 1):
                line = line + " " + str(j) + ":" + str(data[i][j - 1])
            line = line + "\n"
            if (i!=a):
                trainfile.write(line)
            else:
                testfile.write(line)
        trainfile.close()
        testfile.close()
        vals.append(pred())
    sum = 0
    for b in vals:
        sum = sum + b
    return sum/(len(vals))
# 归一的留一法交叉验证
def liuyiyanzheng1():
    # 用于存储每次迭代训练序列的值
    vals = []
    for a in range(0, data.shape[0]):
        easyclear()
        trainfile = open(trainfilename, 'a')
        testfile = open(testfilename, 'a')
        for i in range(0,data.shape[0]):
            # 每一行的数据
            line = str(int(label[i])) + " "
            for j in range(1, data.shape[1] + 1):
                line = line + " " + str(j) + ":" + str(data[i][j - 1])
            line = line + "\n"
            if (i!=a):
                trainfile.write(line)
            else:
                testfile.write(line)
        trainfile.close()
        testfile.close()
        vals.append(scale_pred())
    sum = 0
    for b in vals:
        sum = sum + b
    return sum/(len(vals))


def pred():
    y,x = svm_read_problem(trainfilename)
    m = svm_train(y[0:], x[0:], '-c 4')
    a,b = svm_read_problem(testfilename)
    p_label, p_acc, p_val = svm_predict(a[:], b[:], m)
    print(p_label)
    print(p_acc)
    print(p_val)
    return p_acc[0]

# 归一化的方法
def scale_pred():
    file1,file2 = guiyihua(trainfilename,testfilename)
    y, x = svm_read_problem(file1)
    m = svm_train(y[0:], x[0:], '-c 4')
    a, b = svm_read_problem(file2)
    p_label, p_acc, p_val = svm_predict(a[:], b[:], m)
    print(p_label)
    print(p_acc)
    print(p_val)
    return p_acc[0]

# pred()
# 进行数据归一化
# 返回文件名字
def guiyihua(train_filename,test_filename):
    #清除之前的归一化数据
    clear(train_filename+"_scale",test_filename+"_scale")
    os.system("libsvm-324\\windows\\svm-scale.exe "+train_filename+" > "+train_filename+"_scale")
    os.system("libsvm-324\\windows\\svm-scale.exe "+test_filename+" > "+test_filename+"_scale")
    return train_filename+"_scale",test_filename+"_scale"


def liuyiyanzhengNew():
    vals = []
    a, b = svm_read_problem("all.txt_scale")
    for i in range(0,len(a)):
        if(i==0):
            m = svm_train(a[1:], b[1:], '-c 4')
            e=[a[i]]
            r=[b[i]]
            p_label, p_acc, p_val = svm_predict(e, r, m)
        elif(i==len(a)-1):
            e=[a[len(a)-1]]
            r=[b[len(a)-1]]
            m = svm_train(a[0:len(a)-2], b[0:len(a)-2], '-c 4')
            p_label, p_acc, p_val = svm_predict(e, r, m)
        else:
            e = [a[i]]
            r = [b[i]]
            m = svm_train(a[0:i-1]+a[i+1:], b[0:i-1]+b[i+1:], '-c 4')
            p_label, p_acc, p_val = svm_predict(e, r, m)
        vals.append(p_acc[0])
    sum = 0
    for b in vals:
        sum = sum + b
    return sum / (len(vals))
getallfile()
print(liuyiyanzhengNew())
# a, b = svm_read_problem("all.txt_scale")
# print(b)