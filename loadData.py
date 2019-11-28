import sys
sys.path.append("D:\PROject\ClassifyTask\libsvm-324\python")
import scipy.io as scio
import numpy as np
import os
from svmutil import *


# 数据预处理
dataFile = '作业2_Class_4_data_label.mat'
load_data = scio.loadmat(dataFile)
# print(type(load_data))
# print(load_data.keys())
# print(load_data.values())
print(type(load_data['data']))
print(type(load_data['label']))
# 将数据分组，分为训练组和测试组
# 输出数据的行和列
print(load_data["data"].shape)
print(load_data["label"].shape)
data=load_data["data"]
label=load_data["label"]
# 将两个numpy和到一起
print(np.concatenate((data, label), axis=1)[0])
for a in np.concatenate((data, label), axis=1):
    print(a)

# 暂定每种分类10个数据做训练
n = 10
# 清空文件
def clear():
    if os.path.exists('trainout.txt'):
        os.remove('trainout.txt')
    if os.path.exists('testout.txt'):
        os.remove('testout.txt')
clear()
# 定义一个方法来获取处理所有数据
trainfilename="trainout.txt"
testfilename="testout.txt"
trainfile = open(trainfilename,'a')
testfile = open(testfilename,'a')


# 按照数据行来处理文件
# 先统计各个类别的数据数目
label1=0
label2=0
label3=0
label4=0
for i in range(0,data.shape[0]):
    if (label[i] - 1) < 0.001:
        label1=label1+1
    if (label[i] - 2) < 0.001:
        label2=label2+1
    if (label[i] - 3) < 0.001:
        label3=label3+1
    if (label[i] - 4) < 0.001:
        label4=label4+1

print(label1)
print(label2)
print(label3)
print(label4)
print(label1+label2+label3+label4)
# 第二次循环直接将数据写入文件中
# 定义四个数来计数
la1=0
la2=0
la3=0
la4=0
for i in range(0,data.shape[0]):
    #拼接一行字符串
    line = str(int(label[i]))+" "
    for j in range(1,data.shape[1]+1):
        line = line+" "+str(j)+":"+str(data[i][j-1])
    line = line+"\n"
    #判断标签是否超过计数器，这将影响写入位置
    if (label[i] - 1) < 0.001:
        la1=la1+1
        if(la1<label1*0.8):
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



def pred():
    y,x = svm_read_problem('trainout.txt')
    m = svm_train(y[0:], x[0:], '-c 4')
    a,b = svm_read_problem('testout.txt')
    p_label, p_acc, p_val = svm_predict(a[:], b[:], m)
    print(p_label)
    print(p_acc)
    print(p_val)
pred()