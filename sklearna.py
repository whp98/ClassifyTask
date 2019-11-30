import sys
sys.path.append("libsvm-324\python")
import scipy.io as scio
import numpy as np
import os
from sklearn.decomposition import PCA
pca=PCA(n_components=3)

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
label=np.ravel(label)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
print(type(data),type(label))
#拟合数据

def guiyihuanp(nparray):
    x_normed = nparray / nparray.max(axis=0)
    return x_normed
data=guiyihuanp(data)
data=pca.fit_transform(data)
# #数据方法滤波
# def lvbofangcha(nparray):
#     fangcha = np.var(nparray,axis=0)
#     print(len(fangcha))
#     de=[]
#     for i in range(0,nparray.shape[1]):
#         if(fangcha[i]<0.02):
#             de.append(i)
#     print(len(de))
#     return np.delete(nparray,de,axis=1)
# data=lvbofangcha(data)

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
# data,aaa=pca(data)
print(data.shape)
def liuyiyanzhengNew():
    vals = []
    for i in range(0,len(label)):
        clf = GaussianNB()
        #clf=M
        clf.fit(np.delete(data,i,axis=0), np.delete(label,i,axis=0))
        #print(np.delete(data,i,axis=1).shape)
        #print(type(np.delete(data,i)))
        #clf.fit(data, label,)
        pre=clf.predict([data[i]])
        if(pre==label[i]):
            vals.append(1)
        else:
            vals.append(0)
    sum = 0
    for b in vals:
        sum = sum + b
    print(vals)
    return sum / (len(vals))

print(liuyiyanzhengNew())