import scipy.io as scio
import numpy as np

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

# 定义一个方法来获取处理所有数据

