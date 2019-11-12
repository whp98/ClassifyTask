import scipy.io as scio

dataFile = '作业2_Class_4_data_label.mat'
data = scio.loadmat(dataFile)
print(data)