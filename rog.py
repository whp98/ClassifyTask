from svmutil import *
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
label = []
#获取测试集的label
def getLabel(filename1):
    f1 = open(filename1,'r')
    a = f1.readlines()
    for line in a:
        b = line.split(' ')
        label.append(int(b[0]))

#在确定好阈值的时候获取新的label
def getNewLabel(p_val,val):
    NewLabel = []
    for i in p_val:
        if i[0] > val:
            NewLabel.append(1)
        else:
            NewLabel.append(0)
    return NewLabel

#获取ROC曲线的绘制，和点的获取
def getPoints(p_val,label):
    x = [0,1]
    y = [0,1]
    t = [-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4]
    for i in range(0,10):
        newlabel = getNewLabel(p_val,t[i])
        res = getTPFPTNFN(label,newlabel)
        x.append((res[0]/(res[0]+res[3])))
        y.append(res[1]/(res[1]+res[2]))
    plt.plot(x, y)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()
    return x,y
#预测
def pred():
    y,x = svm_read_problem('train1.txt')
    m = svm_train(y[0:], x[0:], '-c 0.3 -m 3000 -h 0')
    a,b = svm_read_problem('test1.txt')
    p_label, p_acc, p_val = svm_predict(a[:], b[:], m)
    print(p_acc)
    fpr,tpr,thresholds = metrics.roc_curve(label,p_val)
    getPoints(p_val,label)

#混淆矩阵
def getTPFPTNFN(label,p_label):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(0,len(label)):
        a = label[i]
        b = p_label[i]
        if a ==0 and b==0:
            TN = TN+1
        if a==0 and b==1:
            FP = FP + 1
        if a==1 and b==0:
            FN = FN + 1
        if a==1 and b==1:
            TP = TP + 1
    print(" \t \t1\t0\t合计")
    print("实际\t1\tTP="+str(TP)+"\tFN="+str(FN)+"\tTP+FN="+str(TP+FN))
    print("实际\t0\tFP="+str(FP)+"\tTN="+str(TN)+"\tTP+TN="+str(TP+TN))
    print("合计\t \tTP+FP="+str(TP+FP)+"\tTN+FN="+str(TN+FN)+"\tALL="+str(TN+TP+FN+FP))
    print("灵敏度 "+str(TP/(TP+FN))+" 特异度 "+str(TN/(TN+FP))+" 约登指数"+str((TP/(TP+FN))+(TN/(TN+FP))-1)+"\n")
    return TP,FP,TN,FN

getLabel('test1.txt')
pred()
