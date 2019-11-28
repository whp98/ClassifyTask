import numpy
import pandas as pd
f1 = open(r"D:\Develop\Documents\Section3SVMSVR\T3-5 用户贷款风险预测\个人征信\dataset\user_info_train.txt")
f2 = open(r"D:\Develop\Documents\Section3SVMSVR\T3-5 用户贷款风险预测\个人征信\dataset\overdue_train.txt")
f3 = open(r"D:\Develop\Documents\Section3SVMSVR\T3-5 用户贷款风险预测\Billing_feature.csv")
f4 = open("trainl.txt","a")
f5 = open("testl.txt","a")
f6 = open("all.txt","a")
df1 = pd.read_csv(f1,header=None)
df2 = pd.read_csv(f2,header=None)
df3 = pd.read_csv(f3,header=0)
f1.close
f2.close
f3.close
#print(df)
df1 = df1.sort_values(by=[0])
#接下来进行数据清洗
#将我们不想要的数据清洗掉
#数据量
n = 5000
t1 = 0
t2 = 0
for i in range(0,df1.shape[0]):
    if(t1<2000 and int(df2.iloc[i][1])==0):
        t1 = t1 + 1
        print(str(t1)+"at 1")
        if((df3.iloc[i][0]+df3.iloc[i][1]+df3.iloc[i][3]+df3.iloc[i][4]+df3.iloc[i][5])>0.001):
            f4.write(str(df2.iloc[i][1])+' ')
            f6.write(str(df2.iloc[i][1])+' ')
            for i1 in range(1,12):
                if i1<=5:
                    f4.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                    f6.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                else:
                    #i3 = i3 + df3.iloc[i][i1-6]
                    f4.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
                    f6.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
            #f4.write(' '+str(6)+':'+str(i3/5))
            f4.write("\n")
            f6.write("\n")
    if(t2 < 2000 and int(df2.iloc[i][1])==1):
        t2 = t2 + 1
        print(str(t2)+"at 2")
        if((df3.iloc[i][0]+df3.iloc[i][1]+df3.iloc[i][3]+df3.iloc[i][4]+df3.iloc[i][5])>0.001):
            f4.write(str(df2.iloc[i][1])+' ')
            f6.write(str(df2.iloc[i][1])+' ')
            for i1 in range(1,12):
                if i1<=5:
                    f4.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                    f6.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                else:
                    #i3 = i3 + df3.iloc[i][i1-6]
                    f4.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
                    f6.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
            #f4.write(' '+str(6)+':'+str(i3/5))
            f4.write("\n")
            f6.write("\n")
    if(t1>=2000 and t1<2500 and int(df2.iloc[i][1])==0):
        t1 = t1 + 1
        print(str(t1)+"at 3")
        if((df3.iloc[i][0]+df3.iloc[i][1]+df3.iloc[i][3]+df3.iloc[i][4]+df3.iloc[i][5])>0.001):
            f5.write(str(df2.iloc[i][1])+' ')
            f6.write(str(df2.iloc[i][1])+' ')
            for i1 in range(1,12):
                if i1<=5:
                    f5.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                    f6.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                else:
                    #i3 = i3 + df3.iloc[i][i1-6]
                    f5.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
                    f6.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
            #f5.write(' '+str(6)+':'+str(i3/5))
            f5.write("\n")
            f6.write("\n")
    if(t2>=2000 and t2<2500 and int(df2.iloc[i][1])==1):
        t2 = t2+1
        print(str(t2)+"at 4")
        if((df3.iloc[i][0]+df3.iloc[i][1]+df3.iloc[i][3]+df3.iloc[i][4]+df3.iloc[i][5])>0.001):
            f5.write(str(df2.iloc[i][1])+' ')
            f6.write(str(df2.iloc[i][1])+' ')
            for i1 in range(1,12):
                if i1<=5:
                    f5.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                    f6.write(' '+str(i1)+':'+str(df1.iloc[i][i1]))
                else:
                    #i3 = i3 + df3.iloc[i][i1-6]
                    f5.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
                    f6.write(' '+str(i1)+':'+str(df3.iloc[i][i1-6]))
            #f5.write(' '+str(6)+':'+str(i3/5))
            f5.write("\n")
            f6.write("\n")
print(df2.shape)
f4.close
f5.close
f6.close
