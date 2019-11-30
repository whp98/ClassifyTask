# 本项目完成一个分类任务
给定数据，由于数据维度很高，分类过程中并不需要所有的特征，故可以对特征进行处理，选择对分类最有效的特征。

TODO——list

 -[x] 针对数据实现一个分类算法

 -[x] 前面的结果实现留一法交叉验证算法  

 -[ ] 给出特征选择（降维）后的分类误差（使用更少的有效特征进行分类）

 -[ ] 写作一个实验的报告，主要是报告实验过程

 -[ ] 提供原始代码/可执行的程序

## 数据降维算法
>主要的数据挖掘算法有
>
>>1.主成分分析算法（PCA）
>>
>>2.LDA
>>
>>3.局部线性嵌入 （LLE）
>>
>>4.Laplacian Eigenmaps 拉普拉斯特征映射
## 数据分类算法
>主要的数据分类算法
>
>决策树
>
>神经网络
>
>朴素贝叶斯和贝叶斯信念网络
>
>支持向量机
>
>Rule-based methods
>
>Distance-based methods
>
>Memory based reasoning

# 子任务实现

## 任务实现一：选择并实现一种分类算法应用于对给定的数据；
我们决定使用支持向量机来实现，分类算法,首先支持，首先我们进行数据预处理  
根据libsvm的格式存储要分类的数据，每种类型的数据使用20%作为验证数据集  


## 任务实现二：使用全部特征进行分类计算出留一法交叉验证(love one out cross-valiation)的分类误差，并以此作为参考基值(baseline)；
实现我们的分类算法，并开始准备验证最终验证结果是
```
96.44670050761421

Process finished with exit code 0
96.44670050761421
```
得出结论baseline=

## 任务实现三：设计并实现一种特征选择方法或降维方法，给出特征选择（降维）后的分类误差（即使用更少的有效特征进行分类），结果必须比参考基值(baseline)好。


## 任务实现四：写作一个实验的报告，主要是报告实验过程，参数，计算结果(形式自定，可以给出结果的图表)，并给出小组成员的姓名、学号和班级。


## 任务实现五：提供原始代码/可执行的程序作为参考和作业结果的备份；


## 任务实现六：截止时间11月30日24:00"前,发送最终版本的作业到邮箱地址 76411308@qq.com

