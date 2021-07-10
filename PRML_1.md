# 1. Introduction
 
什么是模式识别?

The field of pattern recognition is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of
these regularities to take actions such as classifying the data into different categories

通过算法自动发现数据中的规律,并将规律用于数据分类

---

经典问题,手写数字识别,理论上可以通过人工设计规则或者根据笔画启发式的区分,但是通常这类方法会产生大量复杂的规则和例外,通常效果较差

模型将从未见过的新数据正确分类的能力称为泛化，generalization

In practical applications, the variability of the input vectors will be such that the training data can comprise only a tiny fraction of all possible input vectors, and so  generalization is a central goal in pattern recognition.

在实际应用中，训练集中的数据只是实际数据的很小一部分，因此泛化能力成为了模式识别中的一个主要目标

---

For most practical applications, the original input variables are typically preprocessed to transform them into some new space of variables where, it is hoped, the pattern recognition problem will be easier to solve.

大部分的应用会对数据集进行预处理，转换到另一个变量空间中

例如，手写数字识别，规范了图片大小，和像素数字表示的范围

在这种处理中测试集应当执行与训练集相同的步骤

---

书中数据预处理包含了构造特征的含义，预处理本身就是在提取特征，这种特征的提取，同时也是在删除无用信息，使数据能够支持更复杂精密的模型

---

关于各种学习任务的分类

正样本有对应标签时，有监督学习

正样本无标签时，更具作用进一步分类

1. 寻找与目标样本相似的样本，聚类 clustering
2. 根据输入确定数据的分布, 密度估计 density estimation
3. 以可视化为目的降维, 可视化 visualization

在这部分中,并没有无监督学习,半监督学习的概念

---

强化学习,通过学习选择动作的策略获得最大的奖励reward

在多个阶段组成的游戏中,只能在整体结束后获得一个reward,其中每一步都做出了贡献,但无法确认其中的某一步是积极作用还是消极作用,这一问题时典型的信用分配问题 credit assignment

强化学习要平衡探索exploration与利用exploitation

---

第一章对支撑各种模型的基础思想做出相对非正式的介绍,这些思想会在后续反复出现,构成一些实际应用中较为复杂的机器学习模型.

同时第一张介绍了三个贯穿全书的工具,概率论,决策理论,信息论