#!/usr/bin/env python
# coding: utf-8

# 原文代码作者：https://github.com/wzyonggege/statistical-learning-method
# 
# 配置环境：python 3.6 代码全部测试通过。
# 
# 整理者（微信公众号）：机器学习之美（ID: BreakIntoAI）


# # 第4章 朴素贝叶斯

# 基于贝叶斯定理与特征条件独立假设的分类方法。
# 
# 模型：
# 
# - 高斯模型
# - 多项式模型
# - 伯努利模型

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math


# In[2]:


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]


# In[3]:


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[4]:


X_test[0], y_test[0]


# 参考：https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# 
# ## GaussianNB 高斯朴素贝叶斯
# 
# 特征的可能性被假设为高斯
# 
# 概率密度函数：
# $$P(x_i | y_k)=\frac{1}{\sqrt{2\pi\sigma^2_{yk}}}exp(-\frac{(x_i-\mu_{yk})^2}{2\sigma^2_{yk}})$$
# 
# 数学期望(mean)：$\mu$，方差：$\sigma^2=\frac{\sum(X-\mu)^2}{N}$

# In[5]:


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))


# In[6]:


model = NaiveBayes()


# In[7]:


model.fit(X_train, y_train)


# In[8]:


print(model.predict([4.4,  3.2,  1.3,  0.2]))


# In[9]:


model.score(X_test, y_test)


# scikit-learn实例
# 
# # sklearn.naive_bayes

# In[10]:


from sklearn.naive_bayes import GaussianNB


# In[11]:


clf = GaussianNB()
clf.fit(X_train, y_train)


# In[12]:


clf.score(X_test, y_test)


# In[14]:


clf.predict([[4.4,  3.2,  1.3,  0.2]])


# In[15]:


from sklearn.naive_bayes import BernoulliNB, MultinomialNB # 伯努利模型和多项式模型

