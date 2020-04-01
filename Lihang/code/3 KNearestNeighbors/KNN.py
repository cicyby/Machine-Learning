#!/usr/bin/env python
# coding: utf-8

# 原文代码作者：https://github.com/wzyonggege/statistical-learning-method
# 
# 配置环境：python 3.6 代码全部测试通过。
# 
# 整理者（微信公众号）：机器学习之美（ID: BreakIntoAI）


# #  第3章 k近邻法

# #### 距离度量

# In[1]:


import math
from itertools import combinations


# - p = 1 曼哈顿距离
# - p = 2 欧氏距离
# - p = inf  闵式距离minkowski_distance 

# In[2]:


def L(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0


# In[3]:


# 课本例3.1
x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]


# In[4]:


# x1, x2
for i in range(1, 5):
    r = { '1-{}'.format(c):L(x1, c, p=i) for c in [x2, x3]}
    print(min(zip(r.values(), r.keys())))


# python实现，遍历所有数据点，找出n个距离最近的点的分类情况，少数服从多数

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter


# In[6]:


# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# data = np.array(df.iloc[:100, [0, 1, -1]])


# In[7]:


plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


# In[8]:


data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
            
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
                
        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs, key=lambda x:x)[-1]
        return max_count
    
    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


# In[10]:


clf = KNN(X_train, y_train)


# In[11]:


clf.score(X_test, y_test)


# In[12]:


test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))


# In[13]:


plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


# # scikitlearn

# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)


# In[16]:


clf_sk.score(X_test, y_test)


# ### sklearn.neighbors.KNeighborsClassifier
# 
# - n_neighbors: 临近点个数
# - p: 距离度量
# - algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
# - weights: 确定近邻的权重
