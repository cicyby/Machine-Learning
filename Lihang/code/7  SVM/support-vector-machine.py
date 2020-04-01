#!/usr/bin/env python
# coding: utf-8

# 原文代码作者：https://github.com/wzyonggege/statistical-learning-method
# 
# 配置环境：python 3.6 代码全部测试通过。
# 
# 整理者（微信公众号）：机器学习之美（ID: BreakIntoAI）
# 
# ![wechat](../qrcode_for_BreakIntoAI.jpg)

# # 第7章 支持向量机

# ----
# 分离超平面：$w^Tx+b=0$
# 
# 点到直线距离：$r=\frac{|w^Tx+b|}{||w||_2}$
# 
# $||w||_2$为2-范数：$||w||_2=\sqrt[2]{\sum^m_{i=1}w_i^2}$
# 
# 直线为超平面，样本可表示为：
# 
# $w^Tx+b\ \geq+1$
# 
# $w^Tx+b\ \leq+1$
# 
# #### margin：
# 
# **函数间隔**：$label(w^Tx+b)\ or\ y_i(w^Tx+b)$
# 
# **几何间隔**：$r=\frac{label(w^Tx+b)}{||w||_2}$，当数据被正确分类时，几何间隔就是点到超平面的距离
# 
# 为了求几何间隔最大，SVM基本问题可以转化为求解:($\frac{r^*}{||w||}$为几何间隔，(${r^*}$为函数间隔)
# 
# $$\max\ \frac{r^*}{||w||}$$
# 
# $$(subject\ to)\ y_i({w^T}x_i+{b})\geq {r^*},\ i=1,2,..,m$$
# 
# 分类点几何间隔最大，同时被正确分类。但这个方程并非凸函数求解，所以要先①将方程转化为凸函数，②用拉格朗日乘子法和KKT条件求解对偶问题。
# 
# ①转化为凸函数：
# 
# 先令${r^*}=1$，方便计算（参照衡量，不影响评价结果）
# 
# $$\max\ \frac{1}{||w||}$$
# 
# $$s.t.\ y_i({w^T}x_i+{b})\geq {1},\ i=1,2,..,m$$
# 
# 再将$\max\ \frac{1}{||w||}$转化成$\min\ \frac{1}{2}||w||^2$求解凸函数，1/2是为了求导之后方便计算。
# 
# $$\min\ \frac{1}{2}||w||^2$$
# 
# $$s.t.\ y_i(w^Tx_i+b)\geq 1,\ i=1,2,..,m$$
# 
# ②用拉格朗日乘子法和KKT条件求解最优值：
# 
# $$\min\ \frac{1}{2}||w||^2$$
# 
# $$s.t.\ -y_i(w^Tx_i+b)+1\leq 0,\ i=1,2,..,m$$
# 
# 整合成：
# 
# $$L(w, b, \alpha) = \frac{1}{2}||w||^2+\sum^m_{i=1}\alpha_i(-y_i(w^Tx_i+b)+1)$$
# 
# 推导：$\min\ f(x)=\min \max\ L(w, b, \alpha)\geq \max \min\ L(w, b, \alpha)$
# 
# 根据KKT条件：
# 
# $$\frac{\partial }{\partial w}L(w, b, \alpha)=w-\sum\alpha_iy_ix_i=0,\ w=\sum\alpha_iy_ix_i$$
# 
# $$\frac{\partial }{\partial b}L(w, b, \alpha)=\sum\alpha_iy_i=0$$
# 
# 带入$ L(w, b, \alpha)$
# 
# $\min\  L(w, b, \alpha)=\frac{1}{2}||w||^2+\sum^m_{i=1}\alpha_i(-y_i(w^Tx_i+b)+1)$
# 
# $\qquad\qquad\qquad=\frac{1}{2}w^Tw-\sum^m_{i=1}\alpha_iy_iw^Tx_i-b\sum^m_{i=1}\alpha_iy_i+\sum^m_{i=1}\alpha_i$
# 
# $\qquad\qquad\qquad=\frac{1}{2}w^T\sum\alpha_iy_ix_i-\sum^m_{i=1}\alpha_iy_iw^Tx_i+\sum^m_{i=1}\alpha_i$
# 
# $\qquad\qquad\qquad=\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\alpha_iy_iw^Tx_i$
# 
# $\qquad\qquad\qquad=\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)$
# 
# 再把max问题转成min问题：
# 
# $\max\ \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)=\min \frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum^m_{i=1}\alpha_i$
# 
# $s.t.\ \sum^m_{i=1}\alpha_iy_i=0,$
# 
# $ \alpha_i \geq 0,i=1,2,...,m$
# 
# 以上为SVM对偶问题的对偶形式
# 
# -----
# #### kernel
# 
# 在低维空间计算获得高维空间的计算结果，也就是说计算结果满足高维（满足高维，才能说明高维下线性可分）。
# 
# #### soft margin & slack variable
# 
# 引入松弛变量$\xi\geq0$，对应数据点允许偏离的functional margin 的量。
# 
# 目标函数：$\min\ \frac{1}{2}||w||^2+C\sum\xi_i\qquad s.t.\ y_i(w^Tx_i+b)\geq1-\xi_i$ 
# 
# 对偶问题：
# 
# $$\max\ \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)=\min \frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum^m_{i=1}\alpha_i$$
# 
# $$s.t.\ C\geq\alpha_i \geq 0,i=1,2,...,m\quad \sum^m_{i=1}\alpha_iy_i=0,$$
# 
# -----
# 
# #### Sequential Minimal Optimization
# 
# 首先定义特征到结果的输出函数：$u=w^Tx+b$.
# 
# 因为$w=\sum\alpha_iy_ix_i$
# 
# 有$u=\sum y_i\alpha_iK(x_i, x)-b$
# 
# 
# ----
# 
# $\max \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_j<\phi(x_i)^T,\phi(x_j)>$
# 
# $s.t.\ \sum^m_{i=1}\alpha_iy_i=0,$
# 
# $ \alpha_i \geq 0,i=1,2,...,m$
# 
# -----
# 参考资料：
# 
# [1] :[Lagrange Multiplier and KKT](http://blog.csdn.net/xianlingmao/article/details/7919597)
# 
# [2] :[推导SVM](https://my.oschina.net/dfsj66011/blog/517766)
# 
# [3] :[机器学习算法实践-支持向量机(SVM)算法原理](http://pytlab.org/2017/08/15/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5-%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA-SVM-%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/)
# 
# [4] :[Python实现SVM](http://blog.csdn.net/wds2006sdo/article/details/53156589)

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]


# In[3]:


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[4]:


plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()


# ----
# 
# 

# In[5]:


class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel
    
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        
        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0
        
    def _KKT(self, i):
        y_g = self._g(i)*self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1
    
    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j]*self.Y[j]*self.kernel(self.X[i], self.X[j])
        return r
    
    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k]*x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k]*x2[k] for k in range(self.n)]) + 1)**2
    
        return 0
    
    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]
    
    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        
        for i in index_list:
            if self._KKT(i):
                continue
            
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j
        
    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha      
    
    def fit(self, features, labels):
        self.init_args(features, labels)
        
        for t in range(self.max_iter):
            # train
            i1, i2 = self._init_alpha()
            
            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1]+self.alpha[i2]-self.C)
                H = min(self.C, self.alpha[i1]+self.alpha[i2])
            else:
                L = max(0, self.alpha[i2]-self.alpha[i1])
                H = min(self.C, self.C+self.alpha[i2]-self.alpha[i1])
                
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2*self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue
                
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)
            
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)
            
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new-self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new-self.alpha[i2])+ self.b 
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new-self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new-self.alpha[i2])+ self.b 
            
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2
                
            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return 'train done!'
            
    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
            
        return 1 if r > 0 else -1
    
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)
    
    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1, 1)*self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w


# In[6]:


svm = SVM(max_iter=200)


# In[7]:


svm.fit(X_train, y_train)


# In[8]:


svm.score(X_test, y_test)


# ## sklearn.svm.SVC

# In[9]:


from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)


# In[10]:


clf.score(X_test, y_test)


# ### sklearn.svm.SVC
# 
# *(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)*
# 
# 参数：
# 
# - C：C-SVC的惩罚参数C?默认值是1.0
# 
# C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
# 
# - kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
#     
#     – 线性：u'v
#     
#     – 多项式：(gamma*u'*v + coef0)^degree
# 
#     – RBF函数：exp(-gamma|u-v|^2)
# 
#     – sigmoid：tanh(gamma*u'*v + coef0)
# 
# 
# - degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
# 
# 
# - gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
# 
# 
# - coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
# 
# 
# - probability ：是否采用概率估计？.默认为False
# 
# 
# - shrinking ：是否采用shrinking heuristic方法，默认为true
# 
# 
# - tol ：停止训练的误差值大小，默认为1e-3
# 
# 
# - cache_size ：核函数cache缓存大小，默认为200
# 
# 
# - class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
# 
# 
# - verbose ：允许冗余输出？
# 
# 
# - max_iter ：最大迭代次数。-1为无限制。
# 
# 
# - decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
# 
# 
# - random_state ：数据洗牌时的种子值，int值
# 
# 
# 主要调节的参数有：C、kernel、degree、gamma、coef0。
