#!/usr/bin/env python
# coding: utf-8

# 原文代码作者：https://github.com/wzyonggege/statistical-learning-method
# 
# 配置环境：python 3.6 代码全部测试通过。
# 
# 整理者（微信公众号）：机器学习之美（ID: BreakIntoAI）


# # 第9章 EM算法及其推广

# # Expectation Maximization algorithm
# 
# ### Maximum likehood function
# 
# [likehood & maximum likehood](http://fangs.in/post/thinkstats/likelihood/)
# 
# > 在统计学中，似然函数（likelihood function，通常简写为likelihood，似然）是一个非常重要的内容，在非正式场合似然和概率（Probability）几乎是一对同义词，但是在统计学中似然和概率却是两个不同的概念。概率是在特定环境下某件事情发生的可能性，也就是结果没有产生之前依据环境所对应的参数来预测某件事情发生的可能性，比如抛硬币，抛之前我们不知道最后是哪一面朝上，但是根据硬币的性质我们可以推测任何一面朝上的可能性均为50%，这个概率只有在抛硬币之前才是有意义的，抛完硬币后的结果便是确定的；而似然刚好相反，是在确定的结果下去推测产生这个结果的可能环境（参数），还是抛硬币的例子，假设我们随机抛掷一枚硬币1,000次，结果500次人头朝上，500次数字朝上（实际情况一般不会这么理想，这里只是举个例子），我们很容易判断这是一枚标准的硬币，两面朝上的概率均为50%，这个过程就是我们运用出现的结果来判断这个事情本身的性质（参数），也就是似然。

# $$P(Y|\theta) = \prod[\pi p^{y_i}(1-p)^{1-y_i}+(1-\pi) q^{y_i}(1-q)^{1-y_i}]$$
# 
# ### E step:
# 
# $$\mu^{i+1}=\frac{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}}{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}+(1-\pi) (q^i)^{y_i}(1-(q^i))^{1-y_i}}$$

# In[1]:


import numpy as np
import math


# In[2]:


pro_A, pro_B, por_C = 0.5, 0.5, 0.5

def pmf(i, pro_A, pro_B, por_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow((1-pro_B), 1-data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow((1-pro_C), 1-data[i])
    return pro_1 / (pro_1 + pro_2)


# ### M step:
# 
# $$\pi^{i+1}=\frac{1}{n}\sum_{j=1}^n\mu^{i+1}_j$$
# 
# $$p^{i+1}=\frac{\sum_{j=1}^n\mu^{i+1}_jy_i}{\sum_{j=1}^n\mu^{i+1}_j}$$
# 
# $$q^{i+1}=\frac{\sum_{j=1}^n(1-\mu^{i+1}_jy_i)}{\sum_{j=1}^n(1-\mu^{i+1}_j)}$$

# In[3]:


class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob
        
    # e_step
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow((1-self.pro_B), 1-data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1-self.pro_C), 1-data[i])
        return pro_1 / (pro_1 + pro_2)
    
    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(count):
            _ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            pro_A = 1/ count * sum(_pmf)
            pro_B = sum([_pmf[k]*data[k] for k in range(count)]) / sum([_pmf[k] for k in range(count)])
            pro_C = sum([(1-_pmf[k])*data[k] for k in range(count)]) / sum([(1-_pmf[k]) for k in range(count)])
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(d+1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C
    


# In[4]:


data=[1,1,0,1,0,0,1,0,1,1]


# In[5]:


em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)


# In[6]:


# 第一次迭代
f.send(1)


# In[7]:


# 第二次
f.send(2)


# In[8]:


em = EM(prob=[0.4, 0.6, 0.7])
f2 = em.fit(data)
next(f2)


# In[9]:


f2.send(1)


# In[10]:


f2.send(2)
