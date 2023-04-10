#!/usr/bin/env python
# coding: utf-8

# $\hat{Y} = \hat{\beta}_{0} + \sum \limits _{j=1} ^{p} X_{j}\hat{\beta}_{j} $

# In[1]:


get_ipython().system('cd ~/w')


# In[2]:


import pandas as pd
from transformations import (
    normalize_lot_size,
    normalize_house_size,
    baths_to_beds_ratio,
    lot_size_factor,
    apply_transformations,
)


# In[3]:


transformations = [
    normalize_lot_size,
    normalize_house_size,
    baths_to_beds_ratio,
    lot_size_factor,
]


# In[4]:


get_ipython().system('date -Ins')


# In[5]:


df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")


# In[6]:


get_ipython().system('date -Ins')


# In[7]:


apply_transformations(df_train, transformations)
apply_transformations(df_test, transformations)


# In[8]:


get_ipython().system('date -Ins')


# In[9]:


df_train.to_csv("data/transform_with_nan_train.csv", index=False)
df_test.to_csv("data/transform_with_nan_test.csv", index=False)


# In[10]:


get_ipython().system('date -Ins')


# # -------------------------------------------------------------

# In[11]:


df_train.describe()


# In[12]:


df_test.describe()


# In[13]:


df_train[df_train.lot_size_ratio < 0.5].sort_values("lot_size_ratio", ascending=True)


# In[14]:


df_test[df_test.lot_size_ratio < 0.5].sort_values("lot_size_ratio", ascending=True)


# In[15]:


df_train[(df_train.lot_size_ratio >= 0.5) & (df_train.lot_size_ratio < 1.0)].sort_values("lot_size_ratio", ascending=True)


# In[16]:


df_test[(df_test.lot_size_ratio >= 0.5) & (df_test.lot_size_ratio < 1.0)].sort_values("lot_size_ratio", ascending=True)


# In[17]:


df_train[df_train.lot_size_ratio > 50.0].sort_values("lot_size_ratio", ascending=False)


# In[18]:


df_test[df_test.lot_size_ratio > 50.0].sort_values("lot_size_ratio", ascending=False)


# In[ ]:




