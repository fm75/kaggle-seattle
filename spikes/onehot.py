#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from typing import List


# In[15]:


df1 = pd.read_csv("data/transform_train.csv")
df2 = pd.read_csv("data/transform_test.csv")


# In[19]:


sorted(df1.zip_code.unique())


# In[20]:


sorted(df2.zip_code.unique())


# In[5]:


zips = pd.get_dummies(df.zip_code, prefix_sep='_', prefix="")


# In[6]:


dfenc = pd.concat([df, zips], axis=1)


# In[7]:


dfenc.head(5)


# In[8]:


a = ["beds", "baths"]
#.extend(list(zips.columns.values))
a += list(zips.columns.values)


# In[9]:


features = df.drop(["zip_code", "price",], axis=1)
y = df.price


# In[10]:


features


# In[11]:


y


# In[62]:


d1 = dict(v=["a", "d", "b", "c", "f"])
d2 = dict(v=["a", "d", "b", "c", "a", "g"])


# In[63]:


df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)


# In[64]:


df1


# In[65]:


df2


# In[66]:


dfo1 = pd.get_dummies(df1, prefix="", prefix_sep="", columns=['v'])
dfo1
                      


# In[67]:


dfo1 = dfo1.reindex(columns=["a", "b", "c", "d", "f", "g"]).fillna(0)


# In[68]:


dfo1


# In[69]:


df2


# In[70]:


df2 = pd.get_dummies(df2, prefix="", prefix_sep="", columns=['v'])
df2 = df2.reindex(columns=["a", "b", "c", "d", "f", "g"]).fillna(0)
df2


# In[72]:


[1, 2] + [x for x in (1, 2, 3)]
        


# In[ ]:




