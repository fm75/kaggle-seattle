#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('cd ~/w')


# In[2]:


import pandas as pd


# ### Remove NaNs

# In[3]:


df = pd.read_csv("data/train.csv").dropna()


# In[4]:


df_test = pd.read_csv("data/test.csv").dropna()


# In[5]:


df_test.info()


# In[6]:


df_test.describe()


# ### Normalize lot size to square feet

# In[7]:


df['lot_sq_ft'] = df.lot_size
df.loc[df.lot_size_units == 'acre', 'lot_sq_ft'] = 43_560 * df.lot_size


# In[8]:


df_test['lot_sq_ft'] = df_test.lot_size
df_test.loc[df_test.lot_size_units == 'acre', 'lot_sq_ft'] = 43_560 * df_test.lot_size


# In[9]:


df_test.describe()


# ### Create bads-baths, beds/baths, lot_size/size columns

# In[10]:


df["bed_bath_diff"]  = df.beds - df.baths
df["bed_bath_ratio"] = df.beds / df.baths
df["lot_size_ratio"] = df.lot_sq_ft / df["size"]


# In[11]:


df_test["bed_bath_diff"]  = df_test.beds - df_test.baths
df_test["bed_bath_ratio"] = df_test.beds / df_test.baths
df_test["lot_size_ratio"] = df_test.lot_sq_ft / df_test["size"]


# ### drop units columns and lot_size column

# In[12]:


omit_columns = ["size_units", "lot_size_units", "lot_size"]
df      = df.drop(omit_columns, axis=1)
df_test = df_test.drop(omit_columns, axis=1)


# In[13]:


df.to_csv("data/transform_train.csv", index=False)


# In[14]:


df_test.to_csv("data/transform_test.csv", index=False)


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


df_test.describe()


# In[18]:


df[df.lot_size_ratio < 1.0]


# In[19]:


df[df.lot_size_ratio > 50.0]


# In[20]:


df_test[df_test.lot_size_ratio < 1.0]


# In[21]:


df_test[df_test.lot_size_ratio > 50.0]


# In[ ]:




