#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')


# In[3]:


train.head(1)


# In[4]:


test.head(1)


# In[5]:


train.describe()


# In[6]:


test.describe()


# In[7]:


fig, axes = plt.subplots(1, 2)
axes[0].set_title("beds train")
axes[1].set_title("beds test")

x = train.beds.values
y = test.beds.values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# In[8]:


fig, axes = plt.subplots(1, 2)
axes[0].set_title("baths train")
axes[1].set_title("baths test")

x = train.baths.values
y = test.baths.values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# ### house size

# In[9]:


train.size_units.unique(), train.size_units.unique()


# In[10]:


train.hist('size')


# In[11]:


fig, axes = plt.subplots(1, 2)
axes[0].set_title("house size train")
axes[1].set_title("house size test")

x = train["size"].values
y = test["size"].values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# In[12]:


fig, axes = plt.subplots(1, 2)
axes[0].set_title("house size train")
axes[1].set_title("house size test")

df0 = train[train["size"] < 5_000]
df1 = test[test["size"] < 5_000]

x = df0["size"].values
y = df1["size"].values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# In[13]:


len(df0), len(df1)


# ### lot size

# In[14]:


train.lot_size_units.unique(), test.lot_size_units.unique()


# In[15]:


train[train.lot_size_units.isna()]


# In[16]:


test[test.lot_size_units.isna()]


# In[17]:


train[train.lot_size.isna()]


# In[18]:


test[test.lot_size.isna()]


# ### zip_codes

# In[19]:


sorted(train.zip_code.unique())


# In[20]:


sorted(test.zip_code.unique())


# ### price

# In[21]:


train.price.min(), train.price.max(), train.price.mean(), train.price.median()


# In[22]:


test.price.min(), test.price.max(), test.price.mean(), test.price.median()


# In[23]:


fig, axes = plt.subplots(1, 2)
axes[0].set_title("house price train")
axes[1].set_title("house price test")

x = train["price"].values
y = test ["price"].values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# In[24]:


normal_limit =  2_000_000
jumbo_limit  = 10_000_000


# In[25]:


train_small = train[(train.price <= normal_limit)]
test_small  = test [(test.price  <= normal_limit)]

fig, axes = plt.subplots(1, 2)
axes[0].set_title("house price train < $2MM")
axes[1].set_title("house price test < $2MM")

x = train_small["price"].values
y = test_small["price"].values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# In[26]:


train_small = train[(train.price <= normal_limit)]
test_small  = test [(test.price  <= normal_limit)]

fig, axes = plt.subplots(1, 2)
axes[0].set_title("house price train < $10MM")
axes[1].set_title("house price test < $10MM")

x = train_small["price"].values
y = test_small["price"].values

axes[0].hist(x)
axes[1].hist(y)
plt.show()


# In[27]:


dfjumbo = train[train.price>jumbo_limit]
dfjumbo


# ### zip code

# In[28]:


train.hist('zip_code',bins=100, figsize=(16,6))


# In[29]:


test.hist('zip_code',bins=100, figsize=(16,6))


# ### Price vs zip

# In[30]:


train.plot.scatter("zip_code", "price", figsize=(16,4))


# In[31]:


test.plot.scatter("zip_code", "price", figsize=(16,4))


# In[32]:


train[train.price<jumbo_limit].plot.scatter("zip_code", "price", figsize=(16,4))


# In[33]:


train_small.plot.scatter("zip_code", "price", figsize=(16,4))


# In[34]:


df = pd.read_csv('data/transform_train.csv')


# In[35]:


correl = df.corr().round(2)
plt.figure(figsize = (15,10))
sns.heatmap(correl, annot = True, cmap = 'YlOrBr')


# ### Post-Transform

# In[36]:


df1 = pd.read_csv('data/transform_train.csv')
df2 = pd.read_csv('data/transform_test.csv')


# In[37]:


transformed_columns = [
    'lot_sq_ft',
    'bed_bath_diff',
    'bed_bath_ratio',
    'lot_size_ratio',
]    


# In[38]:


df1[transformed_columns].describe()


# In[39]:


df2[transformed_columns].describe()


# In[ ]:




