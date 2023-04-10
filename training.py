#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('cd ~/w')


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv("data/transform_train.csv")


# In[4]:


df_test = pd.read_csv("data/transform_test.csv")
df_test.info()


# In[5]:


df_test.describe()


# In[6]:


px_limit  = 4_000_000
max_beds  = 7
max_baths = 8


# In[7]:


df = df[df.price <= px_limit]
df = df[df.baths <= max_baths]
df = df[df.beds  <= max_beds]


# ## First cut - really naive model. Inputs - bedrooms, bathroom, house size

# In[8]:


df.head(2)


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


X = df[["beds", "baths", "size"]]
y = df.price
regression = LinearRegression().fit(X, y)

regression.score(X, y), regression.coef_, regression.intercept_


# In[11]:


X_test = df_test[["beds", "baths", "size"]]
y_test = df_test.price

regression.score(X_test, y_test)


# ### Include lot size in the model

# In[12]:


X = df[["beds", "baths", "size", "lot_sq_ft"]]
y = df.price
regression = LinearRegression().fit(X, y)

regression.score(X, y), regression.coef_, regression.intercept_


# In[13]:


X_test = df_test[["beds", "baths", "size", "lot_sq_ft"]]
y_test = df_test.price

regression.score(X_test, y_test)


# ### bed bath ratio

# In[14]:


X = df[["beds", "baths", "size", "lot_sq_ft",  "bed_bath_ratio"]]
y = df.price
regression = LinearRegression().fit(X, y)

regression.score(X, y), regression.coef_, regression.intercept_


# In[15]:


X_test = df_test[["beds", "baths", "size", "lot_sq_ft",  "bed_bath_ratio"]]
y_test = df_test.price

regression.score(X_test, y_test)


# ### drop bedrooms

# In[16]:


X = df[["baths", "size", "lot_sq_ft",  "bed_bath_ratio"]]
y = df.price
regression = LinearRegression().fit(X, y)

regression.score(X, y), regression.coef_, regression.intercept_


# In[17]:


X_test = df_test[["baths", "size", "lot_sq_ft",  "bed_bath_ratio"]]
y_test = df_test.price

regression.score(X_test, y_test)


# ### drop bed/bath ratio

# In[18]:


X = df[["baths", "size", "lot_sq_ft",]]
y = df.price
regression = LinearRegression().fit(X, y)

regression.score(X, y), regression.coef_, regression.intercept_


# In[19]:


X_test = df_test[["baths", "size", "lot_sq_ft", ]]
y_test = df_test.price

regression.score(X_test, y_test)


# In[ ]:




