#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List


# In[2]:


def get_train_test_data() -> (pd.DataFrame, pd.DataFrame):
    train = pd.read_csv("data/transform_train.csv")
    test  = pd.read_csv("data/transform_test.csv")
    return (train, test)


# In[3]:


def apply_filters(df: pd.DataFrame, 
        max_price: float = 50_000_000, 
        max_beds: int = 40, 
        max_baths:int = 40
    ) -> pd.DataFrame:
    
    df = df[df.price <= max_price]
    df = df[df.baths <= max_baths]
    df = df[df.beds  <= max_beds]
    
    df = df[(df.lot_size_ratio > 1.0) & (df.lot_size_ratio < 40.0)]
    return df


# In[4]:


def evaluate_model(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    column_list: List[str]
    ) -> (float, float, List[float], float):

    # Training result
    X = df_train[column_list]
    y = df_train.price
    model = LinearRegression().fit(X, y)
    
    # Test score
    X_test = df_test[column_list]
    y_test = df_test.price


    # regression.score(X_test, y_test)
    return (model.score(X, y), model.score(X_test, y_test), model.coef_, model.intercept_)


# In[5]:


df_train, df_test = get_train_test_data()


# In[6]:


df_train.describe()


# In[7]:


df_test.describe()


# In[ ]:





# In[8]:


df_train = apply_filters(df_train)
df_test  = apply_filters(df_test)


# In[9]:


evaluate_model(df_train, df_test, ["beds", "baths", "size", "lot_size_ratio"])


# ### real filters

# In[ ]:





# In[10]:


df_train = apply_filters(df_train, max_price=6_000_000, max_beds=8, max_baths=8)
df_test  = apply_filters(df_test, 6_000_000, 8, 8)


# In[11]:


evaluate_model(df_train, df_test, ["beds", "baths", "size", "lot_size_ratio"])


# In[12]:


evaluate_model(df_train, df_test, ["beds", "baths", "size", "bed_bath_ratio", "lot_sq_ft"])


# In[ ]:





# In[13]:


df_train, df_test = get_train_test_data()


# In[14]:


df_train = apply_filters(df_train, max_price=5_000_000, max_beds=6, max_baths=8)
df_test  = apply_filters(df_test, 6_000_000, 6, 8)


# In[15]:


evaluate_model(df_train, df_test, ["beds", "baths", "size", "bed_bath_ratio", "lot_sq_ft", "lot_size_ratio"])


# In[16]:


import sklearn
sklearn.__version__


# In[ ]:




