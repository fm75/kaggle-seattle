#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/jovyan/w")


# In[2]:


import pandas as pd
from transformations import (
    get_unique_category_values, 
    ohe
)


# ### Set up disjoint sets of zip codes in two DataFrames to test one-hot encoding of them
# - This problem may not exist if you split a dataset into training and test **after** one-hot encoding.
# - When you want to make predictions, you **will** need to encode the input consistently with the model.
# - So it will be useful to have a mechanism for one-hot encoding where only one category value exists.

# In[3]:


row1 = dict(x=1, zip_code='19711')
row2 = dict(x=2, zip_code='18540')
row3 = dict(x=3, zip_code='32163')
df1 = pd.DataFrame([row1, row2])
df2 = pd.DataFrame([row2, row3])


# In[4]:


unique_zips = get_unique_category_values([df1, df2], 'zip_code')


# In[5]:


assert unique_zips == ['18540', '19711', '32163']


# In[6]:


ohe1 =ohe(df1, "zip_code", unique_zips)
ohe1


# In[7]:


ohe2 = ohe(df2, "zip_code", unique_zips)
ohe2


# ### Zip Codes, by default, are read in as integers by read_csv, which is awkward when they start with zero.

# In[8]:


get_ipython().system('cat data/foo.csv')


# In[9]:


example = "data/foo.csv"


# In[10]:


df = pd.read_csv(example)
df


# ### We can force them to strings using `converters` or `dtype`.

# #### Converters

# In[ ]:


to_str = lambda x: str(x)


# In[ ]:


df = pd.read_csv(example, converters={"zip": to_str})
df


# In[ ]:


df = pd.read_csv(example, converters={"zip": lambda x: str(x)})
df


# #### dtype

# In[ ]:


df = pd.read_csv(example, dtype={"zip": str})
df

