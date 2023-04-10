#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from typing import List, Tuple
from seattle import apply_filters


# In[11]:


row1 = dict(x=1, zip_code=19711)
row2 = dict(x=2, zip_code=18540)
row3 = dict(x=3, zip_code=32163)
row4 = dict(x=3, zip_code=19711)
df1 = pd.DataFrame([row1, row2, row3, row4])


# In[12]:


apply_filters(df1, [])


# In[4]:


filters = [("x", 2.0, 3.0)]


# In[5]:


apply_filters(df1, filters)


# In[6]:


filter_zips = [("zip_code", 19000.0, 30000.0)]


# In[7]:


apply_filters(df1, filter_zips)


# In[8]:


filter_both = filters + filter_zips


# In[9]:


apply_filters(df1, filter_both)


# In[ ]:




