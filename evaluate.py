#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from typing import List, Tuple
from seattle import evaluate_model


# In[2]:


row1 = dict(x=1, y=4.14)
row2 = dict(x=2, y=7.28)
row3 = dict(x=3, y=10.42)
row4 = dict(x=2, y=7.29)

train = pd.DataFrame([row1, row2])
test  = pd.DataFrame([row3, row4])


# In[3]:


evaluate_model(train, test, ['x'], 'y')


# In[ ]:




