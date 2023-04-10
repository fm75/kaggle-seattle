#!/usr/bin/env python
# coding: utf-8

# # Graphical Data Exploration

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Data sets
# We will be using a set for training data and a separate test to test the training.

# In[2]:


df0 = pd.read_csv('data/train.csv')
df1 = pd.read_csv('data/test.csv')


# ### Basic Plotting with Pandas
# - We would like to know the distribution of data. 
# - Often a histogram can be useful to explore the distribution of data.
# 
# - We often may wish to exclude outliers from a model. 
# - A visual representation can help us.

# In[3]:


df0.hist(["beds", "baths", "size", "price"])


# ### Comparison of Training and Testing Data
# There is no guarantee that the distribution of the training and testing data will match.
# 
# Looking at the distribution of features of each can help avoid problems. We
# would like to see them together.
# 
# At this point, we can use our DataFrames for the source of data to visualize, but
# we will want to drop into a lower level than pandas for doing it.

# In[4]:


train = df0.beds.values
test  = df1.beds.values


# In[5]:


plt.hist([train, test])
plt.show()


# #### Normalization
The training set is generally 3 to 4 times the size of the test set.
We could attempt to normalize the data, but it can be simpler just to
display the data side-by-side allowing the scaling of the axes to handle this job.
# In[6]:


fig, axes = plt.subplots(1, 2)

df0.hist('beds', bins=15, ax=axes[0])
df1.hist('beds', bins=15, ax=axes[1])


# #### Finer Control of Presentation Elements
The previous figure did not allow us to control the title. It was convenient to use
pandas' support for graphical analysis using the DataFrame and matplotlib.pyplot
for creating two axes for a single figure and let the scaling of each axis float.Below we get the data from the series (columns) using the values property.
We then can let matplotlib control titling each axis, and use the hist function
in matplotlib.pyplot directly.
# In[7]:


fig, axes = plt.subplots(1, 2)
axes[0].set_title("beds train")
axes[1].set_title("beds test")

train = df0.beds.values
test  = df1.beds.values

axes[0].hist(train)
axes[1].hist(test)
plt.show()


# In[ ]:





# In[ ]:




