#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path

from seattle import (
    apply_filters,
    fit_model
)
from transformations import (
    normalize_lot_size,
    normalize_house_size,
    baths_to_beds_ratio,
    lot_size_factor,
    apply_transformations,
    get_unique_category_values,
    ohe,
)
from skops.io import dump, load


# ## Get training and test data including categorical columns

# In[2]:


# Retrieve raw data
data_folder = Path("data")
path_trn = data_folder / 'train.csv'
path_tst = data_folder / 'test.csv'

# Raw Data
trn = pd.read_csv(path_trn, dtype={"zip_code": str})
tst = pd.read_csv(path_tst, dtype={"zip_code": str})


# In[3]:


# Select transformations to use
transformations = [
    normalize_lot_size,
    normalize_house_size,
    baths_to_beds_ratio,
    lot_size_factor,
]
# Apply transformations to training and test data (in place)
apply_transformations(trn, transformations)
apply_transformations(tst, transformations)


# In[4]:


# One-hot encode zip_code column on both datasets
zips = get_unique_category_values([trn, tst], "zip_code")
trn = ohe(trn, "zip_code", zips)
tst = ohe(tst, "zip_code", zips)


# In[5]:


trn = pd.concat([trn, tst[tst['98188'] == 1]], copy=True)


# In[6]:


model = load("data/seattle.tuned_linear_regression.model", trusted=True)


# In[7]:


model.rank_

features     = model.feature_names_in_
intercept    = model.intercept_
coefficients = model.coef_


# In[8]:


print(f'{"intercept":<15} {intercept:12,.2f}')
for i in range(model.rank_ + 1):
    print(f'{features[i]:<15} {coefficients[i]:12,.2f}')


# ----

# In[9]:


print(f'Unfiltered Training set {len(trn)} - Test set {len(tst)}')

**Alternatively:**

coef_series = pd.Series(
    data=model.coef_,
    index=model.feature_names_in_
)

coef_series
# In[10]:


Xnew = tst[tst['98188'] == 1][features]
predicted_price = model.predict(Xnew)
actual_price = tst[tst['98188'] == 1].price
print(f'predicted price {predicted_price[0]:>12,.0f}\nactual          {actual_price.values[0]:>12,.0f}\ndifference      {predicted_price[0] - actual_price.values[0]:>12,.0f}')


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


Xnew = trn[features]
Ynew = model.predict(Xnew)
trn['predicted'] = Ynew


# In[13]:


Xnew = tst[features]
Ynew = model.predict(Xnew)
tst['predicted'] = Ynew


# In[14]:


fig, axes = plt.subplots(2,1)
fig.set_figheight(10)
fig.set_figwidth(20)
axes[0].set_title("Training")
axes[1].set_title("Test")
for i in range(2):
    axes[i].set_xlim(left=200_000., right=5_000_000.)
    axes[i].set_ylim(bottom=200_000., top=5_000_000.)
    axes[i].set_xlabel("Sale Price")
    axes[i].set_ylabel("Predicted Price")
    axes[i].set_aspect(aspect="equal")


axes[0].scatter(trn.price, trn.predicted)
axes[1].scatter(tst.price, tst.predicted)
plt.show()


# In[15]:


fig, axes = plt.subplots(6,5)
fig.set_figheight(30)
fig.set_figwidth(20)

for i in range(6):
    for j in range (5):
        axes[i][j].set_xlim(left=200_000., right=5_000_000.)
        axes[i][j].set_ylim(bottom=200_000., top=5_000_000.)
        axes[i][j].set_xlabel("Sale Price")
        axes[i][j].set_ylabel("Predicted Price")
        axes[i][j].set_aspect(aspect="equal")        
        n = 5*i + j + 6
        if n > model.rank_:
            axes[i][j].set_title("")
            break
        zipcode = str(features[n])
        axes[i][j].set_title(zipcode)
        
        df = tst[tst[zipcode] == 1]
        if len(df > 0):
            axes[i][j].scatter(df.price, df.predicted)
        else:
            print("no data for ", zipcode)
plt.show()


# In[ ]:




