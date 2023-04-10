#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from seattle import (
    get_train_test_data,
    one_hot_encode_zips,
    apply_filters,
    evaluate_model
)
    


# In[2]:


df_train, df_test = get_train_test_data()


# In[3]:


print(f'Training set {len(df_train)} - Test set {len(df_test)}')


# In[4]:


df_train, df_test, features, zips = one_hot_encode_zips(df_train, df_test)


# # Tuning

# In[5]:


filters = [
    [("price",       1.0, 4_000_000.0)],
    [("price",       1.0, 6_000_000.0)],
    [("price", 200_000.0, 6_000_000.0)],
    [("price", 250_000.0, 6_000_000.0)],
    [("price", 300_000.0, 6_000_000.0)],
    [("price",       1.0, 7_000_000.0)],
    [("baths", 1.0, 5.0)],
    [("baths", 1.0, 6.0)],
    [("baths", 1.0, 7.0)],
    [("size",    0.0, 6000.0)],
    [("size",  800.0, 6000.0)],
    [("size", 1000.0, 5000.0)],
    [("size", 1000.0, 6000.0)],
    [("size", 1000.0, 7000.0)],
    [("lot_sq_ft", 0.0,  10_000.0)],
    [("lot_sq_ft", 0.0, 100_000.0)],
    [("lot_sq_ft", 0.0, 200_000.0)],
    [("lot_size_ratio", 0.0, 5.0)],
    [("lot_size_ratio", 1.0, 5.0)],
    [
            ("price", 1.0, 6_000_000.0),
            ("baths", 1.0, 7.0),
            ("size", 1000.0, 6000.0),
            ("lot_size_ratio", 0.0, 5.0)
    ],
    [
            ("price", 250_000.0, 6_000_000.0),
            ("baths", 1.0, 7.0),
    ],
    [
            ("price", 200_000.0, 6_000_000.0),
            ("baths", 1.0, 7.0),
            ("size", 1000.0, 6000.0),
    ],

]


# In[6]:


results = list()
result_columns = [
    "train_score",
    "train_size",
    "test_score",
    "test_size",
    "intercept",
]
    

for filter in filters:
    tr = apply_filters(df_train, filter)
    te = apply_filters(df_test,  filter)
    
    train_score, test_score, coeff, intercept = evaluate_model(tr, te, features, "price")
    
    r = {
        "train_score": train_score,
        "train_size": len(tr),
        "test_score": test_score,
        "test_size": len(te),
        "coeff": coeff,
        "intercept": intercept,
    }
    results.append(r)

model_results = pd.DataFrame(results)

model_results[result_columns]


# ### Include one-hot encoding

# In[7]:


results = list()

for filter in filters:
    tr = apply_filters(df_train, filter)
    te = apply_filters(df_test,  filter)
    
    train_score, test_score, coeff, intercept = evaluate_model(tr, te, features+zips, "price")
    
    r = {
        "train_score": train_score,
        "train_size": len(tr),
        "test_score": test_score,
        "test_size": len(te),
        "coeff": coeff,
        "intercept": intercept,
    }
    results.append(r)

model_results = pd.DataFrame(results)

model_results[result_columns]


# In[8]:


results = list()

for filter in filters:
    tr = apply_filters(df_train, filter)
    te = apply_filters(df_test,  filter)
    f = features + zips
    f.remove("bed_bath_diff")
    train_score, test_score, coeff, intercept = evaluate_model(tr, te, f, "price")
    
    r = {
        "train_score": train_score,
        "train_size": len(tr),
        "test_score": test_score,
        "test_size": len(te),
        "coeff": coeff,
        "intercept": intercept,
    }
    results.append(r)

model_results = pd.DataFrame(results)

model_results[result_columns]


# # Tuned Results

# In[9]:


filters = [
    [("price", 200_000.0, 6_000_000.0)],
]


# In[10]:


results = list()

for filter in filters:
    tr = apply_filters(df_train, filter)
    te = apply_filters(df_test,  filter)
    features.remove("bed_bath_diff")
    f = features + zips
    train_score, test_score, coeff, intercept = evaluate_model(tr, te, f, "price")
    
    r = {
        "train_score": train_score,
        "train_size": len(tr),
        "test_score": test_score,
        "test_size": len(te),
        "coeff": coeff,
        "intercept": intercept,
    }
    results.append(r)

model_results = pd.DataFrame(results)


# # Results

# In[11]:


print(f'Unfiltered Training set {len(df_train)} - Test set {len(df_test)}')
print(f'Filtered   Training set {len(tr)} - Test set {len(te)}')


# In[12]:


tr.describe()[features]


# In[13]:


te.describe()[features]


# ## Quality of fit

# In[14]:


model_results[result_columns]


# ### What is the impact of removing houses with 98188 zip code from test data?
# (Since there are no data for it in the training set.)

# In[15]:


results = list()

for filter in filters:
    tr = apply_filters(df_train, filter)
    te = apply_filters(df_test,  filter)
    # features.remove("bed_bath_diff")
    f = features + zips
    train_score, test_score, coeff, intercept = evaluate_model(tr, te[te['98188'] != 1], f, "price")
    
    r = {
        "train_score": train_score,
        "train_size": len(tr),
        "test_score": test_score,
        "test_size": len(te),
        "coeff": coeff,
        "intercept": intercept,
    }
    results.append(r)

model_results = pd.DataFrame(results)


# In[16]:


model_results[result_columns]


# ## Model

# In[17]:


print(f'{"intercept":<15} {intercept:12,.2f}')
for i, feature in enumerate(f):
    print(f'{feature:<15} {coeff[i]:12,.2f}')


# In[18]:


tr[tr['98188'] == 1]


# In[19]:


te[te['98188'] == 1]


# ## It will be useful to persist models

# In[20]:


from sklearn.linear_model import LinearRegression
from skops.io import dump, load


# In[21]:


response = "price"
features = features + zips
if response in features:
    features.remove(response)

X = tr[features]
y = tr[response]
model = LinearRegression().fit(X, y)


# In[22]:


Xnew = te[te['98188'] == 1][f]
predicted_price = model.predict(Xnew)
actual_price = te[te['98188'] == 1].price
print(f'predicted price {predicted_price[0]:>12,.0f}\nactual          {actual_price.values[0]:>12,.0f}\ndifference      {predicted_price[0] - actual_price.values[0]:>12,.0f}')


# In[23]:


dump(model, "data/seattle.linear_regression.model")


# In[24]:


loaded_model = load("data/seattle.linear_regression.model", trusted=True)


# In[25]:


assert (model.coef_== loaded_model.coef_).all()

assert model.intercept_  == loaded_model.intercept_
assert model.rank_       == loaded_model.rank_
assert model.score(X, y) == loaded_model.score(X, y)


# In[26]:


from skops.io import dumps, loads


# In[27]:


x = dumps(model)
loadsed = loads(x, trusted=True)


# In[28]:


assert (model.coef_== loadsed.coef_).all()

assert model.intercept_  == loadsed.intercept_
assert model.rank_       == loadsed.rank_
assert model.score(X, y) == loadsed.score(X, y)


# In[29]:


get_ipython().run_cell_magic('timeit', '', 'assert (model.coef_== loaded_model.coef_).all()\n')


# In[30]:


import numpy as np


# In[31]:


get_ipython().run_cell_magic('timeit', '', 'assert np.array_equal(model.coef_, loaded_model.coef_)\n')


# In[32]:


get_ipython().run_cell_magic('timeit', '', 'assert np.array_equiv(model.coef_, loaded_model.coef_)\n')


# In[33]:


n = len(model.coef_)


# In[34]:


get_ipython().run_cell_magic('timeit', '', 'for i in range(n):\n    assert model.coef_[i] == loaded_model.coef_[i]\n')


# In[ ]:




