#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from seattle import (
    get_train_test_data,
    one_hot_encode_zips,
    apply_filters,
    fit_model
)
    


# ## Get training and test data including categorical columns

# In[2]:


df_train, df_test = get_train_test_data()
df_train, df_test, features, zips = one_hot_encode_zips(df_train, df_test)


# In[3]:


features.remove("bed_bath_diff")
features.remove("price")
features = features + zips


# In[4]:


filters = [
    [("price", 200_000.0, 6_000_000.0)],
]


# In[5]:


result_columns = [
    "train_score",
    "test_score",
    "train_size",
    "test_size",
    "intercept",
]


# In[6]:


results = list()
for filter in filters:
    tr = apply_filters(df_train, filter)
    te = apply_filters(df_test,  filter)
    X = tr[features]
    y = tr["price"]
    model = fit_model(X, y)
        
    r = {
        "train_score": model.score(X, y),
        "test_score":  model.score(te[features], te["price"]),
        "train_size":  len(tr),
        "test_size":   len(te),
        "coeff":       model.coef_,
        "intercept":   model.intercept_,
        "model":       model,
    }
    results.append(r)

model_results = pd.DataFrame(results)


# In[7]:


print(f'Unfiltered Training set {len(df_train)} - Test set {len(df_test)}')
print(f'Filtered   Training set {len(tr)} - Test set {len(te)}')


# In[8]:


model_results[result_columns]


# In[9]:


model.rank_


# In[10]:


model = results[0]["model"]
intercept = model.intercept_
print(f'{"intercept":<15} {intercept:12,.2f}')
for i in range(model.rank_):
    print(f'{model.feature_names_in_[i]:<15} {model.coef_[i]:12,.2f}')


# **Alternatively:**
# 
# coef_series = pd.Series(
#     data=model.coef_,
#     index=model.feature_names_in_
# )
# 
# coef_series

# ## It will be useful to persist models

# In[11]:


from sklearn.linear_model import LinearRegression
from skops.io import dump, load


# In[12]:


response = "price"
features = features + zips
if response in features:
    features.remove(response)

X = tr[features]
y = tr[response]
model = LinearRegression().fit(X, y)


# In[13]:


Xnew = te[te['98188'] == 1][features]
predicted_price = model.predict(Xnew)
actual_price = te[te['98188'] == 1].price
print(f'predicted price {predicted_price[0]:>12,.0f}\nactual          {actual_price.values[0]:>12,.0f}\ndifference      {predicted_price[0] - actual_price.values[0]:>12,.0f}')


# In[14]:


dump(model, "data/seattle.linear_regression.model")


# In[15]:


loaded_model = load("data/seattle.linear_regression.model", trusted=True)


# In[16]:


assert (model.coef_== loaded_model.coef_).all()

assert model.intercept_  == loaded_model.intercept_
assert model.rank_       == loaded_model.rank_
assert model.score(X, y) == loaded_model.score(X, y)


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


Xnew = tr[features]
Ynew = model.predict(Xnew)
tr['predicted'] = Ynew


# In[19]:


Xnew = te[features]
Ynew = model.predict(Xnew)
te['predicted'] = Ynew


# In[20]:


fig, axes = plt.subplots(2,1)
fig.set_figheight(10)
fig.set_figwidth(20)
axes[0].set_title("Training")
axes[1].set_title("Test")
for i in range(2):
    axes[i].set_xlim(left=200_000., right=4_000_000.)
    axes[i].set_ylim(bottom=200_000., top=4_000_000.)
    axes[i].set_xlabel("Sale Price")
    axes[i].set_ylabel("Predicted Price")
    axes[i].set_aspect(aspect="equal")


axes[0].scatter(tr.price, tr.predicted)
axes[1].scatter(te.price, te.predicted)
plt.show()


# In[ ]:




