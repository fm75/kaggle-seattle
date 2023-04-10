#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/jovyan/w")


# In[2]:


import pandas as pd

from seattle import (
    get_train_test_data,
    get_train_test_data_with_nan,
    apply_filters,
    fit_model
)
    


# In[3]:


from typing import List
import numpy as np


def one_hot_encode_zips(df1: pd.DataFrame, df2: pd.DataFrame) -> (
        pd.DataFrame, 
        pd.DataFrame,
        List[str],
        List[str],
        ):
    """
    One-hot encode the zip codes (create a "0/1" column for each zip code and 
    populate it with one for the correct zip for each row).
    Return **copies** of the DataFrames so that the raw data can be reused with
    different models and filtering strategies easily.
    Also return a list of quantitative feature columns and a list of the zips to
    be used by the models when training.
    """
    
    # First we need to deal with the fact that neither the training nor the test data
    # have a full set of the Seattle Housing zip codes.
    z1 = df1.zip_code.unique()
    z2 = df2.zip_code.unique()
    
    # Note - we won't need to convert zips from int to str if we do it when
    # populating the DataFrame with read_csv().
    zips = sorted([str(x) for x in list(np.unique(np.concatenate((z1,z2), axis=0)))])
    
    # We need a full set up the columns, less the "zip_code" column, which will be
    # removed by the get_dummies one-hot encoding call. This set of column names,
    # cols, will be used to add NA values where we have no zip codes in the 
    # reindex operation.
    qfeats = df1.columns.to_list()
    cols = qfeats + zips

    f1 = pd.get_dummies(df1, prefix="", prefix_sep="", columns=["zip_code"])
    f2 = pd.get_dummies(df2, prefix="", prefix_sep="", columns=["zip_code"])

    cols.remove('zip_code')

    f1 = f1.reindex(columns=cols).fillna(0)
    f2 = f2.reindex(columns=cols).fillna(0)
    
    # qfeats = quantitative feature columns
    # zips is the one-hot encoded columns for zip codes.
    qfeats.remove('zip_code')
    return (f1, f2, qfeats, zips)


# In[4]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.feature_selection import SelectPercentile, chi2


# ## Get training and test data including categorical columns

# In[5]:


df_train, df_test = get_train_test_data()
df_train, df_test, features, zips = one_hot_encode_zips(df_train, df_test)


# In[6]:


df_train, df_test = get_train_test_data_with_nan()
df_train, df_test, features, zips = one_hot_encode_zips(df_train, df_test)


# In[7]:


numerical_features   = features
categorical_features = zips

numerical_features.remove("price")
# numerical_features.remove("bed_bath_diff")

numerical_featurescategorical_features
# In[8]:


numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        #("selector", SelectPercentile(chi2, percentile=98)),

    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        # ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# In[9]:


clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())]
)


# In[10]:


clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", ElasticNet(fit_intercept=True, alpha=0.0))]
)


# In[11]:


X_train = df_train[numerical_features + categorical_features]
y_train = df_train["price"]


# In[12]:


filters = [
    [
    ("price", 200_000.0, 6_000_000.0)
    ],
]


# In[13]:


for filter in filters:
    tr = apply_filters(df_train, filter)
X_train = tr[numerical_features + categorical_features]
y_train = tr["price"]


# In[14]:


X_test = df_test[numerical_features + categorical_features]
y_test = df_test["price"]


# In[15]:


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_train, y_train))


# In[16]:


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_train, y_train))


# In[17]:


clf.fit(X_test, y_test)
print("model score: %.3f" % clf.score(X_test, y_test))
clf.feature_names_in_, clf.n_features_in_


# In[18]:


clf.fit(X_test, y_test)
print("model score: %.3f" % clf.score(X_test, y_test))
clf.feature_names_in_, clf.n_features_in_


# In[19]:


clf.get_params()


# In[20]:


X_train


# In[21]:


# features.remove("bed_bath_diff")
# features.remove("price")
features = features + zips


# In[22]:


result_columns = [
    "train_score",
    "test_score",
    "train_size",
    "test_size",
    "intercept",
]


# In[23]:


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


# In[24]:


print(f'Unfiltered Training set {len(df_train)} - Test set {len(df_test)}')
print(f'Filtered   Training set {len(tr)} - Test set {len(te)}')


# In[25]:


model_results[result_columns]


# In[26]:


model.rank_


# In[27]:


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

# In[28]:


from sklearn.linear_model import LinearRegression
from skops.io import dump, load


# In[29]:


response = "price"
features = features + zips
if response in features:
    features.remove(response)

X = tr[features]
y = tr[response]
model = LinearRegression().fit(X, y)


# In[30]:


Xnew = te[te['98188'] == 1][features]
predicted_price = model.predict(Xnew)
actual_price = te[te['98188'] == 1].price
print(f'predicted price {predicted_price[0]:>12,.0f}\nactual          {actual_price.values[0]:>12,.0f}\ndifference      {predicted_price[0] - actual_price.values[0]:>12,.0f}')


# In[31]:


dump(model, "data/seattle.linear_regression.model")


# In[32]:


loaded_model = load("data/seattle.linear_regression.model", trusted=True)


# In[33]:


assert (model.coef_== loaded_model.coef_).all()

assert model.intercept_  == loaded_model.intercept_
assert model.rank_       == loaded_model.rank_
assert model.score(X, y) == loaded_model.score(X, y)


# In[34]:


import matplotlib.pyplot as plt


# In[35]:


Xnew = tr[features]
Ynew = model.predict(Xnew)
tr['predicted'] = Ynew


# In[36]:


Xnew = te[features]
Ynew = model.predict(Xnew)
te['predicted'] = Ynew


# In[37]:


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




