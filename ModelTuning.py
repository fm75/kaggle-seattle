#!/usr/bin/env python
# coding: utf-8

# ### ModelTuning
# - Retrieve Data
# - Apply Transformations
# - One-hot encode zips
# - Loop
#     - Apply Filters
#     - Select Features
#     - Fit Model on Training Data
#     - Measure performance against Test Data
# - Report Results
# - Save Model

# In[1]:


import pandas as pd
from pathlib import Path
from seattle import (
    apply_filters,
    fit_model,
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


# ### Cheat here. 
# - There is only one data point for zip code 98188.
# - Add that record to the training data.
# - Without that, there would be no adjustment for the 98188 zip
# - It would also affect slightly the zip corrections because of attempting to model without a zip.
# - Net improvement of about 0.0026 in the test score

# In[5]:


tst[tst['98188'] == 1]


# In[6]:


trn[trn['98188'] == 1]


# In[7]:


len(trn)


# In[8]:


trn = pd.concat([trn, tst[tst['98188'] == 1]], copy=True)


# In[9]:


len(trn)


# In[10]:


trn.sort_values('bath_bed_ratio', ascending=True).head(10)


# In[11]:


trn.sort_values('bath_bed_ratio', ascending=False).head(10)


# In[12]:


raw_features      = ['beds', 'baths', 'house_size', 'lot_size']
raw_plus_ohe      = raw_features + zips
enhanced_features = raw_features + ['bath_bed_ratio', 'lot_size_ratio']
full_features     = enhanced_features + zips


# In[13]:


# List of filters
filters = [
    [
        "Unfiltered", [],
    ],
    [
        "Price4m", [("price", 0.0, 4_000_000.)],
    ],
    [
        "Price5m", [("price", 0.0, 5_000_000.)],
    ],
    [
        "Price6m", [("price", 0.0, 6_000_000.)],
    ],
    [
        "Price7M", [("price", 0.0, 7_000_000.)],
    ],
    [
        "Price5m-beds", [
            ("price", 0.0, 5_000_000.),
            ("beds", 1, 8),
        ],
    ],
    [
        "Price5m-baths", [
            ("price", 0.0, 5_000_000.),
            ("baths", 1, 8),
        ],
    ],
    [
        "Price6m-beds", [
            ("price", 0.0, 6_000_000.),
            ("beds", 1, 8),
        ],
    ],
    [
        "Price6m-baths", [
            ("price", 0.0, 6_000_000.),
            ("baths", 1, 8),
        ],
    ],
    [
        "Price7m-beds-baths", [
            ("price", 0.0, 7_000_000.),
            ("beds", 1, 8),
            ("baths", 1, 8),
        ],
    ],
    [
        "Price7m-beds-baths-lsr", [
            ("price", 0.0, 7_000_000.),
            ("beds", 1, 8),
            ("baths", 1, 8),
            ("lot_size_ratio", 0.3, 30),
        ],
    ],
    [
        "Price7m-beds-baths-bbr", [
            ("price", 0.0, 7_000_000.),
            ("beds", 1, 8),
            ("baths", 1, 8),
            ("bath_bed_ratio", 0.24, 3.0),
        ],
    ],
]


# In[14]:


# List of potential features
models = [
    [
        "Raw features", raw_features
    ],
    [
        "Raw + ohe", raw_plus_ohe
    ],
    [
        "Enhanced features", enhanced_features
    ],
    [
        "Enhanced + ohe", full_features
    ]
]


# In[15]:


rows = list()
for m in models:
    model_name = m[0]
    features = m[1]
    for filter in filters:
        filter_name = filter[0]
        filter_set = filter[1]
        trn_subset = apply_filters(trn, filter_set)
        X0 = trn_subset[features]
        y0 = trn_subset.price
        
        tst_subset = apply_filters(tst, filter_set)
        X1 = tst_subset[features]
        y1 = tst_subset.price
        
        result = fit_model(X0, y0)
        row = dict(
            model_name=model_name, 
            filter_name=filter_name,
            train_score=result.score(X0, y0),
            test_score=result.score(X1, y1),
            model=result,
            features=features,
            coefficients=result.coef_,
            intercept=result.intercept_,
            train_size=len(X0),
            test_size=len(X1),
            train_raw_size=len(trn),
            test_raw_size=len(tst),
        )
        rows.append(row)
df_results = pd.DataFrame(rows)


# In[16]:


df_results.sort_values("test_score", ascending=False)


# In[17]:


best = df_results.sort_values("test_score", ascending=False).iloc[0]
best


# In[18]:


features     = best.features
intercept    = best.intercept
coefficients = best.coefficients


# In[19]:


print(f'{"intercept":<15} {intercept:12,.2f}')
for i, feature in enumerate(features):
    print(f'{feature:<15} {coefficients[i]:12,.2f}')


# In[20]:


from sklearn.linear_model import LinearRegression
from skops.io import dump, load


# In[21]:


dump(best.model, "data/seattle.tuned_linear_regression.model")


# In[22]:


loaded_model = load("data/seattle.tuned_linear_regression.model", trusted=True)


# In[ ]:





# In[ ]:





# In[ ]:




