import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from typing import List, Tuple


# def one_hot_encode_zips(df1: pd.DataFrame, df2: pd.DataFrame) -> (
#         pd.DataFrame, 
#         pd.DataFrame,
#         List[str],
#         List[str],
#         ):
#     """
#     One-hot encode the zip codes (create a "0/1" column for each zip code and 
#     populate it with one for the correct zip for each row).
#     Return **copies** of the DataFrames so that the raw data can be reused with
#     different models and filtering strategies easily.
#     Also return a list of quantitative feature columns and a list of the zips to
#     be used by the models when training.
#     """
    
#     # First we need to deal with the fact that neither the training nor the test data
#     # have a full set of the Seattle Housing zip codes.
#     z1 = df1.zip_code.unique()
#     z2 = df2.zip_code.unique()
    
#     # Note - we won't need to convert zips from int to str if we do it when
#     # populating the DataFrame with read_csv().
#     zips = sorted([str(x) for x in list(np.unique(np.concatenate((z1,z2), axis=0)))])
    
#     # We need a full set up the columns, less the "zip_code" column, which will be
#     # removed by the get_dummies one-hot encoding call. This set of column names,
#     # cols, will be used to add NA values where we have no zip codes in the 
#     # reindex operation.
#     qfeats = df1.columns.to_list()
#     cols = qfeats + zips

#     f1 = pd.get_dummies(df1, prefix="", prefix_sep="", columns=["zip_code"])
#     f2 = pd.get_dummies(df2, prefix="", prefix_sep="", columns=["zip_code"])

#     cols.remove('zip_code')

#     f1 = f1.reindex(columns=cols).fillna(0)
#     f2 = f2.reindex(columns=cols).fillna(0)
    
#     # qfeats = quantitative feature columns
#     # zips is the one-hot encoded columns for zip codes.
#     qfeats.remove('zip_code')
#     return (f1, f2, qfeats, zips)


def get_train_test_data() -> (pd.DataFrame, pd.DataFrame):
    """Return DataFrames with train and test data"""
    train = pd.read_csv("data/transform_train.csv")
    test  = pd.read_csv("data/transform_test.csv")
    return (train, test)


def get_train_test_data_with_nan() -> (pd.DataFrame, pd.DataFrame):
    """Return DataFrames with train and test data"""
    train = pd.read_csv("data/transform_with_nan_train.csv")
    test  = pd.read_csv("data/transform_with_nan_test.csv")
    return (train, test)


def apply_filters(df: pd.DataFrame,
        filters: List[Tuple[str, float, float]]
    ) -> pd.DataFrame:
    """
    Input:
        DataFrame to be filtered
        List of (Name: str, min: float, max: float) min, max inclusive
    Returns a filtered version of the input DataFrame
    """
    dfc = df.copy()
    for feature in filters:
        fcol = feature[0]
        fmin = feature[1]
        fmax = feature[2]
        dfc = dfc[(dfc[fcol] >= fmin) & (dfc[fcol] <= fmax)]
    
    return dfc


def evaluate_model(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    features: List[str],
    response: str
    ) -> (float, float, List[float], float):
    """Run linear regression on training data. Apply the fit to the test data.
    features is a list of names of the independent variables.
    response is the name of the dependent variable.
    Return fit score, test score, coefficient and y intercept.
    """
    if response in features:
        features.remove(response)
    # Training result
    X = df_train[features]
    y = df_train[response]
    model = LinearRegression().fit(X, y)
    
    # Test score
    X_test = df_test[features]
    y_test = df_test[response]

    return (model.score(X, y), model.score(X_test, y_test), model.coef_, model.intercept_)


def fit_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Determine linear regression parameters and return the full model.
    """
    return LinearRegression().fit(X, y)
