import numpy as np
import pandas as pd
from typing import List, Callable


def normalize_lot_size(df: pd.DataFrame):
    """Convert lot size to square feet if the lot_size_units is 'acre'.
    """
    
    df.loc[df.lot_size_units == 'acre', 'lot_size'] = 43_560 * df.lot_size
    df.drop("lot_size_units", axis=1, inplace=True)

    
def normalize_house_size(df: pd.DataFrame):
    """Convert lot size to square feet if the lot_size_units is 'acre'.
    Rename the column "size" to "house_size".
    """

    df.loc[df.size_units == 'acre', 'size'] = 43_560 * df["size"]
    df.rename(columns={"size": "house_size"}, inplace=True)
    df.drop("size_units", axis=1, inplace=True)

    
def baths_to_beds_ratio(df: pd.DataFrame):
    """Create bath_bed_ratio column"""
    df["bath_bed_ratio"] = df.baths / df.beds

    
def lot_size_factor(df: pd.DataFrame):
    """Create bath_bed_ratio column"""
    df["lot_size_ratio"] = df.lot_size / df.house_size

    
def apply_transformations(df: pd.DataFrame, transformations: List[Callable]):
    """Serially, apply a list of inplace transformations to the df DataFrame."""
    for transformation in transformations:
        transformation(df)


def get_unique_category_values(dfs: List[pd.DataFrame], category: str) -> List[str]:
    """Create a sorted list of category values across multiple DataFrames.
    This is useful in case where a training dataset and/or a test dataset might not
    have a complete set of all possible category values.
    """
    xlist = np.ndarray((0,))
    for df in dfs:
        z = df[category].unique()
        xlist = [x for x in list(np.unique(np.concatenate((xlist, z), axis=0)))]
    return sorted(xlist)


def ohe(df: pd.DataFrame, ohe_category: str, unique_values: List[str]) -> pd.DataFrame:
    """One-hot encode a category in a DataFrame across a full list of unique_values,
    some of which could possibly be missing in the DataFrame. Then remove the original
    column.
    This is not an inplace transformation.
    """
    cols = df.columns.to_list() + unique_values
    dft = pd.get_dummies(df, prefix="", prefix_sep="", columns=[ohe_category])
    dft = dft.reindex(columns=cols).fillna(0)
    dft.drop(ohe_category, axis=1, inplace=True)
    return dft