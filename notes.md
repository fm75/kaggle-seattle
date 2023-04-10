# Notes on Seattle House Pricing Data

## Raw Content
- [Data Source](https://www.kaggle.com/datasets/samuelcortinhas/house-price-prediction-seattle)
- CSV files 
- training data - 2016 records
- testing data  =  505 records
Explored in an [exploration notebook](exploration.ipynb)

### Data

| Field | Description
|:---|:---|
| beds | bedroom count
| baths | bathroom count
| size  | size of house (all in sq ft)
| size_units | sqft for all records
| lot_size | area of property (possibly NA)
| lot_size_units | sqft, acre, or NA
| zip_code | zip code of house
| price | house sale prices, August - December 2022

## Exploration Notes - Numerical Features
| Field | Min | 25% | 50% | 75% | Max | Mean | 
| :--- | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | 
| beds (train) | 1 | 2 | 3 | 4 | 15 | 2.86
| beds (test) | 1 | 2 | 3 | 4 | 15 | 2.95
| baths (train) | 0.5 | 1.5 | 2 | 2.5 | 9 | 2.16
| baths (test) | 1 | 1.5 | 2 | 2.5 | 7 | 2.22
| size (train) | 250 | 1069 | 1560 | 2222 | 11010 | 1735
| size (test) | 376 | 1171 | 1690 | 2400 | 6139 | 1852

## Additional Post-transformed Data
Rows with missing data (lot size, lot size unit) deleted

| Data | Rows (1) | Rows (2)
| --- | --- | ---
| Training rows | 2016 | 1669
| Testing rows  |  505 |  428
1. Raw Data
2. Rows with missing lot size deleted


| Field | Min | 25% | 50% | 75% | Max | Mean | 
| :--- | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| lot_sq_ft (train)* | 500 | 2734 | 5000 | 7389 | 10,890,000| 18,790
| lot_sq_ft (test)* | 529 | 3479 | 5011 | 7500 | 176,854 | 8,961
| bed_bath_diff (train) | -4.0 | 0.0 | 1.0 | 1.5 | 9.0 | 0.83
| bed_bath_diff (test) | -2.5 | 0.0 | 1.0 | 1.5 | 3.5 | 0.85
| bed_bath_ratio (train) | 0.3 | 1.0 | 1.3 | 2.0 | 6.0 | 1.5
| bed_bath_ratio (test) | 0.3 | 1.0 | 1.3 | 1.8 | 4.0 | 1.5
| lot_size_ratio (train) | 0.34 | 1.33 | 2.33 | 4.38 | 43,560 | 38.00
| lot_size_ratio (test) | 0.40 | 1.47 | 2.34 | 4.18 | 255 | 6.50



\* From transforming all lot size data to square feet

## Categorical Data
If zip codes are to be used in the model, they must be categorical and unordered. They will be one-hot encoded, i.e. for each possible zip code in the data a row will be encode as a 1 for its zip and as a 0 for all other zips. 

This encoding takes place in the [model with zip codes notebook](ModelWithZipCodes.ipynb)