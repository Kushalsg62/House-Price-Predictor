import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def preprocess_data(house_data):
    # The same preprocessing steps you have in your notebook
    house_data = house_data.drop(['area_type', 'availability', 'balcony', 'society'], axis=1)
    house_data = house_data.dropna()
    house_data['BHK'] = house_data['size'].apply(lambda x: int(x.split(' ')[0]))
    house_data['total_sqft'] = house_data['total_sqft'].apply(convert_sqft_num)
    house_data = house_data.copy()
    house_data['price_per_sqft'] = house_data['price'] * 100000 / house_data['total_sqft']
    house_data.location = house_data.location.apply(lambda x: x.strip())
    location_stats = house_data.groupby('location')['location'].count().sort_values(ascending=False)
    locationlessthan10 = location_stats[location_stats <= 10]
    house_data.location = house_data.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
    house_data = house_data[~(house_data.total_sqft / house_data.BHK < 300)]
    house_data = remove_pps_outliers(house_data)
    house_data = remove_bhk_outliers(house_data)
    house_data = house_data[house_data.bath < house_data.BHK + 2]
    house_data = house_data.drop(['size', 'price_per_sqft'], axis='columns')
    dummies = pd.get_dummies(house_data.location)
    house_data = pd.concat([house_data, dummies.drop('other', axis='columns')], axis='columns')
    house_data = house_data.drop('location', axis='columns')
    X = house_data.drop('price', axis='columns')
    y = house_data.price
    return house_data, X, y

def convert_sqft_num(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft < (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for BHK, BHK_df in location_df.groupby('BHK'):
            bhk_stats[BHK] = {
                'mean': np.mean(BHK_df.price_per_sqft),
                'std': np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0]
            }
        for BHK, BHK_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(BHK - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, BHK_df[BHK_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

def train_model(X, y):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "XGB Regressor": XGBRegressor(),
        "Extra Trees Regressor": ExtraTreesRegressor()
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        model.fit(X, y)
        r2 = model.score(X, y)
        if r2 > best_score:
            best_model = model
            best_score = r2

    return best_model

def price_predict(location, sqft, bath, BHK, model, X):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]
