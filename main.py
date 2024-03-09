import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv("C:\\Users\\mumtaz\\OneDrive\\Desktop\\internship project\\House Price prediction\\bengaluru_house_prices.csv")
print(data.head())
print(data.shape)
print(data.info())

data.isna().sum()
data.drop(columns=['area_type', 'availability', 'society'],inplace=True)
data.describe()
data.info()

data['location'].value_counts()
data['location'] = data['location'].fillna('Sarjapur Road')
data['size'].value_counts()
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
data['balcony'] = data['balcony'].fillna(data['balcony'].median())
data['size'] = data['size'].astype(str)
data['bhk'] = data['size'].str.split().str.get(0).astype(float)
data = data[data['bhk'] <= 20]
data.info()

def convertRange(x):
    if '_' in str(x):
        temp = x.split('_')
        return(float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
data['total_sqft'] = data['total_sqft'].apply(convertRange)
data['price_per_sqft']= data['price']*100000 / data['total_sqft']
data['price_per_sqft']
data.describe()

data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()

location_count_less_10 = location_count[location_count <= 10]

data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)

data = data[((data['total_sqft']/data['bhk']) >= 300 )]

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        gen_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

data = remove_outliers_sqft(data)

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    bhk_stats = {}
    for location, location_df in df.groupby('location'):
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }

    for bhk, bhk_df in data.groupby('bhk'):
        stats = bhk_stats.get(bhk - 1)
        if stats and stats['count'] > 5:
            exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < (stats['mean'])].index.values)

    return df.drop(exclude_indices, axis='index')

data = bhk_outlier_remover(data)

data.drop(columns=['size', 'price_per_sqft'], inplace=True)
data.head()


data.to_csv("Cleaned_data.csv")
X = data[['total_sqft', 'bath', 'balcony', 'bhk']]
y = data['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=69)
_, X_test, _, y_test = train_test_split(X,y, test_size=0.9, random_state=420)

print(X_train.shape)
print(X_test.shape)


# Create a preprocessor for numerical and categorical features
numeric_transformer = numeric_transformer = make_pipeline(StandardScaler(), PolynomialFeatures(degree=1))
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = make_column_transformer((numeric_transformer, ['total_sqft', 'bath', 'balcony']),
                                       (categorical_transformer, ['bhk']))

linear_pipeline = make_pipeline(preprocessor, LinearRegression())
lasso_pipeline = make_pipeline(preprocessor, Lasso())
ridge_pipeline = make_pipeline(preprocessor, Ridge())

linear_pipeline.fit(X_train, y_train)
linear_pred = linear_pipeline.predict(X_test)
print('Linear Regression R^2:', r2_score(y_test, linear_pred))

lasso_pipeline.fit(X_train, y_train)
lasso_pred = lasso_pipeline.predict(X_test)
print('Lasso Regression R^2:', r2_score(y_test, lasso_pred))

ridge_pipeline.fit(X_train, y_train)
ridge_pred = ridge_pipeline.predict(X_test)
print('Ridge Regression R^2:', r2_score(y_test, ridge_pred))

with open('ridge_model.pkl', 'wb') as file:
    pickle.dump(ridge_pipeline, file)