import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet')
df_jan2023 = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
df_feb2023 = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')

######Question 1
print('Q1: ', len(df_jan2023.columns)) 

######Question 2
df_jan2023['duration'] = df_jan2023['tpep_dropoff_datetime'] - df_jan2023['tpep_pickup_datetime']
df_jan2023.duration = df_jan2023.duration.apply(lambda td: td.total_seconds() / 60)
print('Q2: ', df_jan2023.duration.std())

df_feb2023['duration'] = df_feb2023['tpep_dropoff_datetime'] - df_feb2023['tpep_pickup_datetime']
df_feb2023.duration = df_feb2023.duration.apply(lambda td: td.total_seconds() / 60)

######Question 3
df_jan2023_filt = df_jan2023[(df_jan2023.duration >= 1) & (df_jan2023.duration <= 60)]
perc = 100 - ((len(df_jan2023.axes[0]) - len(df_jan2023_filt.axes[0]))/len(df_jan2023.axes[0]) *100)
print('Q3: ', perc)

df_feb2023_filt = df_feb2023[(df_jan2023.duration >= 1) & (df_feb2023.duration <= 60)]

######Question 4
categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

df_jan2023_filt[categorical] = df_jan2023_filt[categorical].astype(str)
df_feb2023_filt[categorical] = df_feb2023_filt[categorical].astype(str)

train_dicts = df_jan2023_filt[categorical + numerical].to_dict(orient='records')
val_dicts = df_feb2023_filt[categorical + numerical].to_dict(orient='records')

df_jan2023_filt['PU_DO'] = df_jan2023_filt['PULocationID'] + '_' + df_jan2023_filt['DOLocationID']
df_feb2023_filt['PU_DO'] = df_feb2023_filt['PULocationID'] + '_' + df_feb2023_filt['DOLocationID']

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

target = 'duration'
y_train = df_jan2023_filt[target].values
y_val = df_feb2023_filt[target].values

print('Q4: ', X_train)

######Question 5
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

print('Q5: ', mean_squared_error(y_train, y_pred, squared=False))

######Question 6
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

print('Q6: ', mean_squared_error(y_val, y_pred, squared=False))

lr = Lasso(0.01)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

print('Q6: ', mean_squared_error(y_val, y_pred, squared=False))