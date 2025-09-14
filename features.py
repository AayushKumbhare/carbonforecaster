import pandas as pd
import numpy as np
from datetime import datetime

#----------Dataframing----------
data = pd.read_csv('/Users/aayushkumbhare/Desktop/carbon-footprint/data/data.csv')

data = data.dropna(subset=['carbon_intensity'])

#----------Feature Engineering----------
def change_to_datetime(data):
    data['datetime'] = pd.to_datetime(data['Datetime (UTC)'])

def create_time_features(data):
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['is_weekend'] = data['datetime'].dt.dayofweek >= 5

    return data

def lag_features(data):
    data['carbon_1hr_ago'] = data['carbon_intensity'].shift(1)
    data['carbon_1hr_ago'] = data['carbon_1hr_ago'].bfill()
    
    return data

def seasonal_features(data):
    data['is_summer'] = False
    for i, month in enumerate(data['month']):
        if 6 <= month <= 8:
            data.loc[i, 'is_summer'] = True

    return data


def trig_time(data):
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    return data


def carbon_trend(data):
    data['carbon_trend'] = data['carbon_intensity'] - data['carbon_1hr_ago']
    data['trend_magnitude'] = abs(data['carbon_trend'])
    data['is_stable'] = abs(data['carbon_trend']) < 10

    return data


def mean_and_prev_day(data):
    data['carbon_24hr_ago'] = data['carbon_intensity'].shift(24)
    data['carbon_24hr_ago'] = data['carbon_24hr_ago'].bfill()
    data['carbon_avg_3hr'] = data['carbon_intensity'].rolling(window=3).mean()

    return data


def create_all_features(data):
    lag_features(data)
    seasonal_features(data)
    trig_time(data)
    carbon_trend(data)
    mean_and_prev_day(data)
    return data

def test_features(data):
    print(f"Dataset shape: {data.shape}")
    print(f"Feature columns: {list(data.columns)}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

data = create_all_features(data)
data.to_csv('/Users/aayushkumbhare/Desktop/carbon-footprint/data/output.csv')