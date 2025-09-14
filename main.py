import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import streamlit as st
from datetime import datetime


def convert_to_datetime(date_string):
    date_format = "%Y-%m-%d %H:%M:%S"
    datetime_object = datetime.strptime(date_string, date_format)
    return datetime_object

def load_features(start_date, end_date):
    start_date = convert_to_datetime(start_date)
    end_date = convert_to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    df = pd.DataFrame(date_range, columns=['Datetime'])


    df['month'] = df['Datetime'].dt.month
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['hour'] = df['Datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_summer'] = df['month'].isin([6, 7, 8])

    return df

def add_energy(df, training_set):
    energy_lookup = {}
    for idx, row in training_set.iterrows():
        energy_lookup[row['Datetime']] = {
            'CFE': row['Carbon-free energy percentage (CFE%)'],
            'RE': row['Renewable energy percentage (RE%)']
        }

    df['Carbon-free energy percentage (CFE%)'] = np.nan
    df['Renewable energy percentage (RE%)'] = np.nan
    
    for idx, row in df.iterrows():
        lookup_date = row['Datetime'].replace(year=2024)
        
        if lookup_date in energy_lookup:
            df.at[idx, 'Carbon-free energy percentage (CFE%)'] = energy_lookup[lookup_date]['CFE']
            df.at[idx, 'Renewable energy percentage (RE%)'] = energy_lookup[lookup_date]['RE']
        else:
            df.at[idx, 'Carbon-free energy percentage (CFE%)'] = 35.0
            df.at[idx, 'Renewable energy percentage (RE%)'] = 28.0
    
    return df

def predict_carbon_intensity(df, model):
    # Get the feature columns your model expects (exclude Datetime)
    feature_columns = [
        'Carbon-free energy percentage (CFE%)', 
        'Renewable energy percentage (RE%)', 
        'hour', 
        'day_of_week', 
        'is_summer', 
        'is_weekend',
        'month', 
        'hour_cos', 
        'hour_sin'
    ]

    X = df[feature_columns]
    predictions = model.predict(X)
    df['predicted_carbon_intensity'] = predictions
    
    return df

def heatmap(df):
    pivot_df = df.pivot_table(
        index='hour',
        columns=df['Datetime'].dt.date,
        values='predicted_carbon_intensity'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(col) for col in pivot_df.columns],
        y=pivot_df.index,
        colorscale='RdYlGn_r',
        colorbar=dict(
            title="Carbon Intensity<br>(gCO2/kWh)",
            thickness=20,
            len=0.8
        ),
        hovertemplate="Date: %{x}<br>Hour: %{y}:00<br>Carbon: %{z:.1f} gCO2/kWh<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="🌱 Carbon Intensity Forecast - Optimal GPU Training Times",
            font=dict(size=20),
            x=0.07
        ),
        xaxis_title="Date",
        yaxis_title="Hour of Day",
        height=700,
        width=1200, 
        font=dict(size=12),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_yaxes(
        ticktext=[f"{h:02d}:00" for h in range(24)],
        tickvals=list(range(24))
    )
    
    return fig

def find_optimal_windows(df, window_hours=6):
    rolling_avg = df['predicted_carbon_intensity'].rolling(window=window_hours).mean()
    window_map = {}


    for i, avg_val in enumerate(rolling_avg):
        if pd.notna(avg_val) and i + window_hours - 1 < len(df):
            start_time = df.iloc[i]['Datetime']
            window_map[start_time] = rolling_avg.iloc[i]

    sorted_items = sorted(window_map.items(), key=lambda x: x[1])
    candidates = []
    candidates.append(sorted_items[0])

    for pair in sorted_items[1:]:
        valid = True
        for selected in candidates:
            time_diff = pair[0] - selected[0]
            if abs(time_diff.days) < 30:
                valid = False
                break
        
        if valid:
            candidates.append(pair)
            if len(candidates) >= 3:
                break

    return candidates

def calculate_gpu_emissions(training_hours, carbon_intensity_g_per_kwh, gpu_power_watts=400):
    gpu_power_kw = gpu_power_watts / 1000
    energy_kwh = gpu_power_kw * training_hours
    total_co2_grams = energy_kwh * carbon_intensity_g_per_kwh
    total_co2_kg = total_co2_grams / 1000
    
    return total_co2_kg
