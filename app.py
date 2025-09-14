import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from main import *
from training_time import *

data = pd.read_csv('data/final.csv')
data = data.dropna()

features = ['Carbon-free energy percentage (CFE%)', 'Renewable energy percentage (RE%)', 'hour', 'day_of_week', 'is_summer', 'is_weekend','month', 'hour_cos', 'hour_sin']
target = 'carbon_intensity'

x = data[features]
y = data[target]

split_idx = int(len(data) * 0.8)

x_train = x.iloc[:split_idx]
x_test = x.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

training_set = pd.read_csv('data/final.csv')
training_set['Datetime'] = pd.to_datetime(training_set['Datetime'], format='%Y-%m-%d %H:%M:%S')

st.sidebar.header("Model Configuration")
GPU_SPECS = {
    "NVIDIA A100 (80GB)": {
        "flops": 312e12,
        "memory": 80,
        "power_watts": 400,
        "display_name": "A100 (80GB) - 400W"
    },
    "NVIDIA H100 (80GB)": {
        "flops": 989e12,
        "memory": 80,
        "power_watts": 700,
        "display_name": "H100 (80GB) - 700W"
    },
    "NVIDIA A100 (40GB)": {
        "flops": 312e12,
        "memory": 40,
        "power_watts": 400,
        "display_name": "A100 (40GB) - 400W"
    },
    "NVIDIA RTX 4090": {
        "flops": 165e12,
        "memory": 24,
        "power_watts": 450,
        "display_name": "RTX 4090 (24GB) - 450W"
    },
    "NVIDIA RTX 4080": {
        "flops": 120e12,
        "memory": 16,
        "power_watts": 320,
        "display_name": "RTX 4080 (16GB) - 320W"
    },
    "NVIDIA RTX 4070": {
        "flops": 90e12,
        "memory": 12,
        "power_watts": 200,
        "display_name": "RTX 4070 (12GB) - 200W"
    }
}

# Streamlit dropdown
selected_gpu_key = st.sidebar.selectbox(
    "GPU Type",
    options=list(GPU_SPECS.keys()),
    format_func=lambda x: GPU_SPECS[x]["display_name"]
)

# Get the selected GPU specs
selected_gpu = GPU_SPECS[selected_gpu_key]
# Model parameters
num_parameters = st.sidebar.number_input(
    "Number of Parameters (billions)", 
    min_value=0.1, 
    max_value=1000.0, 
    value=7.0, 
    step=0.1,
    format="%.1f"
)

# Training data
num_tokens = st.sidebar.number_input(
    "Training Tokens (billions)", 
    min_value=1.0, 
    max_value=10000.0, 
    value=20.0 * num_parameters, 
    step=10.0,
    format="%.1f"
)

num_gpus = st.sidebar.slider(
    "Number of GPUS", 
    min_value=1, 
    max_value=128, 
    value=32
)

flops = num_tokens * num_parameters * 6
use_chinchilla = st.sidebar.checkbox("Use Chinchilla tokens (≈20× params)", value=True)
if use_chinchilla:
    num_tokens = 20.0 * num_parameters

if st.sidebar.button("Calculate Training Time"):
    actual_params = int(num_parameters * 1e9)
    actual_tokens = int(num_tokens * 1e9)
    total_flops = 6 * actual_params * actual_tokens

    tt = TrainTime(
        params=actual_params,
        gpu_memory_gb=selected_gpu['memory'],
        precision="bf16",
        optimizer="adam",
        zero_stage=2,
        activation_checkpoint=True,
        base_eff=0.35
    )
    result = tt.estimate_time(total_flops, selected_gpu['flops'], num_gpus=num_gpus)
    st.session_state.result = result

    st.sidebar.success(f"Estimated: {result['hours']:.2f} hours")
    st.sidebar.write(f"• Total FLOPs: {total_flops:,.3e}")
    st.sidebar.write(f"• Cluster rate: {result['cluster_flops_per_s']:,.3e} FLOPs/s")
    st.sidebar.write(f"• Effective efficiency: {result['efficiency_used']:.3f}")
    st.sidebar.write(f"• Used VRAM/GPU: {result['used_vram_gb']} GB "
                        f"(headroom {100*result['headroom']:.1f}%)")

if 'result' in st.session_state:
    st.header("📅 Select Analysis Period")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2026, 1, 15).date(),
            min_value=datetime(2026, 1, 1).date(),
            max_value=datetime(2026, 12, 31).date()
        )

    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime(2026, 12, 15).date(),
            min_value=datetime(2026, 1, 1).date(),
            max_value=datetime(2026, 12, 31).date()
        )

    start_time_input = st.time_input("Start Time", value=datetime.strptime("08:00:00", "%H:%M:%S").time())
    end_time_input = st.time_input("End Time", value=datetime.strptime("20:00:00", "%H:%M:%S").time())

    start_datetime = datetime.combine(start_date, start_time_input).strftime('%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.combine(end_date, end_time_input).strftime('%Y-%m-%d %H:%M:%S')

    if st.button("🔄 Update Analysis Period") or 'df' not in st.session_state:
        with st.spinner("Loading carbon intensity predictions..."):
            df = load_features(start_datetime, end_datetime)
            df = add_energy(df, training_set)
            df = predict_carbon_intensity(df, model)
            
            st.session_state.df = df
            st.success(f"Analysis updated for {start_date} to {end_date}")

    df = st.session_state.df
    st.info(f"📊 Showing predictions from **{start_datetime}** to **{end_datetime}**")

    st.plotly_chart(heatmap(df), use_container_width=True)
    result = st.session_state.result
    try:
        optimal_windows = find_optimal_windows(df, window_hours=int(result['hours']))
        st.subheader("Recommended Training Windows")
        st.write(f"Here are the 3 best low-carbon periods:")

        for i, (start_time, avg_carbon) in enumerate(optimal_windows[:3], 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Window {i}:** {start_time.strftime('%B %d, %Y at %I:%M %p')}")
                gpu_emissions_saved = calculate_gpu_emissions(result['hours'], avg_carbon, selected_gpu['power_watts'])
                st.write(f"Carbon Emission at this Time: {gpu_emissions_saved:.2f} kg of CO2")
                time_now = datetime.now()
                time_now = time_now.replace(year=2026, minute=0, second=0, microsecond=0)
                print(time_now)
                carbon_now = df.query(f"Datetime == @time_now")['predicted_carbon_intensity'].iloc[0]
                gpu_now = calculate_gpu_emissions(result['hours'], carbon_now, selected_gpu['power_watts'] )
                st.write(f"Carbon Emission today: {gpu_now:.2f} kg of CO2")
            with col2:
                st.metric("Avg Carbon", f"{avg_carbon:.1f} gCO2/kWh")
            with col3:
                end_time = start_time + pd.Timedelta(hours=int(result['hours']))
                st.write(f"Ends: {end_time.strftime('%B %d, %Y at %I:%M %p')}")
                st.write(f"{(((gpu_now - gpu_emissions_saved)/gpu_now) * 100):.2f}% Better")
            
            st.divider()

    except OverflowError:
        st.write("Error: Given parameters are are either too large for memory or improbable for calculation")
    

    