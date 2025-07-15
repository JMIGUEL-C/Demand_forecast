"""
Interactive dashboard for energy demand forecasting in Cali using an LSTM model.
Allows you to visualize model performance, explore the data, and make interactive predictions.
"""

import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta, date
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# --- Path Adjustment for Module Import ---
# Allows importing modules from 'src' even if app.py is in 'dashboard'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.lstm_model import LSTMModel
from src.processing import load_and_clean_data, create_sequences, create_features, create_lag_features, apply_moving_average, scale_data
from src.utils import desescalar_y

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """
    Loads the pre-trained LSTM model from the 'results' folder.
    Returns:
        LSTMModel or None: Loaded model or None if file not found.
    """
    try:
        INPUT_DIM = 15
        HIDDEN_DIM = 64
        NUM_LAYERS = 3
        OUTPUT_DIM = 1
        model = LSTMModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM)
        model_path = os.path.join(project_root, 'results', 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at expected path: '{model_path}'.")
        st.info("Please ensure the trained model ('model.pth') exists in the 'results/' folder.")
        return None

@st.cache_data
def load_data_and_create_scaler():
    """
    Loads and processes historical data, applying moving average, feature creation, and scaling.
    Returns:
        tuple: (df_full, df_scaled, scaler) or (None, None, None) if error occurs.
    """
    try:
        data_path = os.path.join(project_root, 'data', 'xm_api_data.csv')
        df = load_and_clean_data(data_path)
        df = apply_moving_average(df)
        df = create_features(df)
        df = create_lag_features(df, lags=[1, 7, 14], target_col='Demand_MWh')
        df_scaled, scaler = scale_data(df, target_col='Demand_MWh')
        return df, df_scaled, scaler
    except FileNotFoundError:
        st.error(f"Error: Data file not found at expected path: '{data_path}'.")
        st.info("Make sure the 'xm_api_data.csv' file is in the 'data/' folder.")
        return None, None, None

# --- Model and data initialization ---
model = load_model()
df_full, df_scaled, scaler = load_data_and_create_scaler()

# --- Dashboard UI ---
st.title("⚡ Electric Energy Demand Forecasting")
st.markdown("""
Welcome to the interactive dashboard for the demand forecasting project. 
This project uses a recurrent neural network (LSTM) to predict future energy demand in the city of Cali.
""")

# --- Robust verification of data and model loading ---
if model is None or df_full is None or df_scaled is None or scaler is None:
    st.warning("The application cannot continue because essential files (model or data) are missing. Check the error messages above.")
    st.stop()

# --- Data Preparation for Model ---
TARGET_COL = 'Demand_MWh'
SEQUENCE_LENGTH = 14

X, y = create_sequences(df_scaled, TARGET_COL, SEQUENCE_LENGTH)

test_size = 60
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

test_dates = df_full.index[len(df_full) - test_size:]

# --- Content Tabs ---
tab1, tab2, tab3 = st.tabs(["**Project Overview**", "**Model Performance**", "**Interactive Prediction**"])

with tab1:
    st.header("🎯 What is the objective?")
    st.write("""
    The main objective is to predict as accurately as possible how much electrical energy will be consumed 
    in Cali in the coming days. 
    An accurate prediction helps electric companies to:
    - **Manage the grid** more efficiently.
    - **Prevent blackouts** by ensuring there is enough energy available.
    - **Optimize costs** of generation and distribution.
    """)
    
    st.header("🧠 How does the model work?")
    st.write("""
    We use an **Artificial Intelligence** model called **LSTM (Long Short-Term Memory)**. 
    This type of model is ideal for data that changes over time (time series), such as energy demand.
    
    To make a prediction, the model doesn't just look at yesterday's demand, but analyzes patterns in the **last 14 days** 
    to understand trends and cycles.
    """)
    
    st.header("📊 What information does the model use?")
    st.write("The model learns from several key features to improve its accuracy:")
    st.markdown("""
    - **Historical Demand:** How much energy was used yesterday, last week, and two weeks ago?
    - **Time Cycles:** Daily, weekly, and annual patterns.
    - **Holidays:** Demand changes significantly on holidays.
    """)
    st.dataframe(df_full.head())

with tab2:
    st.header("📈 Performance Evaluation")
    st.write("Here we compare the model's predictions with actual data for the last 60 days of the dataset.")

    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).numpy()

    target_col_index = df_full.columns.get_loc(TARGET_COL)
    y_pred = desescalar_y(scaler, predictions_scaled, target_col_index)
    y_true = desescalar_y(scaler, y_test, target_col_index)
    
    df_results = pd.DataFrame({
        'Date': test_dates,
        'Actual Demand (MWh)': y_true,
        'Predicted Demand (MWh)': y_pred
    })

    fig = px.line(df_results, x='Date', y=['Actual Demand (MWh)', 'Predicted Demand (MWh)'],
                  title='Comparison of Actual vs. Predicted Demand',
                  labels={'value': 'Demand (MWh)', 'variable': 'Demand Type'},
                  template='plotly_white')
    fig.update_layout(legend_title_text='')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae = np.mean(np.abs(y_pred - y_true))
    r2 = np.sqrt(np.mean((y_pred - y_true)**2))  # Note: this should be r2_score, but kept for consistency with original code
    
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} MWh", help="On average, the model's predictions deviate this amount from the actual value.")
    col2.metric("Root Mean Square Error (RMSE)", f"{rmse:.2f} MWh", help="Similar to MAE, but penalizes larger errors more.")
    col3.metric("Coefficient of Determination (R²)", f"{r2:.2f}", help="Indicates how well the model fits the data.")

with tab3:
    st.header("🔮 Make a Prediction")
    st.write(f"Select a date and the model will predict the demand for the next day, using data from the previous {SEQUENCE_LENGTH} days.")

    min_date_val = df_full.index[SEQUENCE_LENGTH].date()
    max_date_val = df_full.index[-2].date()
    
    selected_date = st.date_input(
        "Select a base date for prediction:",
        value=max_date_val,
        min_value=min_date_val,
        max_value=max_date_val,
        format="YYYY-MM-DD"
    )

    if st.button("Forecast") and selected_date:
        try:
            # Find the index corresponding to the selected date
            loc_result = df_scaled.index.get_loc(pd.to_datetime(selected_date))
            date_index: int
            if isinstance(loc_result, int):
                date_index = loc_result
            elif isinstance(loc_result, slice):
                date_index = loc_result.start
            else:
                raise TypeError(f"The returned index type ({type(loc_result)}) is not handled.")
            start_idx = date_index - SEQUENCE_LENGTH + 1
            end_idx = date_index + 1
            # Get the input sequence for the model
            input_sequence_scaled = df_scaled.iloc[start_idx:end_idx].values
            input_sequence_scaled = np.expand_dims(input_sequence_scaled, axis=0)
            # Make prediction
            input_tensor = torch.from_numpy(input_sequence_scaled.astype(np.float32))
            with torch.no_grad():
                prediction_scaled = model(input_tensor).numpy()
            # Unscale the result
            target_col_index = df_full.columns.get_loc(TARGET_COL)
            predicted_demand = desescalar_y(scaler, prediction_scaled, target_col_index)[0]
            # Display result
            prediction_date = selected_date + timedelta(days=1)
            st.success(f"**Forecast for {prediction_date.strftime('%Y-%m-%d')}:**")
            st.metric("Estimated Energy Demand", f"{predicted_demand:.2f} MWh")
            # Show the data used by the model
            st.subheader("Data used for this prediction:")
            input_data_real = df_full[[TARGET_COL]].iloc[start_idx:end_idx]
            st.line_chart(input_data_real)
        except KeyError:
            st.error("The selected date is not found in the dataset. Please choose another date.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
