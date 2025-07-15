"""
Data processing module for energy demand forecasting.
Includes functions to load, clean, transform, and scale data, as well as create features and sequences for time series models.
"""
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler


def load_and_clean_data(filepath):
    """
    Loads and cleans data from a CSV file. Converts demand from kWh to MWh,
    renames relevant columns, and applies daily resampling with mean.

    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame with demand and forecast columns in MWh, indexed by date.
    Raises:
        ValueError: If the 'Fecha' column is not present in the file.
    """
    df = pd.read_csv(filepath)
    if "Fecha" not in df.columns:
        raise ValueError("File does not contain 'Fecha' column. Found columns: " + str(df.columns))
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df.set_index("Fecha", inplace=True)
    # Convert demand from kWh to MWh
    df['Demand_MWh'] = df['DemandaAtendida'] / 1000
    df['PronDem_MWh'] = df['PronDem'] / 1000
    numeric_cols = ['Demand_MWh', 'PronDem_MWh']
    df_diario = df[numeric_cols].resample('D').mean().dropna()
    return df_diario


def apply_moving_average(df, window=7):
    """
    Applies moving average to all DataFrame columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        window (int): Moving average window (default 7).
    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """
    return df.rolling(window=window, min_periods=1).mean()


def create_features(df):
    """
    Creates temporal and calendar features from the date index.
    Includes cyclic encoding of day, week, and month, and marks holidays in Colombia.

    Args:
        df (pd.DataFrame): DataFrame indexed by date.
    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df['DayOfYear'] = df.index.dayofyear
    df['sin_dia'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['cos_dia'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfWeek'] = df.index.dayofweek
    df['sin_semana'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['cos_semana'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month'] = df.index.month
    df['sin_mes'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_mes'] = np.cos(2 * np.pi * df['Month'] / 12)
    # Mark holidays in Colombia
    co_holidays = holidays.Colombia()
    df['es_festivo'] = df.index.map(lambda x: 1 if x in co_holidays else 0)
    return df


def create_lag_features(df, lags, target_col='Demand_MWh'):
    """
    Creates lag features for the specified target.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lags (list): List of integers indicating the lags to create.
        target_col (str): Target column to create lags for.
    Returns:
        pd.DataFrame: DataFrame with new lag columns.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df


def scale_data(df, target_col='Demand_MWh'):
    """
    Scales data using sklearn's MinMaxScaler.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column (not used directly, but useful for consistency).
    Returns:
        tuple: (scaled DataFrame, fitted scaler)
    """
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    return df_scaled, scaler


def create_sequences(data, target_col='Demand_MWh', sequence_length=14):
    """
    Generates data sequences for time series models (e.g., LSTM).

    Args:
        data (pd.DataFrame): Input DataFrame.
        target_col (str): Target column to predict.
        sequence_length (int): Length of input sequence.
    Returns:
        tuple: (sequence array X, target array y)
    """
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length)].values
        y = data.iloc[i + sequence_length][target_col]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
