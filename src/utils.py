"""
General utilities for model and data handling in energy demand forecasting.
Includes functions for PyTorch prediction, data splitting, and value unscaling.
"""
import torch
from sklearn.model_selection import train_test_split
import numpy as np

def predict_pytorch(model, X):
    """
    Makes predictions with a PyTorch model and returns results as a numpy array.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X (array-like): Input data for prediction.
    Returns:
        np.ndarray: Model predictions.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred = model(X_tensor).cpu().numpy().flatten()
    return y_pred

def split_train_val(X_train, y_train, val_size=0.2, random_state=42):
    """
    Splits training data into training and validation sets.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        val_size (float): Proportion of data for validation (default 0.2).
        random_state (int): Random seed (default 42).
    Returns:
        tuple: X_train_final, X_val, y_train_final, y_val
    """
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    return X_train_final, X_val, y_train_final, y_val

def desescalar_y(scaler, y, target_col_index):
    """
    Unscales a vector of target values using the original scaler.

    Args:
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler.
        y (np.ndarray): Scaled values to unscale.
        target_col_index (int): Index of the target column in the scaler.
    Returns:
        np.ndarray: Unscaled values in original scale.
    """
    dummy_array = np.zeros((len(y), scaler.n_features_in_))
    dummy_array[:, target_col_index] = y.flatten()
    return scaler.inverse_transform(dummy_array)[:, target_col_index]
