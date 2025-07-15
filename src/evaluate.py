"""
Evaluation and visualization module for energy demand forecasting models.
Includes functions to calculate metrics, save results, and plot predictions.
"""
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test, scaler, target_col_index):
    """
    Evaluates the model on the test set and calculates performance metrics.
    Unscales predictions and actual values before calculating metrics.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels (scaled).
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler for unscaling.
        target_col_index (int): Index of the target column in the scaler.
    Returns:
        dict: Dictionary with 'rmse', 'mae', and 'r2' metrics.
    """
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).numpy()
    # Unscale predictions
    dummy_array = np.zeros((len(predictions_scaled), scaler.n_features_in_))
    dummy_array[:, target_col_index] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, target_col_index]
    dummy_array_true = np.zeros((len(y_test), scaler.n_features_in_))
    dummy_array_true[:, target_col_index] = y_test.flatten()
    y_true = scaler.inverse_transform(dummy_array_true)[:, target_col_index]
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "mae": mean_absolute_error(y_true, predictions),
        "r2": r2_score(y_true, predictions)
    }
    return metrics

def save_metrics(metrics, filepath=None):
    """
    Saves evaluation metrics to a text file within the 'results' folder.
    If no path is specified, saves to 'results/metrics.txt'.

    Args:
        metrics (dict): Dictionary of metrics to save.
        filepath (str, optional): File path where to save the metrics.
    """
    # Get project root (folder containing 'src')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'results')
    if filepath is None or filepath == '':
        filepath = os.path.join(results_dir, 'metrics.txt')
    else:
        # If path is not absolute, make it relative to project root
        if not os.path.isabs(filepath):
            filepath = os.path.join(results_dir, filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

def plot_predictions(dates, y_true, y_pred, model_name, filepath):
    """
    Plots and saves comparison between actual and predicted demand values.

    Args:
        dates (array-like): Dates corresponding to the data.
        y_true (array-like): Actual demand values.
        y_pred (array-like): Model predicted values.
        model_name (str): Model name for title and legend.
        filepath (str): Path where to save the generated figure.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(dates, y_true, label='Actual')
    plt.plot(dates, y_pred, label=f'Prediction {model_name}')
    plt.title(f'Actual vs. Predicted Demand ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Demand (MWh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()
