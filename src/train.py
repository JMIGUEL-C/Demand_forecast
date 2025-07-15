"""
Training module for PyTorch models in energy demand forecasting.
Includes function for training with early stopping and validation loss tracking.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
    """
    Trains a PyTorch model using MSELoss and Adam, with early stopping based on validation loss.

    Args:
        model (torch.nn.Module): Model to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for Adam optimizer.
        patience (int): Number of epochs without improvement to trigger early stopping.
    Returns:
        tuple: (trained model, list of validation losses per epoch)
    """
    X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    val_losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        # Validation set evaluation
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
            y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
        model.train()
        # Early stopping: stops training if no improvement in validation loss
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    return model, val_losses
