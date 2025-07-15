"""
LSTM model definition for energy demand forecasting.
Includes a nn.Module-based class for use with PyTorch.
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for multivariate time series.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the LSTM.
        num_layers (int): Number of stacked LSTM layers.
        output_dim (int): Output dimension (typically 1).
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward propagation of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence, features).
        Returns:
            torch.Tensor: Model output for the last time step.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
