from __future__ import annotations

import torch
from torch import nn

from macro_bilstm.models._init import init_xavier_uniform_


class CNNBiLSTM(nn.Module):
    """
    Paper-matching CNN–BiLSTM (PyTorch):
      - 2x Conv1d(64 filters, kernel=3) + MaxPool1d(pool=2, valid)
      - BiLSTM stack units (200, 100, 50)
      - Dense(25) + output Dense(horizon)
      - Dropout: 0.1 after CNN and FC; 0.5 after each BiLSTM
      - ReLU applied to BiLSTM outputs (as described in the paper)
    """

    def __init__(
        self,
        *,
        lookback: int,
        n_features: int,
        horizon: int,
        conv_filters: int = 64,
        conv_kernel: int = 3,
        pool_size: int = 2,
        lstm_units: tuple[int, int, int] = (200, 100, 50),
        fc_units: int = 25,
        dropout_cnn: float = 0.1,
        dropout_lstm: float = 0.5,
        dropout_fc: float = 0.1,
    ) -> None:
        super().__init__()
        if lookback <= 1:
            raise ValueError("lookback must be > 1")
        if n_features <= 0 or horizon <= 0:
            raise ValueError("n_features and horizon must be positive")

        self.lookback = lookback
        self.n_features = n_features
        self.horizon = horizon

        padding = conv_kernel // 2  # 'same' for odd kernels
        self.conv1 = nn.Conv1d(n_features, conv_filters, kernel_size=conv_kernel, padding=padding)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=conv_kernel, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        self.drop_cnn = nn.Dropout(p=dropout_cnn)

        u1, u2, u3 = lstm_units
        self.bilstm1 = nn.LSTM(input_size=conv_filters, hidden_size=u1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(p=dropout_lstm)
        self.bilstm2 = nn.LSTM(input_size=2 * u1, hidden_size=u2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(p=dropout_lstm)
        self.bilstm3 = nn.LSTM(input_size=2 * u2, hidden_size=u3, batch_first=True, bidirectional=True)
        self.drop3 = nn.Dropout(p=dropout_lstm)

        self.fc = nn.Linear(2 * u3, fc_units)
        self.drop_fc = nn.Dropout(p=dropout_fc)
        self.out = nn.Linear(fc_units, horizon)

        self.relu = nn.ReLU()

        init_xavier_uniform_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        if x.ndim != 3:
            raise ValueError("Expected x with shape (batch, lookback, n_features)")

        x = x.transpose(1, 2)  # (batch, n_features, lookback)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # (batch, conv_filters, lookback')
        x = self.drop_cnn(x)
        x = x.transpose(1, 2)  # (batch, lookback', conv_filters)

        x, _ = self.bilstm1(x)  # (batch, seq, 2*u1)
        x = self.relu(x)
        x = self.drop1(x)

        x, _ = self.bilstm2(x)  # (batch, seq, 2*u2)
        x = self.relu(x)
        x = self.drop2(x)

        _, (h_n, _) = self.bilstm3(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 2*u3)
        h = self.relu(h)
        h = self.drop3(h)

        h = self.relu(self.fc(h))
        h = self.drop_fc(h)
        return self.out(h)

