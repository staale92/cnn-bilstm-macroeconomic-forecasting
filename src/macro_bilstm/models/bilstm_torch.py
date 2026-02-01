from __future__ import annotations

import torch
from torch import nn

from macro_bilstm.models._init import init_xavier_uniform_


class BiLSTMRegressor(nn.Module):
    """Baseline BiLSTM model (paper benchmark deep architecture)."""

    def __init__(
        self,
        *,
        lookback: int,
        n_features: int,
        horizon: int,
        lstm_units: tuple[int, int, int] = (200, 100, 50),
        fc_units: int = 25,
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

        u1, u2, u3 = lstm_units
        self.bilstm1 = nn.LSTM(input_size=n_features, hidden_size=u1, batch_first=True, bidirectional=True)
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
        x, _ = self.bilstm1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x, _ = self.bilstm2(x)
        x = self.relu(x)
        x = self.drop2(x)

        _, (h_n, _) = self.bilstm3(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.relu(h)
        h = self.drop3(h)

        h = self.relu(self.fc(h))
        h = self.drop_fc(h)
        return self.out(h)

