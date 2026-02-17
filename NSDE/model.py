"""
model.py — Physics-informed calibration neural network (PyTorch).

The network outputs sigmoid activations which are mapped through a
differentiable scaler layer to valid Heston / Bates parameter ranges.
"""

import torch
import torch.nn as nn
import numpy as np
from config import DEVICE, DTYPE, PARAM_BOUNDS


class InputNormalization(nn.Module):
    """
    Learnable z-score normalisation that adapts to training data,
    equivalent to keras.layers.Normalization.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features, dtype=DTYPE))
        self.register_buffer("std",  torch.ones(num_features, dtype=DTYPE))

    def adapt(self, X: np.ndarray):
        """Compute and store mean / std from a numpy array."""
        self.mean = torch.tensor(np.nanmean(X, axis=0), dtype=DTYPE, device=DEVICE)
        std = torch.tensor(np.nanstd(X, axis=0), dtype=DTYPE, device=DEVICE)
        std[std == 0] = 1.0          # avoid division by zero for constant cols
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class PhysicsScaler(nn.Module):
    """
    Maps sigmoid outputs ∈ [0, 1] to physics-valid Heston / Bates ranges.
    Fully differentiable — gradients flow straight through.
    """

    def __init__(self, num_params: int = 5):
        super().__init__()
        self.num_params = num_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_params) in [0, 1]
        b = PARAM_BOUNDS
        theta = x[:, 0:1] * (b["theta"][1] - b["theta"][0]) + b["theta"][0]
        rho   = x[:, 1:2] * (b["rho"][1]   - b["rho"][0])   + b["rho"][0]
        v0    = x[:, 2:3] * (b["v0"][1]     - b["v0"][0])    + b["v0"][0]
        xi    = x[:, 3:4] * (b["xi"][1]     - b["xi"][0])    + b["xi"][0]
        kappa = x[:, 4:5] * (b["kappa"][1]  - b["kappa"][0]) + b["kappa"][0]

        parts = [theta, rho, v0, xi, kappa]

        if self.num_params == 8:
            muJ    = x[:, 5:6] * (b["muJ"][1]    - b["muJ"][0])    + b["muJ"][0]
            sigmaJ = x[:, 6:7] * (b["sigmaJ"][1] - b["sigmaJ"][0]) + b["sigmaJ"][0]
            lam    = x[:, 7:8] * (b["lam"][1]    - b["lam"][0])    + b["lam"][0]
            parts += [muJ, sigmaJ, lam]

        return torch.cat(parts, dim=1)


class CalibrationNetwork(nn.Module):
    """
    Full pipeline:  raw features → normalise → hidden layers → sigmoid → physics scale
    """

    def __init__(self, num_features: int, num_params: int = 5):
        super().__init__()
        self.normalizer = InputNormalization(num_features)
        self.scaler = PhysicsScaler(num_params)

        self.backbone = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, dtype=DTYPE),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, dtype=DTYPE),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_params),
            nn.Sigmoid(),           # raw ∈ [0, 1]
        )

        # initialise weights with double precision
        self.to(dtype=DTYPE)

    def adapt(self, X_train: np.ndarray):
        """Adapt the normalisation layer to training data."""
        self.normalizer.adapt(X_train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        raw = self.backbone(x)      # (B, num_params) ∈ [0,1]
        return self.scaler(raw)     # (B, num_params) in physics ranges


def build_model(X_train: np.ndarray, num_params: int = 5) -> CalibrationNetwork:
    """
    Factory: build the network, adapt normalisation, move to DEVICE.
    """
    num_features = X_train.shape[1]
    model = CalibrationNetwork(num_features, num_params).to(DEVICE)
    model.adapt(X_train)
    print(f"\nBuilt CalibrationNetwork  "
          f"({num_features} features → {num_params} params)  on {DEVICE}")
    return model
