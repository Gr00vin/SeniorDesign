"""
config.py — Central configuration for the Heston/Bates PINN calibration pipeline.
All hyperparameters, paths, and device settings live here.
"""

import torch

# =============================================================================
# DEVICE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # double precision for financial numerics

# =============================================================================
# DATA
# =============================================================================
FILEPATH = "spxData_fixed.csv"
TARGET_COLUMN = 17
COLS_TO_IGNORE = [0, 1, 3, 4, 5, 6, 7, 9, 11, 13, 16, 18]
VALIDATION_SPLIT = 0.2

# =============================================================================
# MODEL
# =============================================================================
USE_JUMPS = True                    # False = Heston (5 params), True = Bates (8 params)
NUM_PARAMS = 8 if USE_JUMPS else 5

# =============================================================================
# TRAINING
# =============================================================================
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# =============================================================================
# QUADRATURE (Gauss-Laguerre for characteristic-function pricing)
# =============================================================================
GL_NUM_POINTS = 32                  # number of Gauss-Laguerre nodes

# =============================================================================
# PARAMETER BOUNDS  (sigmoid -> physics range)
#   The neural network outputs sigmoid ∈ [0,1]; the scaler layer maps to:
# =============================================================================
PARAM_BOUNDS = {
    "theta": (0.01, 0.50),          # long-run variance
    "rho":   (-0.99, -0.01),        # correlation (forced negative)
    "v0":    (0.01, 0.50),          # initial variance
    "xi":    (0.01, 1.00),          # vol-of-vol
    "kappa": (0.10, 20.0),          # mean-reversion speed
    # jump parameters (Bates only)
    "muJ":    (-0.20, 0.20),        # jump mean
    "sigmaJ": (0.01, 0.20),        # jump vol
    "lam":    (0.00, 5.00),         # jump intensity
}
