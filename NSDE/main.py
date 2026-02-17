"""
main.py — Entry point for the Heston / Bates PINN calibration pipeline.

Usage:
    python main.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")               # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import (
    DEVICE, DTYPE, FILEPATH, TARGET_COLUMN, COLS_TO_IGNORE,
    USE_JUMPS, NUM_PARAMS, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
)
from data import load_historical_options_data
from model import build_model
from trainer import HestonCalibrationTrainer
from pricing import price_options_from_features


def main():
    print("=" * 70)
    print("PHYSICS-INFORMED NEURAL NETWORK CALIBRATION  (PyTorch + CUDA)")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── 1. LOAD DATA ────────────────────────────────────────────────────────
    print(f"\nLoading data from {FILEPATH} ...")
    X, y, feature_names = load_historical_options_data(
        FILEPATH,
        target_col_index=TARGET_COLUMN,
        other_cols_to_drop=COLS_TO_IGNORE,
    )
    if X is None:
        print("Critical error: data failed to load.")
        return

    print(f"Loaded {len(X)} valid rows  |  Features: {feature_names}")

    # ── 2. TRAIN / VAL SPLIT ────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train {len(X_train)}  |  Val {len(X_val)}")

    # ── 3. BUILD MODEL ─────────────────────────────────────────────────────
    model = build_model(X_train, num_params=NUM_PARAMS)

    # ── 4. TRAIN ────────────────────────────────────────────────────────────
    trainer = HestonCalibrationTrainer(
        model, feature_names=feature_names, learning_rate=LEARNING_RATE
    )
    trainer.train(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

    # ── 5. LOSS CURVE ───────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.loss_history, linewidth=2, label="Training Loss (MSE)")
    plt.title(f"Calibration Convergence ({NUM_EPOCHS} epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("\nLoss curve saved to loss_curve.png")

    # ── 6. VALIDATION ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION: MODEL vs MARKET")
    print("=" * 70)

    n_samples = min(5, len(X_val))
    idx = np.random.choice(len(X_val), n_samples, replace=False)
    X_sample = torch.tensor(X_val[idx], dtype=DTYPE, device=DEVICE)
    y_sample = y_val[idx]

    model.eval()
    with torch.no_grad():
        params = model(X_sample)
        model_prices = price_options_from_features(
            X_sample, params, feature_names, use_jumps=USE_JUMPS
        ).cpu().numpy()
        params_np = params.cpu().numpy()

    for i in range(n_samples):
        mkt = y_sample[i]
        mdl = model_prices[i]
        diff = mdl - mkt
        pct = (diff / mkt * 100) if mkt != 0 else 0.0
        p = params_np[i]

        print(f"\nOption {i+1}:")
        print(f"  Market : ${mkt:.2f}")
        print(f"  Model  : ${mdl:.2f}")
        print(f"  Error  : ${diff:.2f}  ({pct:.1f}%)")
        print(f"  Params : theta={p[0]:.4f}  rho={p[1]:.4f}  "
              f"v0={p[2]:.4f}  xi={p[3]:.4f}  kappa={p[4]:.2f}", end="")
        if NUM_PARAMS == 8:
            print(f"  muJ={p[5]:.4f}  sigmaJ={p[6]:.4f}  lam={p[7]:.2f}")
        else:
            print()

    # ── 7. SAVE ─────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "heston_calibration_model.pt")
    print(f"\nModel weights saved to heston_calibration_model.pt")

    import pandas as pd
    pd.DataFrame({"epoch": range(1, len(trainer.loss_history) + 1),
                   "mse": trainer.loss_history}).to_csv(
        "training_history.csv", index=False
    )
    print("Training history saved to training_history.csv")
    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
