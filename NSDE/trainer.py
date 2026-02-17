"""
trainer.py — Physics-informed training loop (PyTorch + CUDA).

The loss is computed by:
  1. NN predicts Heston/Bates parameters from market features.
  2. Closed-form characteristic-function pricer computes model option prices.
  3. MSE between model prices and market mid-prices is back-propagated
     end-to-end through both the pricer and the network.
"""

import time
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import DEVICE, DTYPE, USE_JUMPS
from pricing import price_options_from_features


class HestonCalibrationTrainer:
    def __init__(self, model, feature_names: list[str], learning_rate: float = 1e-3):
        self.model = model
        self.feature_names = feature_names
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history: list[float] = []

    # ── single training step ────────────────────────────────────────────────

    def _train_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()

        # forward: NN → params → characteristic-function price
        params = self.model(X_batch)
        pred_prices = price_options_from_features(
            X_batch, params, self.feature_names, use_jumps=USE_JUMPS
        )

        loss = torch.mean((y_batch - pred_prices) ** 2)

        # backward through pricer + NN
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred_prices.detach()

    # ── full training loop ──────────────────────────────────────────────────

    def train(self, X_train, y_train, batch_size: int = 64, epochs: int = 200):
        print(f"\nStarting training  |  {len(X_train)} samples  |  "
              f"batch {batch_size}  |  {epochs} epochs  |  device={DEVICE}")

        X_t = torch.tensor(X_train, dtype=DTYPE, device=DEVICE)
        y_t = torch.tensor(y_train, dtype=DTYPE, device=DEVICE)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            running_loss = 0.0
            n_batches = 0
            last_pred = None

            for xb, yb in loader:
                loss_val, pred = self._train_step(xb, yb)
                running_loss += loss_val
                n_batches += 1
                last_pred = pred

            avg_loss = running_loss / n_batches
            self.loss_history.append(avg_loss)

            # MAE on last batch for a quick readout
            if last_pred is not None:
                mae = torch.mean(torch.abs(yb - last_pred)).item()
            else:
                mae = float("nan")

            elapsed = time.time() - t0
            print(f"Epoch {epoch:>4d}/{epochs}  |  "
                  f"MSE {avg_loss:12.4f}  |  MAE ${mae:8.2f}  |  "
                  f"{elapsed:.1f}s")

        print("\nTraining complete.")
