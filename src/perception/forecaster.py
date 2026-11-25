"""``LSTMForecaster`` - multivariate, multi-step forecasting. PyTorch, CPU, deterministic.

Trained on real monitoring sites (Section 8), run live on the twin stream. Two design choices
carry the whole thing, and both exist because of the stream -> pond domain gap:

**Delta targets.** The network predicts each channel's *change* from the last observed value,
scaled by a fixed per-channel constant from ``models.yaml`` - never the absolute level. A
network trained on absolute levels learns "British river in April, 9 mg/L" and is useless on a
28 C tilapia pond. A network trained on deltas learns the *shape* of a diurnal oxygen cycle,
which does transfer.

**Fixed scales, not fitted ones.** Standardising by statistics fitted on the pooled training
data would be a privacy leak in the federated setting (a client's mean would have to cross the
boundary) and would make the three regimes incomparable. The scales are constants in config,
so ``independent-local``, ``federated`` and ``central`` share an identical preprocessing.

The availability mask is fed in as extra input channels, so the model can tell "the pond is at
zero" from "the probe is offline" - which is exactly the distinction real sites force on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..common.runtime import device, load_config, set_seed
from ..common.schema import Forecast


class _LSTMNet(nn.Module):
    def __init__(self, n_channels: int, hidden: int, layers: int, horizon: int, dropout: float):
        super().__init__()
        self.n_channels = n_channels
        self.horizon = horizon
        # Input is 2C: the delta-encoded window, plus its availability mask.
        self.lstm = nn.LSTM(
            input_size=2 * n_channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :])
        return y.view(-1, self.horizon, self.n_channels)


class LSTMForecaster:
    """Fit / predict / save / load. The weights are what federated learning averages."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, seed: int = 20240101):
        cfg = cfg or load_config("models")
        self.window = int(cfg["window"])
        self.horizon = int(cfg["horizon"])
        self.fc_cfg = cfg["forecaster"]
        self.channels: List[str] = list(self.fc_cfg["channels"])
        self.scales = np.array(
            [float(self.fc_cfg["delta_scale"][c]) for c in self.channels], dtype=np.float32
        )
        anchors = self.fc_cfg.get("anchor_channels", self.channels)
        self.anchor_idx = [self.channels.index(c) for c in anchors if c in self.channels]
        self.seed = int(seed)
        set_seed(seed)
        self.net = _LSTMNet(
            len(self.channels),
            int(self.fc_cfg["hidden_size"]),
            int(self.fc_cfg["num_layers"]),
            self.horizon,
            float(self.fc_cfg["dropout"]),
        ).to(device())
        self.step_minutes = 5.0
        self.trained_steps = 0

    # ------------------------------------------------------------------ encoding
    def make_windows(
        self, values: np.ndarray, mask: Optional[np.ndarray] = None, stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(T, C) series -> (N, W, 2C) inputs, (N, H, C) targets, (N, H, C) target masks.

        A window is kept only if the anchor step (the last input step) has a real reading in
        every channel the model forecasts - otherwise "the change from the last value" has no
        defined origin. Gaps elsewhere are carried through as mask zeros.
        """
        T, C = values.shape
        if mask is None:
            mask = np.isfinite(values)
        W, H = self.window, self.horizon
        if T < W + H:
            return (
                np.empty((0, W, 2 * C), np.float32),
                np.empty((0, H, C), np.float32),
                np.empty((0, H, C), bool),
            )

        filled = _forward_fill(values)
        xs, ys, ms = [], [], []
        for start in range(0, T - W - H + 1, stride):
            win = filled[start : start + W]
            win_mask = mask[start : start + W]
            last = win[-1]
            if not np.isfinite(last[self.anchor_idx]).all():
                continue
            tgt = filled[start + W : start + W + H]
            tgt_mask = mask[start + W : start + W + H]
            if tgt_mask.sum() == 0:
                continue
            # A channel this site does not carry has no origin; zero is a neutral one, and the
            # target mask already excludes it from the loss.
            last = np.nan_to_num(last, nan=0.0)
            xs.append(
                np.concatenate(
                    [
                        np.nan_to_num((win - last) / self.scales, nan=0.0),
                        win_mask.astype(np.float32),
                    ],
                    axis=1,
                )
            )
            ys.append(np.nan_to_num((tgt - last) / self.scales, nan=0.0))
            ms.append(tgt_mask)
        if not xs:
            return (
                np.empty((0, W, 2 * C), np.float32),
                np.empty((0, H, C), np.float32),
                np.empty((0, H, C), bool),
            )
        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(ms, dtype=bool),
        )

    # ------------------------------------------------------------------ training
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        M: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Masked-MSE training. Returns the loss history summary."""
        if len(X) == 0:
            return {"loss": float("nan"), "n": 0, "epochs": 0}
        epochs = int(epochs if epochs is not None else self.fc_cfg["epochs"])
        bs = int(self.fc_cfg["batch_size"])
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.fc_cfg["lr"]))
        clip = float(self.fc_cfg["grad_clip"])

        xt = torch.from_numpy(X)
        yt = torch.from_numpy(Y)
        mt = torch.from_numpy((M if M is not None else np.ones(Y.shape, bool)).astype(np.float32))
        n = len(xt)
        g = torch.Generator().manual_seed(self.seed)

        self.net.train()
        last_loss = float("nan")
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g)
            total, count = 0.0, 0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                pred = self.net(xt[idx])
                w = mt[idx]
                loss = (((pred - yt[idx]) ** 2) * w).sum() / w.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                opt.step()
                total += loss.detach().item() * len(idx)
                count += len(idx)
            last_loss = total / max(count, 1)
            self.trained_steps += 1
            if verbose:
                print(f"    epoch {ep + 1}/{epochs}  masked MSE {last_loss:.5f}")
        return {"loss": last_loss, "n": n, "epochs": epochs}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_array(self, window: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """(W, C) window -> (H, C) absolute-value forecast."""
        if mask is None:
            mask = np.isfinite(window)
        filled = _forward_fill(window)
        last = np.nan_to_num(filled[-1], nan=0.0)
        x = np.concatenate(
            [
                np.nan_to_num((filled - last) / self.scales, nan=0.0),
                mask.astype(np.float32),
            ],
            axis=1,
        )[None, ...].astype(np.float32)
        self.net.eval()
        delta = self.net(torch.from_numpy(x)).numpy()[0]
        return last + delta * self.scales

    def predict(self, history) -> Optional[Forecast]:
        """Forecast from a live history DataFrame (the SensorStream frame)."""
        import pandas as pd

        if history is None or len(history) < self.window:
            return None
        cols = []
        for c in self.channels:
            cols.append(
                history[c].to_numpy(dtype=float)
                if c in history.columns
                else np.full(len(history), np.nan)
            )
        arr = np.column_stack(cols)[-self.window :]
        mean = self.predict_array(arr)
        return Forecast(
            channels=list(self.channels),
            horizon=self.horizon,
            mean=mean,
            step_minutes=self.step_minutes,
            source="lstm",
        )

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, Y: np.ndarray, M: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Per-channel RMSE/MAE in real units, plus per-horizon RMSE for the DO channel."""
        if len(X) == 0:
            return {}
        self.net.eval()
        pred = self.net(torch.from_numpy(X)).numpy()
        mask = M if M is not None else np.ones(Y.shape, bool)
        out: Dict[str, float] = {}
        for i, c in enumerate(self.channels):
            m = mask[:, :, i]
            if m.sum() == 0:
                continue
            err = (pred[:, :, i] - Y[:, :, i]) * self.scales[i]
            out[f"rmse_{c}"] = float(np.sqrt((err[m] ** 2).mean()))
            out[f"mae_{c}"] = float(np.abs(err[m]).mean())
        # Skill against persistence, which in delta space is simply predicting zero.
        for i, c in enumerate(self.channels):
            m = mask[:, :, i]
            if m.sum() == 0:
                continue
            base = (Y[:, :, i] * self.scales[i])[m]
            err = ((pred[:, :, i] - Y[:, :, i]) * self.scales[i])[m]
            denom = float(np.sqrt((base**2).mean()))
            out[f"skill_{c}"] = float(1.0 - np.sqrt((err**2).mean()) / denom) if denom > 0 else np.nan
        out["n_windows"] = int(len(X))
        return out

    # ------------------------------------------------------------------ weights
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().copy() for k, v in self.net.state_dict().items()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.net.load_state_dict(
            {k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()}, strict=True
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "channels": self.channels,
                "window": self.window,
                "horizon": self.horizon,
                "scales": self.scales,
                "step_minutes": self.step_minutes,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path, cfg=None) -> "LSTMForecaster":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(cfg)
        model.net.load_state_dict(blob["state_dict"])
        model.channels = list(blob["channels"])
        model.window = int(blob["window"])
        model.horizon = int(blob["horizon"])
        model.scales = np.asarray(blob["scales"], dtype=np.float32)
        model.step_minutes = float(blob.get("step_minutes", 5.0))
        return model


def _forward_fill(a: np.ndarray) -> np.ndarray:
    """Column-wise forward fill, then back fill, so a window has no interior NaN holes."""
    out = np.array(a, dtype=float, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = np.where(np.isfinite(col))[0]
        if len(idx) == 0:
            continue
        first = idx[0]
        col[:first] = col[first]
        last_good = col[first]
        for i in range(first, len(col)):
            if np.isfinite(col[i]):
                last_good = col[i]
            else:
                col[i] = last_good
        out[:, j] = col
    return out
