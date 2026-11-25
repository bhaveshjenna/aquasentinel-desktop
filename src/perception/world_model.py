"""``ActionWorldModel`` - an action-conditioned LSTM that answers "what happens if I do X?".

This is the learned replacement for the hand-written pond model inside the MPC rollout. The
forecaster in ``forecaster.py`` predicts what the water will do *on its own*; it has never seen
an actuator, so it cannot be asked about one. This network can: it takes the observed window
**and a candidate action**, and predicts the trajectory that action produces. The planner then
scores those learned trajectories and takes the argmin, so the action the agent applies is
chosen from predictions made by a neural network rather than by a differential equation someone
wrote by hand.

**Why it is trained on the twin, and why that is stated rather than hidden.** The 42 real
monitoring sites are rivers. They contain no aerator, no feeding schedule and no liming, so
they contain no ``(action -> response)`` pairs at all. No amount of federation over real data
can teach a model what an aerator does. The action-conditioned part is therefore learned from
twin rollouts under randomised action sequences. What *is* carried over from real data is the
**encoder**: its LSTM weights are warm-started from ``forecaster_federated.pt``, the model
FedAvg produced across the 42 real sites, so the representation the decision is made on is
still the federated one.

**Federation here is across simulated ponds**, not across the real sites: each client is a pond
with its own perturbed physical parameters (volume, depth, aeration efficiency, algal
productivity, temperature regime), which is the non-IID structure a real multi-farm deployment
would have. The same ``fedavg`` in ``src/federated/simulate.py`` aggregates them, and a client
never shares a row - only weights.

Architecture::

    window (W x 2C: delta-encoded channels + availability mask)
        -> LSTM encoder            <- warm-started from the federated forecaster
        -> h (hidden)
        concat action features (8)
        -> MLP head
        -> (H x C) deltas from the last observed value
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..common.runtime import device, load_config, set_seed
from ..common.schema import Action

# The channels the planner reasons over. Deliberately identical to `PLAN_CHANNELS` in
# policy.py and to the forecaster's channel list - that identity is what lets the encoder be
# warm-started from the federated forecaster without any surgery.
CHANNELS: List[str] = ["do", "temp_c", "ph", "tan", "no2", "turbidity"]

# aerator, exchange_small, exchange_large, feed_reduced, feed_skip, lime, salt, hold_fraction
N_ACTION_FEATURES = 8


def action_features(action: Action, hold_fraction: float = 1.0) -> np.ndarray:
    """Encode an action for the network.

    ``hold_fraction`` is how much of the horizon the action is actually held for before the
    planner reverts to no-op. The planner holds a candidate for ``hold_action_steps`` and then
    stops, so a model told only "aerator=1.0" would systematically over-predict the effect.
    """
    return np.array(
        [
            float(action.aerator),
            1.0 if action.exchange == "small" else 0.0,
            1.0 if action.exchange == "large" else 0.0,
            1.0 if action.feed == "reduced" else 0.0,
            1.0 if action.feed == "skip" else 0.0,
            1.0 if action.lime == "dose" else 0.0,
            1.0 if action.salt == "dose" else 0.0,
            float(np.clip(hold_fraction, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


class _WorldNet(nn.Module):
    def __init__(self, n_channels: int, hidden: int, layers: int, horizon: int, dropout: float):
        super().__init__()
        self.n_channels = n_channels
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=2 * n_channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        # The action is fused AFTER the encoder rather than appended to every input step: the
        # window is the past, where the action has not happened yet. Injecting it into the past
        # would teach the model that the action was already in effect during the observations.
        self.head = nn.Sequential(
            nn.Linear(hidden + N_ACTION_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * n_channels),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = torch.cat([out[:, -1, :], a], dim=1)
        return self.head(h).view(-1, self.horizon, self.n_channels)


class ActionWorldModel:
    """Fit / predict / save / load / get_weights / set_weights. FedAvg averages the weights."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, seed: int = 20240101):
        cfg = cfg or load_config("models")
        self.window = int(cfg["window"])
        self.horizon = int(cfg["horizon"])
        self.wm_cfg = dict(cfg.get("world_model", {}))
        self.channels = list(CHANNELS)
        base = cfg["forecaster"]["delta_scale"]
        self.scales = np.array([float(base[c]) for c in self.channels], dtype=np.float32)
        self.seed = int(seed)
        set_seed(seed)
        self.net = _WorldNet(
            len(self.channels),
            int(self.wm_cfg.get("hidden_size", 64)),
            int(self.wm_cfg.get("num_layers", 1)),
            self.horizon,
            float(self.wm_cfg.get("dropout", 0.0)),
        ).to(device())
        self.step_minutes = 5.0
        self.trained_steps = 0

    # ------------------------------------------------------------------ encoding
    def encode_window(self, window: np.ndarray) -> np.ndarray:
        """(W, C) absolute values -> (W, 2C) delta-encoded input with an all-present mask."""
        w = np.asarray(window, dtype=np.float32)
        mask = np.isfinite(w)
        filled = np.nan_to_num(w, nan=0.0)
        last = filled[-1]
        return np.concatenate(
            [(filled - last) / self.scales, mask.astype(np.float32)], axis=1
        ).astype(np.float32)

    # ------------------------------------------------------------------ training
    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """X (N, W, 2C) encoded windows, A (N, 8) actions, Y (N, H, C) scaled deltas."""
        epochs = int(epochs if epochs is not None else self.wm_cfg.get("epochs", 12))
        bs = int(self.wm_cfg.get("batch_size", 256))
        lr = float(self.wm_cfg.get("lr", 0.003))
        clip = float(self.wm_cfg.get("grad_clip", 1.0))
        dev = device()

        xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(dev)
        at = torch.from_numpy(np.asarray(A, dtype=np.float32)).to(dev)
        yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(dev)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossf = nn.MSELoss()
        n = len(xt)
        last = float("nan")
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            total, seen = 0.0, 0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                pred = self.net(xt[idx], at[idx])
                loss = lossf(pred, yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                opt.step()
                total += float(loss.item()) * len(idx)
                seen += len(idx)
            last = total / max(seen, 1)
            if verbose:
                print(f"      loss {last:.6f}")
        self.net.eval()
        self.trained_steps += epochs * max(n // bs, 1)
        return {"loss": last, "n": int(n), "epochs": epochs}

    # ------------------------------------------------------------------ inference
    def predict(self, window: np.ndarray, action: Action, hold_fraction: float = 1.0) -> Dict[str, np.ndarray]:
        """(W, C) window + action -> {channel: (H,) absolute trajectory}."""
        x = self.encode_window(window)[None, ...]
        a = action_features(action, hold_fraction)[None, ...]
        with torch.no_grad():
            out = self.net(
                torch.from_numpy(x).to(device()), torch.from_numpy(a).to(device())
            ).cpu().numpy()[0]
        last = np.nan_to_num(np.asarray(window, dtype=np.float32)[-1], nan=0.0)
        abs_traj = last[None, :] + out * self.scales[None, :]
        return {c: abs_traj[:, i].astype(float) for i, c in enumerate(self.channels)}

    def predict_batch(self, window: np.ndarray, actions: List[Action], hold_fraction: float = 1.0) -> List[Dict[str, np.ndarray]]:
        """One forward pass for every candidate - this is the whole planner tick."""
        x = np.repeat(self.encode_window(window)[None, ...], len(actions), axis=0)
        a = np.stack([action_features(act, hold_fraction) for act in actions])
        with torch.no_grad():
            out = self.net(
                torch.from_numpy(x).to(device()), torch.from_numpy(a).to(device())
            ).cpu().numpy()
        last = np.nan_to_num(np.asarray(window, dtype=np.float32)[-1], nan=0.0)
        res = []
        for k in range(len(actions)):
            abs_traj = last[None, :] + out[k] * self.scales[None, :]
            res.append({c: abs_traj[:, i].astype(float) for i, c in enumerate(self.channels)})
        return res

    # ------------------------------------------------------------------ weights
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().copy() for k, v in self.net.state_dict().items()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.net.load_state_dict(
            {k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()}, strict=True
        )

    def warm_start_encoder(self, forecaster_path: Path) -> bool:
        """Copy the federated forecaster's LSTM weights into this encoder.

        This is the link back to the real data: the representation the planner's decision is
        computed from is the one FedAvg produced across 42 real monitoring sites. Only the
        encoder transfers - the action head has no counterpart there. Returns False rather than
        raising if the shapes do not line up, so a config change degrades to random init with a
        warning instead of a crash.
        """
        path = Path(forecaster_path)
        if not path.exists():
            return False
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            sd = blob["state_dict"]
            enc = {k[len("lstm.") :]: v for k, v in sd.items() if k.startswith("lstm.")}
            if not enc:
                return False
            self.net.lstm.load_state_dict(enc, strict=True)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ io
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
    def load(cls, path: Path, cfg=None) -> "ActionWorldModel":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(cfg)
        m.net.load_state_dict(blob["state_dict"])
        m.channels = list(blob["channels"])
        m.window = int(blob["window"])
        m.horizon = int(blob["horizon"])
        m.scales = np.asarray(blob["scales"], dtype=np.float32)
        m.step_minutes = float(blob.get("step_minutes", 5.0))
        m.net.eval()
        return m
