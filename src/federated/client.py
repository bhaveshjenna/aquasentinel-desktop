"""``PondClient`` - one real monitoring site, one independent model, no shared raw data.

The isolation is structural, not a promise: a client holds its own ``SiteDataset``, builds its
own windows, owns its own ``LSTMForecaster`` instance, and exposes exactly four operations to
the outside world - ``local_train``, ``get_weights``, ``set_weights``, ``evaluate``. Nothing
in this class can reach another client's arrays, and the server in ``simulate.py`` only ever
receives weight tensors and a sample count.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..common.runtime import load_config
from ..data.registry import SiteDataset
from ..perception.forecaster import LSTMForecaster


class PondClient:
    def __init__(
        self,
        site: SiteDataset,
        model_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 7,
        stride: int = 4,
        val_fraction: float = 0.15,
        data_fraction: float = 1.0,
    ) -> None:
        self.site = site
        self.client_id = site.client_id
        self.source = site.source
        self.region = site.region
        self.seed = int(seed)
        self.data_fraction = float(data_fraction)
        self.model_cfg = model_cfg or load_config("models")
        self.model = LSTMForecaster(self.model_cfg, seed=seed)

        values = site.matrix(self.model.channels)
        mask = site.mask_matrix(self.model.channels)
        X, Y, M = self.model.make_windows(values, mask, stride=stride)

        # The validation tail is the END of the series, never a random split: a random split
        # over overlapping windows leaks the future into training and flatters every regime.
        n = len(X)
        n_val = int(n * float(val_fraction))
        n_train = n - n_val
        if self.data_fraction < 1.0:
            keep = max(int(n_train * self.data_fraction), 1)
            # Data-poor clients keep the OLDEST slice, so their validation tail is unaffected.
            self.X, self.Y, self.M = X[:keep], Y[:keep], M[:keep]
        else:
            self.X, self.Y, self.M = X[:n_train], Y[:n_train], M[:n_train]
        self.Xv, self.Yv, self.Mv = X[n_train:], Y[n_train:], M[n_train:]

        self.channels_present = list(site.channels)
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ interface
    def __len__(self) -> int:
        return int(len(self.X))

    @property
    def n_samples(self) -> int:
        """Sample count for FedAvg weighting. This, and weights, is all the server learns."""
        return int(len(self.X))

    def local_train(self, epochs: Optional[int] = None, verbose: bool = False) -> Dict[str, float]:
        info = self.model.fit(self.X, self.Y, self.M, epochs=epochs, verbose=verbose)
        info["client"] = self.client_id
        self.history.append(info)
        return info

    def get_weights(self) -> Dict[str, np.ndarray]:
        return self.model.get_weights()

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.model.set_weights(weights)

    def evaluate(self, on: str = "val") -> Dict[str, float]:
        X, Y, M = (self.Xv, self.Yv, self.Mv) if on == "val" else (self.X, self.Y, self.M)
        out = self.model.evaluate(X, Y, M)
        out["client"] = self.client_id
        out["source"] = self.source
        out["region"] = self.region
        out["n_train"] = self.n_samples
        return out

    def evaluate_weights(self, weights: Dict[str, np.ndarray], on: str = "val") -> Dict[str, float]:
        """Score foreign weights on this client's held-out tail without keeping them."""
        saved = self.get_weights()
        try:
            self.set_weights(weights)
            return self.evaluate(on)
        finally:
            self.set_weights(saved)

    def reset_model(self, seed: Optional[int] = None) -> None:
        """Fresh initialisation - used to start each regime from identical conditions."""
        self.model = LSTMForecaster(self.model_cfg, seed=int(seed if seed is not None else self.seed))

    def summary(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "source": self.source,
            "region": self.region,
            "channels": "|".join(self.channels_present),
            "n_train_windows": self.n_samples,
            "n_val_windows": int(len(self.Xv)),
            "data_fraction": self.data_fraction,
            "rows": len(self.site),
        }
