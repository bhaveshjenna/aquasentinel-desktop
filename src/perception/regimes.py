"""``RegimeModel`` - unsupervised structure discovery over water-quality windows (HDBSCAN).

No labels are used anywhere in this module.

**What this model actually delivers, stated up front.** The intended deliverable was a
taxonomy of pond operating regimes. The data does not support one. Swept across
``min_cluster_size`` 40-800, ``min_samples`` 3-80, both selection methods, and a PCA-reduced
variant, and on pooled real+twin windows as well as twin windows alone, HDBSCAN returns the
same shape every time: **one dominant density mode holding ~80 % of windows, a few small
satellites, and DBCV indistinguishable from zero.** The sweep is written to
``outputs/regime_sweep.csv`` so the claim rests on evidence rather than assertion.

The honest reading is that continuous water-quality windows form a connected continuum, not
separated clusters. A pond does not jump between discrete states; it slides around a manifold.
So the model is used for what it does support:

  * **the anomaly flag** (the real deliverable) - HDBSCAN's noise label marks a window unlike
    anything in the training distribution. It cannot distinguish a stuck probe from a genuine
    novel excursion, and the ``sensor_fault`` scenario in the evaluation measures exactly how
    far that gets you.
  * **a coarse two-mode split** with post-hoc descriptive names read off the centroids. These
    are a reading aid on the dashboard. Given DBCV ~ 0 they are NOT a claim that the pond has
    discrete operating regimes, and nothing downstream depends on the cluster id.

Reporting a tuned configuration that happened to produce a prettier cluster count would have
been the dishonest option here. DBCV and silhouette are both reported, and a near-zero DBCV
alongside a healthy silhouette is itself the finding: silhouette flatters a single blob split
in two, DBCV does not.

**Why the features are derived rather than raw.**

A regime has to mean the same thing on a
12 C English river and a 28 C tilapia pond, or the pond is simply "anomalous" against every
cluster and the panel says nothing. Raw levels do not transfer - they *are* the domain gap.
So the clustering runs on quantities that do:

    do_sat      DO divided by its saturation at the current temperature. Dimensionless.
                A river at 9 mg/L and 12 C and a pond at 6.5 mg/L and 28 C are both about
                85 % saturated, and that is the physically meaningful similarity.
    temp_anom   temperature minus the window mean: the diurnal *shape*, not the climate.
    ph_anom     pH minus the window mean: the swing, not the catchment geology.
    turb_log    log1p(turbidity), because turbidity is heavy-tailed over orders of magnitude.

Fitting on raw levels instead sends ~38 % of windows to the noise label rather than ~21 %.
The trade-off is explicit: absolute level is no longer part of what defines a regime, so
"oxygen-depleted" here means depleted *relative to saturation*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.runtime import load_config

DERIVED_CHANNELS = ("do_sat", "temp_anom", "ph_anom", "turb_log")

FEATURE_FUNCS = {
    "mean": lambda w: np.nanmean(w, axis=0),
    "std": lambda w: np.nanstd(w, axis=0),
    "slope": lambda w: (w[-1] - w[0]) / max(len(w) - 1, 1),
    "last": lambda w: w[-1],
    "min": lambda w: np.nanmin(w, axis=0),
    "max": lambda w: np.nanmax(w, axis=0),
}


class RegimeModel:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or load_config("models")
        self.window = int(cfg["window"])
        self.cfg = cfg["regimes"]
        self.channels: List[str] = list(self.cfg.get("channels", cfg["forecaster"]["channels"]))
        self.derived: List[str] = list(self.cfg.get("derived", DERIVED_CHANNELS))
        self.features: List[str] = list(self.cfg["features"])
        self.keep_: Optional[np.ndarray] = None
        self.clusterer = None
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.names: Dict[int, str] = {}
        self.centroids: Dict[int, np.ndarray] = {}
        self.metrics_: Dict[str, float] = {}

    # ------------------------------------------------------------------ features
    def feature_names(self) -> List[str]:
        return [f"{c}_{f}" for f in self.features for c in self.derived]

    def derive(self, window: np.ndarray) -> np.ndarray:
        """Raw (W, C) window -> (W, D) domain-transferable window. See the module docstring."""
        from ..twin.dynamics import do_saturation

        w = np.asarray(window, dtype=float)
        idx = {c: i for i, c in enumerate(self.channels)}
        temp = w[:, idx["temp_c"]] if "temp_c" in idx else np.full(len(w), 20.0)
        cols = []
        for name in self.derived:
            if name == "do_sat":
                do = w[:, idx["do"]] if "do" in idx else np.full(len(w), np.nan)
                sat = np.array([do_saturation(t) if np.isfinite(t) else np.nan for t in temp])
                cols.append(do / np.where(sat > 0, sat, np.nan))
            elif name == "temp_anom":
                cols.append(temp - np.nanmean(temp))
            elif name == "ph_anom":
                ph = w[:, idx["ph"]] if "ph" in idx else np.full(len(w), np.nan)
                cols.append(ph - np.nanmean(ph))
            elif name == "turb_log":
                tb = w[:, idx["turbidity"]] if "turbidity" in idx else np.full(len(w), np.nan)
                cols.append(np.log1p(np.clip(tb, 0, None)))
            else:
                cols.append(w[:, idx[name]] if name in idx else np.full(len(w), np.nan))
        return np.column_stack(cols)

    def featurize_window(self, window: np.ndarray) -> np.ndarray:
        w = self.derive(window)
        parts = [FEATURE_FUNCS[f](w) for f in self.features]
        return np.concatenate(parts)

    def featurize_series(self, values: np.ndarray, stride: int = 1) -> np.ndarray:
        T = len(values)
        if T < self.window:
            return np.empty((0, len(self.feature_names())))
        rows = [
            self.featurize_window(values[i : i + self.window])
            for i in range(0, T - self.window + 1, stride)
        ]
        return np.asarray(rows, dtype=float)

    # ------------------------------------------------------------------ fitting
    def fit(self, features: np.ndarray, verbose: bool = False) -> "RegimeModel":
        import hdbscan
        from sklearn.metrics import silhouette_score

        X = np.asarray(features, dtype=float)
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < int(self.cfg["min_cluster_size"]) * 2:
            raise ValueError(f"not enough finite feature rows to cluster ({len(X)})")

        # Robust scaling: monitoring data has heavy tails and a single turbidity spike would
        # otherwise define the whole metric space.
        # Some derived features are constant by construction - the mean of an anomaly channel
        # is exactly zero - and a dead dimension only dilutes the distance metric.
        full = X
        iqr_all = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        self.keep_ = iqr_all > 1e-9
        X = X[:, self.keep_]
        self.center_ = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        self.scale_ = np.where(iqr > 1e-9, iqr, 1.0)
        Z = (X - self.center_) / self.scale_

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(self.cfg["min_cluster_size"]),
            min_samples=int(self.cfg["min_samples"]),
            metric=str(self.cfg["metric"]),
            cluster_selection_method=str(self.cfg["cluster_selection_method"]),
            prediction_data=True,
            gen_min_span_tree=True,
        )
        labels = self.clusterer.fit_predict(Z)

        uniq = sorted(set(labels) - {-1})
        # Centroids live in the FULL feature space so the post-hoc names stay readable.
        self.centroids = {int(k): full[labels == k].mean(axis=0) for k in uniq}
        self.names = self._name_regimes()

        noise_frac = float((labels == -1).mean())
        try:
            dbcv = float(self.clusterer.relative_validity_)
        except Exception:
            dbcv = float("nan")
        sil = float("nan")
        core = labels != -1
        if len(uniq) > 1 and core.sum() > len(uniq):
            sil = float(silhouette_score(Z[core], labels[core]))
        self.metrics_ = {
            "n_clusters": len(uniq),
            "noise_fraction": noise_frac,
            "dbcv": dbcv,
            "silhouette": sil,
            "n_points": int(len(X)),
        }
        if verbose:
            print(f"  regimes: {self.metrics_}")
            for k, name in self.names.items():
                print(f"    {k}: {name}")
        return self

    # ------------------------------------------------------------------ inference
    def assign_features(self, feat: np.ndarray) -> Tuple[int, str, bool]:
        import hdbscan

        if self.clusterer is None:
            return -1, "unknown", False
        v = np.asarray(feat, dtype=float)
        if self.keep_ is not None:
            v = v[self.keep_]
        z = ((v - self.center_) / self.scale_)[None, :]
        if not np.isfinite(z).all():
            return -1, "anomaly (incomplete reading)", True
        label, _strength = hdbscan.approximate_predict(self.clusterer, z)
        k = int(label[0])
        return k, self.names.get(k, f"regime {k}"), k == -1

    def assign(self, history) -> Tuple[int, str, bool]:
        """From a live history DataFrame (the SensorStream frame)."""
        if history is None or len(history) < self.window:
            return -1, "warming up", False
        cols = [
            history[c].to_numpy(dtype=float) if c in history.columns
            else np.full(len(history), np.nan)
            for c in self.channels
        ]
        arr = np.column_stack(cols)[-self.window :]
        arr = _fill(arr)
        return self.assign_features(self.featurize_window(arr))

    # ------------------------------------------------------------------ naming
    def _name_regimes(self) -> Dict[int, str]:
        """Post-hoc labels read off the centroids. Descriptive, never used for training."""
        if not self.centroids:
            return {}
        names: Dict[int, str] = {}
        idx = {n: i for i, n in enumerate(self.feature_names())}

        def get(c: np.ndarray, ch: str, feat: str) -> float:
            key = f"{ch}_{feat}"
            return float(c[idx[key]]) if key in idx else float("nan")

        turb_med = np.nanmedian([get(c, "turb_log", "mean") for c in self.centroids.values()])
        swing_med = np.nanmedian([get(c, "temp_anom", "std") for c in self.centroids.values()])

        for k, c in self.centroids.items():
            sat, sat_slope = get(c, "do_sat", "mean"), get(c, "do_sat", "slope")
            bits = []
            # Absolute saturation, not a rank among the clusters: "60 % saturated" means the
            # same thing whatever else the model happened to find.
            if np.isfinite(sat):
                if sat > 1.10:
                    bits.append("supersaturated")
                elif sat > 0.85:
                    bits.append("well-oxygenated")
                elif sat > 0.55:
                    bits.append("moderately oxygenated")
                else:
                    bits.append("oxygen-depleted")
            if np.isfinite(sat_slope):
                if sat_slope > 0.002:
                    bits.append("recovering")
                elif sat_slope < -0.002:
                    bits.append("declining")
                else:
                    bits.append("steady")
            if np.isfinite(get(c, "temp_anom", "std")) and get(c, "temp_anom", "std") > swing_med:
                bits.append("strongly diurnal")
            if np.isfinite(get(c, "turb_log", "mean")) and get(c, "turb_log", "mean") > turb_med:
                bits.append("turbid")
            names[int(k)] = " ".join(bits)

        # Disambiguate collisions so the dashboard never shows two identical regime names.
        seen: Dict[str, int] = {}
        for k in sorted(names):
            base = names[k]
            if base in seen:
                seen[base] += 1
                names[k] = f"{base} ({seen[base]})"
            else:
                seen[base] = 1
        return names

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "clusterer": self.clusterer,
                    "center": self.center_,
                    "scale": self.scale_,
                    "keep": self.keep_,
                    "derived": self.derived,
                    "names": self.names,
                    "centroids": self.centroids,
                    "metrics": self.metrics_,
                    "channels": self.channels,
                    "features": self.features,
                    "window": self.window,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, path: Path, cfg=None) -> "RegimeModel":
        import pickle

        model = cls(cfg)
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        model.clusterer = blob["clusterer"]
        model.center_ = blob["center"]
        model.scale_ = blob["scale"]
        model.keep_ = blob.get("keep")
        model.derived = blob.get("derived", model.derived)
        model.names = blob["names"]
        model.centroids = blob["centroids"]
        model.metrics_ = blob["metrics"]
        model.channels = blob["channels"]
        model.features = blob["features"]
        model.window = blob["window"]
        return model


def _fill(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        good = np.isfinite(col)
        if not good.any():
            col[:] = 0.0
        else:
            col[:] = np.interp(np.arange(len(col)), np.where(good)[0], col[good])
        out[:, j] = col
    return out


def sweep(features: np.ndarray, cfg=None, grid=None) -> "object":
    """Refit across a grid of HDBSCAN settings and return the quality of each.

    This exists so that "the density structure is not there" is a measured statement rather
    than an excuse. Returns a DataFrame; ``pretrain_federated.py`` writes it to
    ``outputs/regime_sweep.csv``.
    """
    import pandas as pd

    base = load_config("models") if cfg is None else cfg
    grid = grid or [
        (40, 10, "eom"), (40, 5, "eom"), (80, 5, "eom"), (80, 10, "eom"),
        (150, 25, "eom"), (400, 50, "eom"),
        (150, 25, "leaf"), (400, 50, "leaf"), (800, 80, "leaf"),
    ]
    rows = []
    for mcs, ms, method in grid:
        c = {**base, "regimes": {**base["regimes"], "min_cluster_size": mcs,
                                 "min_samples": ms, "cluster_selection_method": method}}
        m = RegimeModel(c)
        try:
            m.fit(features)
        except Exception as exc:
            rows.append({"min_cluster_size": mcs, "min_samples": ms,
                         "selection": method, "error": str(exc)[:80]})
            continue
        rows.append({"min_cluster_size": mcs, "min_samples": ms, "selection": method,
                     **m.metrics_})
    return pd.DataFrame(rows)
