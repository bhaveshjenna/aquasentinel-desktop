"""``RiskModel`` - LightGBM over current channels plus forecast summaries.

Read the honesty note before reading the code. The target is **rule-derived**: it is the risk
class that ``twin/risk_bands.py`` assigns to the *worst point of the forecast horizon*. The
model is therefore not learning water chemistry - it is learning to anticipate a known rule
from an imperfect forecast, quickly and with attributable inputs.

That framing is what makes SHAP meaningful here. "The risk is critical because forecast min DO
is 2.1" is an honest explanation of the model's decision, and it is also, by construction, an
honest explanation of the underlying rule. If the target were a learned notion of "quality",
neither claim would hold.

Why bother modelling a rule at all, rather than just evaluating it? Because the rule needs the
future, the future comes from a forecaster with real error, and a tree ensemble over the whole
forecast summary degrades far more gracefully under that error than a hard threshold on a
single predicted value does. The ablation in the evaluation quantifies exactly that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.runtime import load_config
from ..common.schema import RISK_LEVELS
from ..twin import risk_bands as RB

CURRENT_FEATURES = ["do", "temp_c", "ph", "tan", "nh3", "no2", "turbidity"]


class RiskModel:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, twin_cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or load_config("models")
        self.cfg = cfg["risk"]
        self.twin_cfg = twin_cfg or load_config("twin")
        self.classes: List[str] = list(self.cfg["classes"])
        self.weights: Dict[str, float] = dict(self.cfg["score_weights"])
        self.model = None
        self.feature_names_: List[str] = []
        self.metrics_: Dict[str, float] = {}

    # ------------------------------------------------------------------ features
    def features(self, observed: Dict[str, float], forecast_summary: Dict[str, float]) -> Dict[str, float]:
        row = {f"now_{c}": float(observed.get(c, np.nan)) for c in CURRENT_FEATURES}
        row.update({k: float(v) for k, v in (forecast_summary or {}).items()})
        return row

    def _matrix(self, rows: List[Dict[str, float]]):
        """Feature rows -> a named DataFrame, in a stable column order.

        A DataFrame rather than a bare array on purpose. Fitted on a numpy array, LightGBM's
        sklearn wrapper records placeholder names (``Column_0`` ...) as ``feature_names_in_``,
        and scikit-learn then emits a "does not have valid feature names" warning on *every*
        subsequent predict. Since this model is called once per simulated tick, that came to
        ~92,000 warning writes in a single evaluation grid - pure overhead, and it buried the
        real output. Carrying the real names through fit and predict removes the mismatch and
        makes SHAP's feature labels agree with the model's own.
        """
        import pandas as pd

        if not self.feature_names_:
            keys: List[str] = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            self.feature_names_ = sorted(keys)
        data = [[r.get(k, np.nan) for k in self.feature_names_] for r in rows]
        return pd.DataFrame(data, columns=self.feature_names_, dtype=float)

    # ------------------------------------------------------------------ labelling
    def label_from_forecast(self, forecast_summary: Dict[str, float], observed: Dict[str, float]) -> str:
        """The rule the model learns: worst risk over the horizon, or now if that is worse."""
        now, _ = RB.state_risk(observed, self.twin_cfg)
        ahead = RB.worst_forecast_risk(forecast_summary or {}, self.twin_cfg)
        return max([now, ahead], key=lambda lvl: RISK_LEVELS.index(lvl))

    # ------------------------------------------------------------------ training
    def fit(self, rows: List[Dict[str, float]], labels: List[str], verbose: bool = False) -> "RiskModel":
        import lightgbm as lgb

        X = self._matrix(rows)
        y = np.array([self.classes.index(l) for l in labels], dtype=int)
        present = sorted(set(y.tolist()))
        if len(present) < 2:
            raise ValueError(f"risk training needs >1 class, saw {[self.classes[i] for i in present]}")

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(self.classes),
            num_leaves=int(self.cfg["num_leaves"]),
            max_depth=int(self.cfg["max_depth"]),
            learning_rate=float(self.cfg["learning_rate"]),
            n_estimators=int(self.cfg["n_estimators"]),
            min_child_samples=int(self.cfg["min_child_samples"]),
            class_weight="balanced",       # critical excursions are rare and matter most
            random_state=20240101,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(X, y)
        if verbose:
            counts = {self.classes[i]: int((y == i).sum()) for i in present}
            print(f"  risk model fitted on {len(X)} rows, classes {counts}")
        return self

    def evaluate(self, rows: List[Dict[str, float]], labels: List[str]) -> Dict[str, float]:
        from sklearn.metrics import balanced_accuracy_score, f1_score

        if self.model is None:
            return {}
        X = self._matrix(rows)
        y = np.array([self.classes.index(l) for l in labels], dtype=int)
        pred = self.model.predict(X)
        crit = self.classes.index("critical")
        self.metrics_ = {
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "critical_recall": float(((pred == crit) & (y == crit)).sum() / max((y == crit).sum(), 1)),
            "critical_precision": float(((pred == crit) & (y == crit)).sum() / max((pred == crit).sum(), 1)),
            "n": int(len(y)),
        }
        return self.metrics_

    # ------------------------------------------------------------------ inference
    def predict(self, observed: Dict[str, float], forecast_summary: Dict[str, float]) -> Tuple[str, float]:
        """Return (class, score). Score is the probability-weighted severity in [0, 1]."""
        if self.model is None:
            level = self.label_from_forecast(forecast_summary, observed)
            return level, float(self.weights.get(level, 0.0))
        X = self._matrix([self.features(observed, forecast_summary)])
        proba = self.model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        score = float(sum(p * self.weights.get(c, 0.0) for p, c in zip(proba, self.classes)))
        return self.classes[idx], score

    def predict_proba_row(self, row: Dict[str, float]) -> np.ndarray:
        return self.model.predict_proba(self._matrix([row]))[0]

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names_,
                    "classes": self.classes,
                    "metrics": self.metrics_,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, path: Path, cfg=None, twin_cfg=None) -> "RiskModel":
        import pickle

        m = cls(cfg, twin_cfg)
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        m.model = blob["model"]
        m.feature_names_ = blob["feature_names"]
        m.classes = blob["classes"]
        m.metrics_ = blob.get("metrics", {})
        return m


def build_training_set(
    frames: List[Any],
    forecaster,
    twin_cfg=None,
    stride: int = 3,
    max_rows: int = 40000,
    groups: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, float]], List[str], List[Any]]:
    """Build (feature rows, rule-derived labels, group ids) by sliding over trajectories.

    ``frames`` are DataFrames with the modelled channels - twin runs, real sites, or both.
    Labels come from the *true* future in the frame, which keeps the target honest: the model
    is graded against what actually happened, not against its own forecast.

    The third return value is the group id of each row (whatever was passed in ``groups``,
    else the frame index). It exists so the caller can split train/test along a meaningful
    boundary. Splitting these rows by position is a trap: they are generated frame by frame,
    so the tail of the array is one or two whole scenarios and a "test set" built that way
    measures transfer to an unseen scenario while looking like an ordinary holdout.
    """
    import pandas as pd

    from ..data.harmonize import derive_nh3

    model = RiskModel(twin_cfg=twin_cfg)
    rows: List[Dict[str, float]] = []
    labels: List[str] = []
    row_groups: List[Any] = []
    W, H = forecaster.window, forecaster.horizon

    for frame_i, frame in enumerate(frames):
        group = groups[frame_i] if groups is not None else frame_i
        cols = [
            frame[c].to_numpy(dtype=float) if c in frame.columns
            else np.full(len(frame), np.nan)
            for c in forecaster.channels
        ]
        arr = np.column_stack(cols)
        for i in range(W, len(arr) - H, stride):
            window = arr[i - W : i]
            if not np.isfinite(window[-1][forecaster.anchor_idx]).all():
                continue
            fc = forecaster.predict_array(window)
            summary = _summary(fc, forecaster.channels)
            observed = {c: float(window[-1][j]) for j, c in enumerate(forecaster.channels)}
            observed["nh3"] = float(
                derive_nh3(
                    np.array([observed.get("tan", np.nan)]),
                    np.array([observed.get("ph", np.nan)]),
                    np.array([observed.get("temp_c", np.nan)]),
                )[0]
            )
            # Ground truth: the risk the pond ACTUALLY reached over the next H steps.
            future = pd.DataFrame(arr[i : i + H], columns=forecaster.channels)
            future["nh3"] = derive_nh3(
                future.get("tan", pd.Series(np.nan, index=future.index)).to_numpy(),
                future.get("ph", pd.Series(np.nan, index=future.index)).to_numpy(),
                future.get("temp_c", pd.Series(np.nan, index=future.index)).to_numpy(),
            )
            levels = RB.series_risk(future, twin_cfg)
            worst = max(levels, key=lambda lvl: RISK_LEVELS.index(lvl))
            rows.append(model.features(observed, summary))
            labels.append(worst)
            row_groups.append(group)
            if len(rows) >= max_rows:
                return rows, labels, row_groups
    return rows, labels, row_groups


def _summary(fc: np.ndarray, channels: List[str]) -> Dict[str, float]:
    from ..data.harmonize import derive_nh3

    out: Dict[str, float] = {}
    for j, c in enumerate(channels):
        col = fc[:, j]
        out[f"fc_{c}_min"] = float(np.nanmin(col))
        out[f"fc_{c}_max"] = float(np.nanmax(col))
        out[f"fc_{c}_mean"] = float(np.nanmean(col))
        out[f"fc_{c}_end"] = float(col[-1])
        out[f"fc_{c}_slope"] = float(col[-1] - col[0]) / max(len(col) - 1, 1)
    if {"tan", "ph", "temp_c"}.issubset(set(channels)):
        nh3 = derive_nh3(
            fc[:, channels.index("tan")], fc[:, channels.index("ph")], fc[:, channels.index("temp_c")]
        )
        out["fc_nh3_min"] = float(np.nanmin(nh3))
        out["fc_nh3_max"] = float(np.nanmax(nh3))
        out["fc_nh3_mean"] = float(np.nanmean(nh3))
        out["fc_nh3_end"] = float(nh3[-1])
    return out
