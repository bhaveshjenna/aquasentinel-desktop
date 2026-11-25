"""SHAP over the risk model: which inputs drove *this tick's* risk score.

``TreeExplainer`` on the LightGBM ensemble, so attributions are exact rather than sampled and
cheap enough to run on every dashboard tick. The output feeds two places: the risk panel, and
the agent's rationale string - so the operator sees the same explanation the agent reasoned
with, not a separate story told afterwards.

Attributions are for the CRITICAL class specifically. Explaining the argmax class means the
explanation changes meaning as the prediction moves; explaining "what is pushing this toward
critical" is a stable question with a stable answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PRETTY = {
    "now_do": "DO now",
    "now_temp_c": "temperature now",
    "now_ph": "pH now",
    "now_tan": "TAN now",
    "now_nh3": "un-ionised NH3 now",
    "now_no2": "nitrite now",
    "now_turbidity": "turbidity now",
    "fc_do_min": "forecast min DO",
    "fc_do_end": "forecast DO at horizon",
    "fc_do_slope": "forecast DO trend",
    "fc_nh3_max": "forecast max NH3",
    "fc_no2_max": "forecast max nitrite",
    "fc_ph_min": "forecast min pH",
    "fc_ph_max": "forecast max pH",
    "fc_temp_c_max": "forecast max temperature",
}


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("fc_"):
        parts = name.split("_")
        return f"forecast {parts[-1]} {'_'.join(parts[1:-1])}"
    return name.replace("now_", "").replace("_", " ")


class RiskExplainer:
    """Wraps a fitted ``RiskModel``. Degrades to an empty list rather than raising."""

    def __init__(self, risk_model, background: Optional[np.ndarray] = None, top_k: int = 5):
        self.risk = risk_model
        self.top_k = int(top_k)
        self.explainer = None
        self.critical_index = (
            risk_model.classes.index("critical") if "critical" in risk_model.classes else -1
        )
        if risk_model.model is not None:
            try:
                import shap

                self.explainer = shap.TreeExplainer(risk_model.model, data=background)
            except Exception as exc:  # SHAP is optional at runtime, never fatal
                self.error = str(exc)

    def shap_row(self, row: Dict[str, float]) -> Optional[np.ndarray]:
        if self.explainer is None:
            return None
        X = self.risk._matrix([row])
        vals = self.explainer.shap_values(X)
        # LightGBM multiclass gives either a list per class or an (n, features, classes) array.
        if isinstance(vals, list):
            arr = vals[self.critical_index] if self.critical_index >= 0 else vals[-1]
            return np.asarray(arr)[0]
        arr = np.asarray(vals)
        if arr.ndim == 3:
            return arr[0, :, self.critical_index if self.critical_index >= 0 else -1]
        return arr[0]

    def top_drivers(
        self, observed: Dict[str, float], forecast_summary: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """[(pretty feature name, signed SHAP value)], largest magnitude first."""
        row = self.risk.features(observed, forecast_summary)
        try:
            contrib = self.shap_row(row)
        except Exception:
            contrib = None
        if contrib is None:
            return self._fallback(row)
        names = self.risk.feature_names_
        order = np.argsort(-np.abs(contrib))[: self.top_k]
        return [(pretty(names[i]), float(contrib[i])) for i in order]

    def _fallback(self, row: Dict[str, float]) -> List[Tuple[str, float]]:
        """Without SHAP, fall back to LightGBM's global gain importance.

        Clearly weaker - it is global, not per-tick - so the dashboard labels it as such.
        """
        model = self.risk.model
        if model is None or not self.risk.feature_names_:
            return []
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            return []
        order = np.argsort(-imp)[: self.top_k]
        return [(pretty(self.risk.feature_names_[i]) + " (global)", float(imp[i])) for i in order]


def explanation_sentence(drivers: List[Tuple[str, float]]) -> str:
    """One line for the agent's rationale and the intervention log."""
    if not drivers:
        return "no risk attribution available"
    up = [n for n, v in drivers if v > 0][:2]
    down = [n for n, v in drivers if v < 0][:1]
    parts = []
    if up:
        parts.append("raised by " + " and ".join(up))
    if down:
        parts.append("offset by " + down[0])
    return "risk " + "; ".join(parts) if parts else "risk drivers balanced"
