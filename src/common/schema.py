"""Core data contracts shared by the twin, the real-data pipeline, perception, agent and eval.

Everything downstream speaks in these types, so the same forecaster / regime / risk code
runs unchanged whether the input is a real monitoring site or the live twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Channels a probe can report. Real sites carry a per-channel availability mask because
# they differ in which of these actually exist.
SENSOR_CHANNELS: tuple[str, ...] = (
    "do",
    "temp_c",
    "ph",
    "tan",
    "no2",
    "turbidity",
    "conductivity",
)

# Latent twin state - real pond dynamics the operator cannot read off a probe.
LATENT_CHANNELS: tuple[str, ...] = (
    "algae",
    "biomass",
    "alkalinity",
    "bod_pool",
    "feed_pool",
    "chloride",
)

RISK_LEVELS: tuple[str, ...] = ("safe", "warning", "critical")


# --------------------------------------------------------------------------------------
# State / Observation
# --------------------------------------------------------------------------------------
@dataclass
class State:
    """Ground-truth pond state at one instant (twin) or a fully-known real sample."""

    t_hours: float = 0.0
    do: float = float("nan")
    temp_c: float = float("nan")
    ph: float = float("nan")
    tan: float = float("nan")          # total ammonia nitrogen, mg/L as N
    nh3: float = float("nan")          # un-ionised NH3-N, derived from tan + pH + temp
    no2: float = float("nan")
    turbidity: float = float("nan")
    conductivity: float = float("nan")
    # latent
    algae: float = float("nan")
    biomass: float = float("nan")
    alkalinity: float = float("nan")
    bod_pool: float = 0.0
    feed_pool: float = 0.0
    chloride: float = 0.0
    timestamp: Optional[str] = None

    def sensor_vector(self, channels: Sequence[str] = SENSOR_CHANNELS) -> np.ndarray:
        return np.array([getattr(self, c, np.nan) for c in channels], dtype=float)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self, **changes: Any) -> "State":
        return replace(self, **changes)


@dataclass
class Observation:
    """What the perception stack actually sees: sensor- or API-sampled, possibly missing."""

    t_hours: float = 0.0
    values: Dict[str, float] = field(default_factory=dict)
    mask: Dict[str, bool] = field(default_factory=dict)   # True = channel present this sample
    timestamp: Optional[str] = None
    site_id: Optional[str] = None

    @classmethod
    def from_state(cls, s: State, channels: Sequence[str] = SENSOR_CHANNELS) -> "Observation":
        vals = {c: float(getattr(s, c, np.nan)) for c in channels}
        return cls(
            t_hours=s.t_hours,
            values=vals,
            mask={c: bool(np.isfinite(v)) for c, v in vals.items()},
            timestamp=s.timestamp,
        )

    def vector(self, channels: Sequence[str]) -> np.ndarray:
        return np.array([self.values.get(c, np.nan) for c in channels], dtype=float)


@dataclass
class Forecast:
    """Per-channel prediction over the horizon H (rows = steps ahead)."""

    channels: List[str]
    horizon: int
    mean: np.ndarray                                  # (H, C)
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    step_minutes: float = 5.0
    source: str = "lstm"                              # lstm | twin_rollout | persistence

    def channel(self, name: str) -> np.ndarray:
        return self.mean[:, self.channels.index(name)]

    def summary(self) -> Dict[str, float]:
        """Compact features the risk model and the planner both consume."""
        out: Dict[str, float] = {}
        for i, c in enumerate(self.channels):
            col = self.mean[:, i]
            out[f"fc_{c}_min"] = float(np.nanmin(col))
            out[f"fc_{c}_max"] = float(np.nanmax(col))
            out[f"fc_{c}_mean"] = float(np.nanmean(col))
            out[f"fc_{c}_end"] = float(col[-1])
            out[f"fc_{c}_slope"] = float(col[-1] - col[0]) / max(len(col) - 1, 1)
        return out


# --------------------------------------------------------------------------------------
# Action / Decision
# --------------------------------------------------------------------------------------
EXCHANGE_LEVELS = ("none", "small", "large")
FEED_LEVELS = ("normal", "reduced", "skip")
DOSE_LEVELS = ("none", "dose")


@dataclass(frozen=True)
class Action:
    aerator: float = 0.0                  # [0, 1]
    exchange: str = "none"                # none | small | large
    feed: str = "normal"                  # normal | reduced | skip
    lime: str = "none"                    # none | dose
    salt: str = "none"                    # none | dose

    @staticmethod
    def no_op() -> "Action":
        return Action()

    def is_no_op(self) -> bool:
        return self == Action()

    def label(self) -> str:
        bits = []
        if self.aerator > 0:
            bits.append(f"aerator={self.aerator:.2f}")
        if self.exchange != "none":
            bits.append(f"exchange={self.exchange}")
        if self.feed != "normal":
            bits.append(f"feed={self.feed}")
        if self.lime != "none":
            bits.append("lime")
        if self.salt != "none":
            bits.append("salt")
        return ", ".join(bits) if bits else "no_op"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardrailEvent:
    rule: str
    detail: str
    severity: str = "clamp"               # clamp | block | override


@dataclass
class Decision:
    t_hours: float
    proposed: Action
    applied: Action
    risk_score: float = 0.0
    risk_class: str = "safe"
    regime: int = -1
    regime_name: str = "unknown"
    anomaly: bool = False
    shap_top: List[tuple] = field(default_factory=list)      # [(feature, value), ...]
    guardrail_events: List[GuardrailEvent] = field(default_factory=list)
    rationale: str = ""
    planner: str = "reactive"
    unsafe_applied: bool = False

    def changed(self) -> bool:
        return self.proposed != self.applied

    def as_row(self) -> Dict[str, Any]:
        return {
            "t_hours": self.t_hours,
            "proposed": self.proposed.label(),
            "applied": self.applied.label(),
            "risk_score": self.risk_score,
            "risk_class": self.risk_class,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "anomaly": self.anomaly,
            "guardrails": ";".join(e.rule for e in self.guardrail_events),
            "planner": self.planner,
            "unsafe_applied": self.unsafe_applied,
            "rationale": self.rationale,
        }
