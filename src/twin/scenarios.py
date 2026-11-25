"""Scenario definitions: initial-state overrides + a schedule of perturbations + a seed.

A perturbation is a (kind, window, value) triple. ``ScenarioSpec.active(t)`` returns the
perturbations in force at simulated hour ``t`` as a flat dict the twin reads each step.
Sensor-fault perturbations are handled by ``SensorStream``, not the twin - they corrupt the
observation, not the water.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..common.runtime import load_config

# Kinds the twin consumes directly.
TWIN_KINDS = {
    "temp_offset",
    "cloud",
    "feed_pulse",
    "algae_seed",
    "algae_growth",
    "algae_death",
    "alkalinity_set",
    "aerator_fault",
    "inflow_temp",
}
# Kinds the sensor layer consumes.
SENSOR_KINDS = {"sensor_fault"}

# One-shot kinds fire once when the clock passes start_h rather than persisting.
IMPULSE_KINDS = {"feed_pulse", "algae_seed", "alkalinity_set"}


@dataclass
class Perturbation:
    kind: str
    start_h: float = 0.0
    end_h: float = float("inf")
    value: float = 1.0
    channel: Optional[str] = None
    mode: Optional[str] = None
    fired: bool = False

    def in_window(self, t: float, dt: float) -> bool:
        if self.kind in IMPULSE_KINDS:
            return (not self.fired) and t >= self.start_h - 1e-9
        return self.start_h - 1e-9 <= t < self.end_h


@dataclass
class ScenarioSpec:
    scenario_id: str
    label: str
    description: str
    seed: int
    duration_h: float
    start_hour_of_day: float
    initial: Dict[str, float] = field(default_factory=dict)
    perturbations: List[Perturbation] = field(default_factory=list)
    dt_h: float = 5.0 / 60.0

    # -------------------------------------------------------------------- queries
    def active(self, t_hours: float) -> Dict[str, Any]:
        """Twin-facing perturbations in force at ``t_hours``, flattened to {kind: value}."""
        out: Dict[str, Any] = {}
        for p in self.perturbations:
            if p.kind not in TWIN_KINDS or not p.in_window(t_hours, self.dt_h):
                continue
            if p.kind in IMPULSE_KINDS:
                p.fired = True
            out[p.kind] = p.value
        return out

    def sensor_faults(self, t_hours: float) -> List[Perturbation]:
        return [
            p
            for p in self.perturbations
            if p.kind in SENSOR_KINDS and p.start_h - 1e-9 <= t_hours < p.end_h
        ]

    def reset(self) -> "ScenarioSpec":
        """Clear one-shot fired flags so the spec can drive a second run."""
        for p in self.perturbations:
            p.fired = False
        return self

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_h / self.dt_h))


def load_scenarios(cfg: Optional[Dict[str, Any]] = None, dt_h: Optional[float] = None) -> Dict[str, ScenarioSpec]:
    cfg = cfg or load_config("scenarios")
    if dt_h is None:
        dt_h = float(load_config("twin")["dt_minutes"]) / 60.0
    defaults = cfg.get("defaults", {})
    out: Dict[str, ScenarioSpec] = {}
    for sid, spec in cfg["scenarios"].items():
        perts = [
            Perturbation(
                kind=p["kind"],
                start_h=float(p.get("start_h", 0.0)),
                end_h=float(p.get("end_h", float("inf"))),
                value=float(p.get("value", 1.0)),
                channel=p.get("channel"),
                mode=p.get("mode"),
            )
            for p in (spec.get("perturbations") or [])
        ]
        out[sid] = ScenarioSpec(
            scenario_id=sid,
            label=spec.get("label", sid),
            description=spec.get("description", ""),
            seed=int(spec.get("seed", defaults.get("seed", 20240101))),
            duration_h=float(spec.get("duration_h", defaults.get("duration_h", 48.0))),
            start_hour_of_day=float(
                spec.get("start_hour_of_day", defaults.get("start_hour_of_day", 12.0))
            ),
            initial={k: float(v) for k, v in (spec.get("initial") or {}).items()},
            perturbations=perts,
            dt_h=dt_h,
        )
    return out


def get_scenario(scenario_id: str, **kwargs) -> ScenarioSpec:
    scenarios = load_scenarios(**kwargs)
    if scenario_id not in scenarios:
        raise KeyError(f"unknown scenario '{scenario_id}'. Known: {sorted(scenarios)}")
    return scenarios[scenario_id]


def scenario_ids() -> List[str]:
    return list(load_scenarios().keys())
