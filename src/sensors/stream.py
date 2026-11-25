"""``SensorStream`` - the layer that turns twin ground truth into a plausible "live" feed.

Perception never sees the twin's state directly. It sees this: quantised, biased, noisy,
occasionally missing. The ``sensor_fault`` scenario acts here rather than in the twin, because
a broken probe corrupts the *reading*, not the water - which is exactly what the anomaly
detector has to learn to tell apart from a real excursion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..common.runtime import load_config
from ..common.schema import Observation, State
from ..twin.pond import PondTwin
from ..twin.scenarios import ScenarioSpec

STREAM_CHANNELS = ("do", "temp_c", "ph", "tan", "no2", "turbidity")


class SensorStream:
    """Samples a twin every ``sample_every_minutes`` and yields ``Observation``s.

    Sampling is coarser than the twin's integration step, so the twin can be integrated at
    5-minute resolution while the "probe" reports at whatever cadence the operator has.
    """

    def __init__(
        self,
        twin: PondTwin,
        cfg: Optional[Dict[str, Any]] = None,
        scenario: Optional[ScenarioSpec] = None,
        seed: int = 20240101,
    ) -> None:
        self.twin = twin
        self.cfg = cfg or load_config("sensors")
        self.scenario = scenario
        self.rng = np.random.default_rng(seed + 977)

        sample_min = float(self.cfg["sample_every_minutes"])
        twin_min = float(twin.cfg["dt_minutes"])
        self.steps_per_sample = max(int(round(sample_min / twin_min)), 1)
        self.sample_minutes = sample_min

        # A per-run calibration offset, drawn once. This is why two identical ponds with two
        # identical probes still disagree, and why the forecaster must not trust absolute level.
        self.bias = {
            c: float(self.rng.normal(0.0, float(self.cfg["bias"].get(c, 0.0))))
            for c in STREAM_CHANNELS
        }
        self._stuck_value: Dict[str, Optional[float]] = {c: None for c in STREAM_CHANNELS}
        self.history: List[Observation] = []
        self.true_history: List[State] = []

    # ------------------------------------------------------------------------------
    def tick(self, action=None) -> tuple:
        """Advance the twin by one sampling interval; return (true State, Observation)."""
        state = None
        for _ in range(self.steps_per_sample):
            state = self.twin.step(action)
        obs = self.sample(state)
        self.true_history.append(state)
        self.history.append(obs)
        return state, obs

    def sample(self, state: State) -> Observation:
        values: Dict[str, float] = {}
        mask: Dict[str, bool] = {}
        faults = self._faults_at(state.t_hours)

        for c in STREAM_CHANNELS:
            truth = float(getattr(state, c, np.nan))
            noise = float(self.rng.normal(0.0, float(self.cfg["noise_std"].get(c, 0.0))))
            value = truth + noise + self.bias[c]

            fault = faults.get(c)
            if fault is not None:
                value, dropped = self._apply_fault(c, value, fault)
                if dropped:
                    values[c] = float("nan")
                    mask[c] = False
                    continue
            else:
                self._stuck_value[c] = None

            if self.rng.random() < float(self.cfg["dropout_prob"]):
                values[c] = float("nan")
                mask[c] = False
                continue

            step = float(self.cfg["quantization"].get(c, 0.0))
            if step > 0:
                value = round(value / step) * step
            values[c] = float(value)
            mask[c] = True

        # Conductivity is reported by real sites but is not a modelled control channel; it is
        # carried through so the schema matches the real-data path exactly.
        values["conductivity"] = float(state.conductivity)
        mask["conductivity"] = True
        return Observation(t_hours=state.t_hours, values=values, mask=mask)

    # ------------------------------------------------------------------------------
    def _faults_at(self, t_hours: float) -> Dict[str, Any]:
        if self.scenario is None:
            return {}
        return {p.channel: p for p in self.scenario.sensor_faults(t_hours) if p.channel}

    def _apply_fault(self, channel: str, value: float, fault) -> tuple:
        """Returns (value, dropped). Modes: dropout | spike | stuck | drift."""
        mode = (fault.mode or "dropout").lower()
        if mode == "dropout":
            return value, self.rng.random() < float(fault.value)
        if mode == "spike":
            if self.rng.random() < 0.35:
                sign = 1.0 if self.rng.random() < 0.5 else -1.0
                return value + sign * float(fault.value), False
            return value, False
        if mode == "stuck":
            if self._stuck_value[channel] is None:
                self._stuck_value[channel] = value
            return float(self._stuck_value[channel]), False
        if mode == "drift":
            return value + float(fault.value), False
        return value, False

    # ------------------------------------------------------------------------------
    def frame(self):
        """History as a pandas DataFrame - what the perception stack windows over."""
        import pandas as pd

        if not self.history:
            return pd.DataFrame(columns=["t_hours", *STREAM_CHANNELS])
        rows = [{"t_hours": o.t_hours, **o.values} for o in self.history]
        return pd.DataFrame(rows)

    def truth_frame(self):
        """Twin ground truth. Available in the simulation only - never on a real farm."""
        import pandas as pd

        rows = [
            {
                "t_hours": s.t_hours,
                "do": s.do,
                "temp_c": s.temp_c,
                "ph": s.ph,
                "tan": s.tan,
                "nh3": s.nh3,
                "no2": s.no2,
                "turbidity": s.turbidity,
                "algae": s.algae,
                "alkalinity": s.alkalinity,
            }
            for s in self.true_history
        ]
        return pd.DataFrame(rows)


def make_stream(scenario_id: str, seed: Optional[int] = None, twin_cfg=None) -> SensorStream:
    """Convenience: scenario id -> a wired (twin, stream) pair ready to tick."""
    from ..twin.scenarios import get_scenario

    scenario = get_scenario(scenario_id).reset()
    seed = int(seed if seed is not None else scenario.seed)
    twin = PondTwin(
        cfg=twin_cfg,
        scenario=scenario,
        seed=seed,
        start_hour_of_day=scenario.start_hour_of_day,
    )
    return SensorStream(twin, scenario=scenario, seed=seed)
