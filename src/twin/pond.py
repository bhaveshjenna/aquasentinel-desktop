"""``PondTwin`` - the interactive environment the agent acts on.

Discrete-time Euler integration at ``twin.dt_minutes``. ``step(action)`` advances the pond by
one step and returns the new ground-truth ``State``. Actions must produce visible,
physically-plausible responses: that responsiveness is the whole point of the twin, because
the dashboard's money shot is the agent's trajectory against a no-action shadow run.

The twin is a SIMULATION. It is calibrated against statistics of real monitoring data
(``calibrate.py``) but it is not a validated pond model, and nothing here should be read as a
claim about a real farm.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.runtime import load_config
from ..common.schema import Action, State
from . import dynamics as D
from .scenarios import ScenarioSpec

# Channels carrying Ornstein-Uhlenbeck process noise.
NOISE_CHANNELS = ("do", "temp_c", "ph", "tan", "no2", "turbidity")


class PondTwin:
    """A single pond. Deterministic given (config, scenario, seed)."""

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        scenario: Optional[ScenarioSpec] = None,
        seed: int = 20240101,
        start_hour_of_day: float = 12.0,
    ) -> None:
        self.cfg = copy.deepcopy(cfg or load_config("twin"))
        self.scenario = scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.dt_h = float(self.cfg["dt_minutes"]) / 60.0
        self.start_hour_of_day = float(start_hour_of_day)

        self.t_hours = 0.0
        self.step_count = 0
        self.s: Dict[str, float] = {k: float(v) for k, v in self.cfg["initial"].items()}
        self.s.setdefault("chloride", 0.0)
        if scenario is not None:
            self.s.update({k: float(v) for k, v in scenario.initial.items()})

        self._noise = {c: 0.0 for c in NOISE_CHANNELS}
        self._exchange_hours_left = 0.0
        self._exchange_rate = 0.0
        self._lime_pool = 0.0          # kg of undissolved lime
        self._fed_today: set = set()   # feed hours already served on the current day
        self._day = 0
        self.last_terms: Dict[str, float] = {}
        self.last_action = Action.no_op()
        self.delivered_aerator = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def hour_of_day(self) -> float:
        return (self.start_hour_of_day + self.t_hours) % 24.0

    def state(self) -> State:
        nh3 = self.s["tan"] * D.nh3_fraction(self.s["ph"], self.s["temp_c"])
        return State(
            t_hours=self.t_hours,
            do=self.s["do"],
            temp_c=self.s["temp_c"],
            ph=self.s["ph"],
            tan=self.s["tan"],
            nh3=nh3,
            no2=self.s["no2"],
            turbidity=self.s["turbidity"],
            conductivity=600.0 + 1.8 * self.s.get("chloride", 0.0),
            algae=self.s["algae"],
            biomass=self.s["biomass"],
            alkalinity=self.s["alkalinity"],
            bod_pool=self.s["bod_pool"],
            feed_pool=self.s["feed_pool"],
            chloride=self.s.get("chloride", 0.0),
        )

    def copy(self) -> "PondTwin":
        """A deep, independently-evolving fork - this is what the counterfactual runs on.

        The RNG is cloned by state, so the shadow twin sees the *same* noise realisation as
        the live one. Any divergence between them is caused by actions, not by luck.
        """
        twin = PondTwin.__new__(PondTwin)
        twin.__dict__.update({k: copy.deepcopy(v) for k, v in self.__dict__.items()})
        twin.rng = np.random.default_rng()
        twin.rng.bit_generator.state = self.rng.bit_generator.state
        return twin

    # ------------------------------------------------------------------ the step
    def step(self, action: Optional[Action] = None) -> State:
        action = action or Action.no_op()
        self.last_action = action
        cfg, dt = self.cfg, self.dt_h
        pert = self.scenario.active(self.t_hours) if self.scenario else {}

        # --- forcing -------------------------------------------------------------
        light = D.light_fraction(
            self.hour_of_day,
            cfg["light"],
            cloud=float(pert.get("cloud", cfg["light"]["cloud_factor"])),
        )
        temp_target = D.temperature_forcing(
            self.hour_of_day, cfg["temperature"], offset_c=float(pert.get("temp_offset", 0.0))
        )

        # --- discrete action effects (feeding, dosing, exchange scheduling) -------
        self._apply_discrete(action, pert)

        # An aerator_fault perturbation means the command is issued but nothing is delivered.
        self.delivered_aerator = 0.0 if pert.get("aerator_fault") else float(action.aerator)

        # --- rates ---------------------------------------------------------------
        n = D.nitrogen_rates(self.s, cfg)
        ox = D.do_rate(self.s, cfg, light, self.delivered_aerator, n["tan_nitrified"])
        alg = D.algae_rate(
            self.s,
            cfg,
            light,
            growth_multiplier=float(pert.get("algae_growth", 1.0)),
            death_multiplier=float(pert.get("algae_death", 1.0)),
        )
        d_ph = D.ph_rate(self.s, cfg, ox["photosynthesis"], -ox["respiration"] + -ox["bod"])
        d_alk = D.alkalinity_rate(self.s, cfg, n["tan_nitrified"])
        d_turb = D.turbidity_rate(self.s, cfg, alg["d_algae"])

        # --- integrate -----------------------------------------------------------
        self.s["temp_c"] += (
            float(cfg["temperature"]["relax_per_h"]) * (temp_target - self.s["temp_c"]) * dt
        )
        self.s["do"] += ox["total"] * dt
        self.s["tan"] += n["d_tan"] * dt
        self.s["no2"] += n["d_no2"] * dt
        self.s["ph"] += d_ph * dt
        self.s["alkalinity"] += d_alk * dt
        self.s["algae"] += alg["d_algae"] * dt
        self.s["turbidity"] += d_turb * dt

        # feed pool -> nitrogen + oxygen demand
        released = n["feed_release_kg_per_h"] * dt
        self.s["feed_pool"] = max(self.s["feed_pool"] - released, 0.0)
        self.s["bod_pool"] += (
            released * float(cfg["ammonia"]["feed_bod_per_kg"])
            + alg["death"] * float(cfg["algae"]["death_to_bod"]) * dt
        )
        # ox["bod"] is already signed negative (a DO sink); exerting it drains the pool.
        self.s["bod_pool"] = max(self.s["bod_pool"] + ox["bod"] * dt, 0.0)

        # lime dissolution and chloride decay
        self._dissolve_lime(dt)
        self.s["chloride"] *= (1.0 - float(cfg["actions"]["chloride_decay_per_h"]) * dt)

        # water exchange dilutes toward the known inflow composition
        self._apply_exchange(dt, pert)

        # --- process noise + clamps ---------------------------------------------
        if cfg.get("process_noise", True):
            self._add_noise(dt)
        self._clamp()

        self.t_hours += dt
        self.step_count += 1
        self.last_terms = {
            **{f"do_{k}": v for k, v in ox.items()},
            "light": light,
            "temp_target": temp_target,
            "nh3_fraction": D.nh3_fraction(self.s["ph"], self.s["temp_c"]),
            "algae_growth": alg["growth"],
            "algae_death": alg["death"],
            "delivered_aerator": self.delivered_aerator,
        }
        return self.state()

    def run(self, hours: float, action: Optional[Action] = None) -> List[State]:
        return [self.step(action) for _ in range(int(round(hours / self.dt_h)))]

    # ------------------------------------------------------------------ helpers
    def _apply_discrete(self, action: Action, pert: Dict[str, Any]) -> None:
        acts = self.cfg["actions"]

        # feeding happens at scheduled hours; the action decides how much of it goes in
        day = int((self.start_hour_of_day + self.t_hours) // 24)
        if day != self._day:
            self._day = day
            self._fed_today.clear()
        for fh in acts["feed_hours"]:
            key = float(fh)
            if key in self._fed_today:
                continue
            if abs(self.hour_of_day - key) < self.dt_h:
                per_event = float(acts["feed_normal_kg_per_day"]) / max(len(acts["feed_hours"]), 1)
                factor = {"normal": 1.0, "reduced": float(acts["feed_reduced_frac"]), "skip": 0.0}[
                    action.feed
                ]
                self.s["feed_pool"] += per_event * factor
                self._fed_today.add(key)

        # a scenario feed_pulse is an operator error, not an agent action
        if "feed_pulse" in pert:
            self.s["feed_pool"] += float(pert["feed_pulse"])
        if "algae_seed" in pert:
            self.s["algae"] *= float(pert["algae_seed"])
        if "alkalinity_set" in pert:
            self.s["alkalinity"] = float(pert["alkalinity_set"])

        if action.lime == "dose":
            self._lime_pool += float(acts["lime_dose_kg"])
        if action.salt == "dose":
            self.s["chloride"] += float(acts["salt_dose_kg"]) * float(acts["salt_chloride_per_kg"])

        if action.exchange != "none" and self._exchange_hours_left <= 0.0:
            self._exchange_rate = float(
                acts["exchange_small_frac_per_h"]
                if action.exchange == "small"
                else acts["exchange_large_frac_per_h"]
            )
            self._exchange_hours_left = float(acts["exchange_duration_h"])

    def _dissolve_lime(self, dt: float) -> None:
        acts = self.cfg["actions"]
        if self._lime_pool <= 0.0:
            return
        dissolved = min(self._lime_pool, self._lime_pool * float(acts["lime_release_per_h"]) * dt)
        self._lime_pool -= dissolved
        self.s["alkalinity"] += dissolved * float(acts["lime_alkalinity_per_kg"])

    def _apply_exchange(self, dt: float, pert: Dict[str, Any]) -> None:
        if self._exchange_hours_left <= 0.0:
            return
        f = self._exchange_rate * dt
        inflow_temp = float(pert.get("inflow_temp", self.cfg["temperature"]["inflow_c"]))
        inflow = {
            "temp_c": inflow_temp,
            "do": D.do_saturation(inflow_temp, self.cfg["pond"]["salinity_ppt"]) * 0.95,
            "tan": float(self.cfg["ammonia"]["inflow_tan"]),
            "no2": float(self.cfg["ammonia"]["inflow_no2"]),
            "turbidity": float(self.cfg["turbidity"]["inflow_ntu"]),
            "ph": float(self.cfg["ph"]["equilibrium"]),
            "algae": 0.05,
        }
        for k, target in inflow.items():
            self.s[k] += f * (target - self.s[k])
        self._exchange_hours_left -= dt

    def _add_noise(self, dt: float) -> None:
        noise_cfg = self.cfg["noise"]
        theta = float(noise_cfg["ou_theta_per_h"])
        for c in NOISE_CHANNELS:
            sigma = float(noise_cfg.get(c, 0.0))
            if sigma <= 0.0:
                continue
            self.s[c], self._noise[c] = D.ou_step(
                self.s[c], self._noise[c], sigma, theta, dt, self.rng
            )

    def _clamp(self) -> None:
        self.s["do"] = max(self.s["do"], float(self.cfg["oxygen"]["min_do"]))
        self.s["tan"] = max(self.s["tan"], 0.0)
        self.s["no2"] = max(self.s["no2"], 0.0)
        self.s["algae"] = max(self.s["algae"], 0.0)
        self.s["turbidity"] = max(self.s["turbidity"], 0.0)
        self.s["bod_pool"] = max(self.s["bod_pool"], 0.0)
        self.s["chloride"] = max(self.s["chloride"], 0.0)
        self.s["alkalinity"] = max(self.s["alkalinity"], float(self.cfg["ph"]["min_alkalinity"]))
        self.s["ph"] = float(np.clip(self.s["ph"], 3.0, 12.0))
