"""Planners: ``reactive`` and ``predictive`` (learned MPC). Selected in ``agent.yaml``.

All three emit a *proposal*. None of them can act: every proposal goes through
``Guardrails.check`` before it reaches the pond.

A note on what "roll the forecaster forward under an action" means here, because it is the
one place where the honest engineering differs from the obvious reading. The LSTM is trained
on real monitoring sites; it has never seen an aerator and cannot be conditioned on one. So
the predictive planner rolls an **internal pond model** forward under each candidate action -
a second ``PondTwin`` the agent owns, which it has to keep aligned with reality from
observations alone (``InternalModel``), never by reading the real twin's latent state. The
LSTM then supplies a *residual correction*: the difference between its no-action forecast and
the internal model's no-action forecast is added to every candidate rollout, so systematic
internal-model bias is cancelled using something genuinely learned from real data.

That correction is **on by default**, and it is not free. The ablation measures it on every
evaluation run and it makes control worse: a network trained on temperate rivers,
residual-corrected onto a warm-water pond, injects more bias than it removes even damped by
half. It ships enabled as a deliberate choice - the learned forecaster is what the federated
half of the project exists to produce, and keeping it on the actuator path is what makes this
a learned controller rather than a twin rolled forward by classical control. Set
``predictive.use_forecaster: false`` for the better-scoring controller; the evaluation reports
both settings either way (``full`` vs ``no_lstm_correction`` in ``outputs/ablations.csv``), so
the price is quantified rather than hidden.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.runtime import load_config
from ..common.schema import Action, Forecast, Observation
from ..data.harmonize import derive_nh3
from ..twin import risk_bands as RB
from ..twin.pond import PondTwin
from . import actuators as ACT

PLAN_CHANNELS = ("do", "temp_c", "ph", "tan", "no2", "turbidity")


# ======================================================================================
# Internal model - the agent's own belief about the pond
# ======================================================================================
class InternalModel:
    """A ``PondTwin`` the agent owns, corrected toward observations by a complementary filter.

    It never reads the real twin's latent variables (algae, biomass, BOD). It starts from the
    nominal pond configuration and is nudged by what the probe reports, which means its latent
    state is genuinely uncertain - the same position a real operator is in.
    """

    def __init__(self, twin_cfg: Optional[Dict[str, Any]] = None, alpha: float = 0.35,
                 seed: int = 20240101, start_hour_of_day: float = 12.0):
        self.cfg = copy.deepcopy(twin_cfg or load_config("twin"))
        self.cfg["process_noise"] = False          # the belief model is the mean trajectory
        self.alpha = float(alpha)
        self.twin = PondTwin(cfg=self.cfg, seed=seed, start_hour_of_day=start_hour_of_day)

    def update(self, obs: Observation, applied: Action) -> None:
        """Advance the belief one sampling interval, then pull it toward the observation."""
        steps = max(int(round(self._sample_h() / self.twin.dt_h)), 1)
        for _ in range(steps):
            self.twin.step(applied)
        for c in PLAN_CHANNELS:
            v = obs.values.get(c)
            if v is None or not np.isfinite(v):
                continue
            self.twin.s[c] = (1.0 - self.alpha) * self.twin.s[c] + self.alpha * float(v)

    def rollout(self, action: Action, steps: int, hold_steps: int) -> Dict[str, np.ndarray]:
        """Simulate ``steps`` twin steps holding ``action`` for ``hold_steps``, then no_op."""
        sim = self.twin.copy()
        sim.scenario = None                     # the agent cannot know the scenario schedule
        out = {c: np.empty(steps) for c in PLAN_CHANNELS}
        out["nh3"] = np.empty(steps)
        sustained = replace(action, lime="none", salt="none") if (
            action.lime != "none" or action.salt != "none"
        ) else action
        for i in range(steps):
            act = action if i == 0 else (sustained if i < hold_steps else Action.no_op())
            st = sim.step(act)
            for c in PLAN_CHANNELS:
                out[c][i] = getattr(st, c)
            out["nh3"][i] = st.nh3
        out["chloride"] = np.full(steps, sim.s.get("chloride", 0.0))
        return out

    def _sample_h(self) -> float:
        return float(load_config("sensors")["sample_every_minutes"]) / 60.0


class LearnedWorldModel:
    """``InternalModel`` with its physics replaced by an action-conditioned LSTM.

    Composition, not inheritance: an ``InternalModel`` is kept alongside as the belief filter
    and as the warm-up fallback, because the network needs a full observation window before it
    can predict anything and the first few ticks of a run do not have one. Once the window
    fills, ``rollout`` is answered by the network - so the trajectory the planner scores, and
    therefore the action it applies, comes from the learned model.

    The interface is exactly ``InternalModel``'s (``update``, ``rollout``, ``.twin``), so
    nothing in ``PredictivePlanner`` or ``AgentRun`` has to know which one it is holding.
    """

    def __init__(self, world_model, internal: "InternalModel", clip: bool = True):
        self.wm = world_model
        self.inner = internal
        self.clip = clip
        self.history: List[List[float]] = []
        self.used_learned = 0
        self.used_fallback = 0

    @property
    def twin(self):
        # dt_h, chloride and the risk-band config are read off this by the planner.
        return self.inner.twin

    def update(self, obs, applied: Action) -> None:
        self.inner.update(obs, applied)     # keep the fallback belief aligned regardless
        row = [float(obs.values.get(c, np.nan)) for c in self.wm.channels]
        self.history.append(row)
        if len(self.history) > self.wm.window:
            self.history = self.history[-self.wm.window :]

    def rollout(self, action: Action, steps: int, hold_steps: int) -> Dict[str, np.ndarray]:
        if len(self.history) < self.wm.window or steps > self.wm.horizon:
            self.used_fallback += 1
            return self.inner.rollout(action, steps, hold_steps)

        window = np.asarray(self.history[-self.wm.window :], dtype=float)
        if not np.isfinite(window[-1]).all():
            # The delta encoding is measured from the last row; if a probe is out at exactly
            # that step the origin is undefined and the physics fallback is the honest answer.
            self.used_fallback += 1
            return self.inner.rollout(action, steps, hold_steps)

        self.used_learned += 1
        hold_fraction = float(hold_steps) / float(max(steps, 1))
        pred = self.wm.predict(window, action, hold_fraction)

        out: Dict[str, np.ndarray] = {}
        for c in PLAN_CHANNELS:
            v = np.asarray(pred[c][:steps], dtype=float)
            out[c] = np.clip(v, *_PHYS_BOUNDS[c]) if self.clip else v
        out["nh3"] = derive_nh3(out["tan"], out["ph"], out["temp_c"])
        # Chloride is not a predicted channel: it only moves when salt is dosed, and the twin
        # tracks it exactly. Carrying the belief value through keeps the guardrails' salt logic
        # reading the same quantity it always did.
        out["chloride"] = np.full(steps, self.inner.twin.s.get("chloride", 0.0))
        return out


# Physically admissible ranges. A neural network will happily predict a negative dissolved
# oxygen or a pH of 22 when pushed outside its training distribution, and an unclipped
# excursion like that would dominate the planner's cost function and drive a nonsense action.
_PHYS_BOUNDS = {
    "do": (0.0, 25.0),
    "temp_c": (0.0, 45.0),
    "ph": (4.0, 11.0),
    "tan": (0.0, 20.0),
    "no2": (0.0, 20.0),
    "turbidity": (0.0, 500.0),
}


# ======================================================================================
# Planners
# ======================================================================================
class Planner:
    name = "base"

    def propose(self, ctx: "PlanContext") -> Tuple[Action, str]:
        raise NotImplementedError


class PlanContext:
    """Everything a planner is allowed to see. No twin internals, no scenario schedule."""

    def __init__(
        self,
        t_hours: float,
        observed: Dict[str, float],
        forecast: Optional[Forecast] = None,
        forecast_summary: Optional[Dict[str, float]] = None,
        risk_class: str = "safe",
        risk_score: float = 0.0,
        regime_name: str = "unknown",
        anomaly: bool = False,
        shap_top: Optional[List[tuple]] = None,
        model: Optional[InternalModel] = None,
    ):
        self.t_hours = t_hours
        self.observed = observed
        self.forecast = forecast
        self.forecast_summary = forecast_summary or {}
        self.risk_class = risk_class
        self.risk_score = risk_score
        self.regime_name = regime_name
        self.anomaly = anomaly
        self.shap_top = shap_top or []
        self.model = model


class ReactivePlanner(Planner):
    """Transparent utility rules over the current reading and the forecast.

    This is the "baseline plus" controller: it is what a careful operator with a good probe
    and a two-hour forecast would do, and the predictive planner has to beat it.
    """

    name = "reactive"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = (cfg or load_config("agent"))["reactive"]

    def propose(self, ctx: PlanContext) -> Tuple[Action, str]:
        c = self.cfg
        o, fc = ctx.observed, ctx.forecast_summary
        do = o.get("do", np.nan)
        fc_do = fc.get("fc_do_min", do)
        nh3 = o.get("nh3", np.nan)
        no2 = o.get("no2", np.nan)
        ph = o.get("ph", np.nan)

        action = Action()
        why: List[str] = []

        low_do = min(v for v in (do, fc_do) if np.isfinite(v)) if np.isfinite(do) or np.isfinite(fc_do) else np.nan
        if np.isfinite(low_do):
            if low_do < float(c["do_act"]):
                action = ACT.merge(action, aerator=1.0, feed="skip")
                why.append(f"DO {low_do:.2f} below act threshold {c['do_act']}")
            elif low_do < float(c["do_warn"]):
                action = ACT.merge(action, aerator=0.5, feed="reduced")
                why.append(f"DO {low_do:.2f} below warn threshold {c['do_warn']}")

        if np.isfinite(nh3) and nh3 > float(c["nh3_warn"]):
            action = ACT.merge(action, feed="skip", exchange="small")
            why.append(f"un-ionised NH3 {nh3:.3f} > {c['nh3_warn']}")

        if np.isfinite(no2) and no2 > float(c["no2_warn"]):
            action = ACT.merge(action, salt="dose")
            why.append(f"NO2 {no2:.2f} > {c['no2_warn']} - chloride mitigation")

        if np.isfinite(ph) and ph < float(c["ph_low"]):
            action = ACT.merge(action, lime="dose")
            why.append(f"pH {ph:.2f} < {c['ph_low']}")
        elif np.isfinite(ph) and ph > float(c["ph_high"]):
            action = ACT.merge(action, exchange="small", feed="reduced")
            why.append(f"pH {ph:.2f} > {c['ph_high']}")

        return action, ("; ".join(why) if why else "all channels within thresholds")


class PredictivePlanner(Planner):
    """Learned MPC: score every candidate by its predicted consequences, take the argmin.

    cost = w_risk*time_in_risk + w_critical*time_critical
         + w_cost*intervention_cost + w_stab*volatility + w_feed*feed_penalty
    """

    name = "predictive"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, twin_cfg: Optional[Dict[str, Any]] = None):
        self.agent_cfg = cfg or load_config("agent")
        self.cfg = self.agent_cfg["predictive"]
        self.w = self.agent_cfg["cost_weights"]
        self.twin_cfg = twin_cfg or load_config("twin")
        self.candidates = ACT.candidates(self.agent_cfg)
        self.last_scores: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- scoring
    def evaluate(self, ctx: PlanContext, action: Action, correction: Optional[Dict[str, np.ndarray]] = None):
        steps = int(self.cfg["rollout_steps"])
        hold = int(self.cfg["hold_action_steps"])
        traj = ctx.model.rollout(action, steps, hold)
        if correction:
            for c, delta in correction.items():
                if c in traj:
                    traj[c] = traj[c] + delta
            traj["nh3"] = traj["nh3"]  # nh3 rides on tan/ph, already corrected through them

        import pandas as pd

        frame = pd.DataFrame({c: traj[c] for c in ("do", "ph", "nh3", "no2", "temp_c")})
        levels = RB.series_risk(frame, self.twin_cfg)
        dt_h = ctx.model.twin.dt_h
        time_warn = float((levels == "warning").sum()) * dt_h
        time_crit = float((levels == "critical").sum()) * dt_h
        volatility = float(np.mean([np.std(np.diff(traj[c])) for c in ("do", "ph")]))

        cost = (
            float(self.w["risk"]) * (time_warn + time_crit)
            + float(self.w["critical"]) * time_crit
            + float(self.w["intervention"]) * ACT.action_cost(action, self.agent_cfg)
            + float(self.w["stability"]) * volatility
            + float(self.w["feed_loss"]) * ACT.feed_penalty(action, self.agent_cfg)
        )
        return {
            "action": action,
            "cost": cost,
            "time_warning_h": time_warn,
            "time_critical_h": time_crit,
            "min_do": float(np.min(traj["do"])),
            "max_nh3": float(np.max(traj["nh3"])),
            "volatility": volatility,
            "summary": _summary_from_traj(traj),
        }

    def correction(self, ctx: PlanContext) -> Optional[Dict[str, np.ndarray]]:
        """LSTM no-action forecast minus internal-model no-action forecast, per channel.

        This is the only route by which something learned from real data enters the control
        decision, and it enters as a bias correction rather than as a control signal - which
        is the honest use of a model that has never seen an actuator.
        """
        if not self.cfg.get("use_forecaster", True) or ctx.forecast is None:
            return None
        steps = int(self.cfg["rollout_steps"])
        base = ctx.model.rollout(Action.no_op(), steps, 0)
        out: Dict[str, np.ndarray] = {}
        for c in ctx.forecast.channels:
            if c not in base:
                continue
            pred = ctx.forecast.channel(c)
            n = min(len(pred), steps)
            delta = np.zeros(steps)
            delta[:n] = pred[:n] - base[c][:n]
            if n < steps:
                delta[n:] = delta[n - 1] if n else 0.0
            # Damp the correction: the LSTM is a stream model, the pond is not a stream.
            out[c] = 0.5 * delta
        return out

    def propose(self, ctx: PlanContext) -> Tuple[Action, str]:
        if ctx.model is None:
            return Action.no_op(), "no internal model available"
        corr = self.correction(ctx)
        scored = [self.evaluate(ctx, a, corr) for a in self.candidates]
        scored.sort(key=lambda r: r["cost"])
        self.last_scores = scored
        best, do_nothing = scored[0], next(r for r in scored if r["action"].is_no_op())

        gain = do_nothing["cost"] - best["cost"]
        why = (
            f"MPC over {len(self.candidates)} candidates, {self.cfg['rollout_steps']} steps: "
            f"'{ACT.describe(best['action'])}' costs {best['cost']:.2f} vs {do_nothing['cost']:.2f} "
            f"for holding (gain {gain:.2f}); predicted min DO {best['min_do']:.2f} mg/L, "
            f"{best['time_critical_h']:.1f} h critical vs {do_nothing['time_critical_h']:.1f} h"
        )
        if corr is not None:
            why += "; LSTM residual correction applied"
        return best["action"], why


# ======================================================================================
def make_planner(mode: Optional[str] = None, agent_cfg=None, twin_cfg=None) -> Planner:
    agent_cfg = agent_cfg or load_config("agent")
    mode = (mode or agent_cfg["planner"]).lower()
    if mode == "reactive":
        return ReactivePlanner(agent_cfg)
    if mode == "predictive":
        return PredictivePlanner(agent_cfg, twin_cfg)
    raise ValueError(f"unknown planner '{mode}'")


def _summary_from_traj(traj: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c, arr in traj.items():
        if arr.ndim != 1 or len(arr) == 0:
            continue
        out[f"fc_{c}_min"] = float(np.min(arr))
        out[f"fc_{c}_max"] = float(np.max(arr))
        out[f"fc_{c}_mean"] = float(np.mean(arr))
        out[f"fc_{c}_end"] = float(arr[-1])
    return out
