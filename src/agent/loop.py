"""The control loop: perceive -> plan -> guardrail-check -> act -> observe.

``AgentRun`` owns a live pond and a **shadow** pond forked from it at t=0 with the same seed
and the same noise realisation, driven by ``do_nothing``. Every divergence between the two is
therefore attributable to the agent's actions and not to luck - that side-by-side is the
dashboard's counterfactual panel and the evaluation's core comparison.

Perception is injected, not imported: a run works with no models at all (Phase 3), with a
forecaster only, or with the full forecast + regime + risk + SHAP stack (Phase 4+).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.runtime import load_config
from ..common.schema import Action, Decision, Forecast, Observation, State
from ..sensors.stream import SensorStream
from ..twin import risk_bands as RB
from ..twin.pond import PondTwin
from ..twin.scenarios import ScenarioSpec, get_scenario
from . import actuators as ACT
from .guardrails import Guardrails
from .policy import InternalModel, LearnedWorldModel, PlanContext, Planner, make_planner


@dataclass
class Perception:
    """Optional perception stack. Any field may be None; the loop degrades gracefully."""

    forecaster: Any = None
    regimes: Any = None
    risk: Any = None
    explainer: Any = None

    #: Failures are recorded here rather than swallowed. A perception stage that quietly
    #: returns nothing looks identical to one that had nothing to say, and the loop is
    #: designed to degrade rather than crash - so without this the degradation is invisible.
    errors: Dict[str, str] = field(default_factory=dict)

    def run(self, history: pd.DataFrame, observed: Dict[str, float], twin_cfg) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "forecast": None,
            "forecast_summary": {},
            "regime": -1,
            "regime_name": "unknown",
            "anomaly": False,
            "risk_class": "safe",
            "risk_score": 0.0,
            "shap_top": [],
        }
        if self.forecaster is not None:
            try:
                fc: Optional[Forecast] = self.forecaster.predict(history)
            except Exception as exc:
                self.errors["forecaster"] = f"{type(exc).__name__}: {exc}"
                fc = None
            if fc is not None:
                out["forecast"] = fc
                summary = fc.summary()
                # NH3 is derived, so its forecast summary has to be derived too.
                summary.update(_nh3_forecast_summary(fc))
                out["forecast_summary"] = summary
        if self.regimes is not None:
            try:
                label, name, anomaly = self.regimes.assign(history)
                out.update({"regime": label, "regime_name": name, "anomaly": anomaly})
            except Exception as exc:
                self.errors["regimes"] = f"{type(exc).__name__}: {exc}"
        if self.risk is not None:
            try:
                cls, score = self.risk.predict(observed, out["forecast_summary"])
                out["risk_class"], out["risk_score"] = cls, score
            except Exception as exc:
                self.errors["risk"] = f"{type(exc).__name__}: {exc}"
        else:
            cls = RB.worst_forecast_risk(out["forecast_summary"], twin_cfg) if out["forecast_summary"] else "safe"
            now_cls, _ = RB.state_risk(observed, twin_cfg)
            out["risk_class"] = max([cls, now_cls], key=lambda l: ["safe", "warning", "critical"].index(l))
            out["risk_score"] = RB.risk_score(out["risk_class"])
        if self.explainer is not None and self.risk is not None:
            try:
                out["shap_top"] = self.explainer.top_drivers(observed, out["forecast_summary"])
            except Exception as exc:
                self.errors["explainer"] = f"{type(exc).__name__}: {exc}"
        return out


@dataclass
class RunResult:
    scenario_id: str
    controller: str
    seed: int
    states: List[State] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    shadow_states: List[State] = field(default_factory=list)

    # -------------------------------------------------------------------- frames
    def frame(self) -> pd.DataFrame:
        return _states_frame(self.states)

    def shadow_frame(self) -> pd.DataFrame:
        return _states_frame(self.shadow_states)

    def decision_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.as_row() for d in self.decisions])

    # -------------------------------------------------------------------- metrics
    def metrics(self, twin_cfg=None) -> Dict[str, float]:
        dt_h = (self.states[1].t_hours - self.states[0].t_hours) if len(self.states) > 1 else 5 / 60
        live = RB.time_in_risk(self.frame(), twin_cfg, dt_h)
        shadow = RB.time_in_risk(self.shadow_frame(), twin_cfg, dt_h) if self.shadow_states else {}
        interventions = sum(1 for d in self.decisions if not d.applied.is_no_op())
        cost = sum(ACT.action_cost(d.applied) for d in self.decisions)
        return {
            "scenario": self.scenario_id,
            "controller": self.controller,
            "seed": self.seed,
            "duration_h": round(self.states[-1].t_hours, 2) if self.states else 0.0,
            "time_warning_h": round(live["warning"], 3),
            "time_critical_h": round(live["critical"], 3),
            "time_in_risk_h": round(live["warning"] + live["critical"], 3),
            "shadow_time_in_risk_h": round(shadow.get("warning", 0) + shadow.get("critical", 0), 3),
            "interventions": interventions,
            "intervention_cost": round(cost, 2),
            "guardrail_events": sum(len(d.guardrail_events) for d in self.decisions),
            "unsafe_applied": sum(int(d.unsafe_applied) for d in self.decisions),
            "min_do": round(min((s.do for s in self.states), default=float("nan")), 3),
            "max_nh3": round(max((s.nh3 for s in self.states), default=float("nan")), 4),
            **{f"risk_h_{ch}": round(v, 3) for ch, v in
               RB.time_in_risk_by_channel(self.frame(), twin_cfg, dt_h).items()},
        }



def _maybe_wrap_world_model(model: "InternalModel", agent_cfg: Dict[str, Any]):
    """Swap the hand-written pond model for the learned one when configured and available.

    Absence is not an error: if the weights have not been trained yet the planner keeps its
    physics model and the run is still valid, just not a learned-dynamics run. That keeps
    `python -m src.eval.run_eval` working on a fresh clone before
    `scripts/pretrain_world_model.py` has been run.
    """
    pcfg = agent_cfg.get("predictive", {})
    if not pcfg.get("world_model", False):
        return model
    from ..common.runtime import MODELS_DIR, load_config
    from ..perception.world_model import ActionWorldModel

    source = str(pcfg.get("world_model_source", "federated"))
    path = MODELS_DIR / f"world_model_{source}.pt"
    if not path.exists():
        return model
    try:
        wm = ActionWorldModel.load(path)
    except Exception:
        return model
    clip = bool(load_config("models").get("world_model", {}).get("clip_to_physical_bounds", True))
    return LearnedWorldModel(wm, model, clip=clip)

class AgentRun:
    """One scenario run. Tick it step by step (dashboard) or to completion (evaluation)."""

    def __init__(
        self,
        scenario: str | ScenarioSpec,
        controller: str = "predictive",
        seed: Optional[int] = None,
        perception: Optional[Perception] = None,
        twin_cfg: Optional[Dict[str, Any]] = None,
        agent_cfg: Optional[Dict[str, Any]] = None,
        with_shadow: bool = True,
        guardrails_enabled: bool = True,
    ) -> None:
        self.scenario = get_scenario(scenario).reset() if isinstance(scenario, str) else scenario.reset()
        self.agent_cfg = agent_cfg or load_config("agent")
        self.twin_cfg = twin_cfg or load_config("twin")
        self.controller = controller
        self.seed = int(seed if seed is not None else self.scenario.seed)
        self.perception = perception or Perception()

        twin = PondTwin(
            cfg=self.twin_cfg,
            scenario=self.scenario,
            seed=self.seed,
            start_hour_of_day=self.scenario.start_hour_of_day,
        )
        self.stream = SensorStream(twin, scenario=self.scenario, seed=self.seed)

        # The shadow is forked before the first step, so it experiences the identical
        # perturbation schedule and the identical noise draws.
        self.shadow: Optional[SensorStream] = None
        if with_shadow:
            shadow_scn = get_scenario(self.scenario.scenario_id).reset()
            shadow_twin = PondTwin(
                cfg=self.twin_cfg,
                scenario=shadow_scn,
                seed=self.seed,
                start_hour_of_day=self.scenario.start_hour_of_day,
            )
            self.shadow = SensorStream(shadow_twin, scenario=shadow_scn, seed=self.seed)

        self.planner: Optional[Planner] = None
        self.model: Optional[InternalModel] = None
        if controller not in ("do_nothing", "static_threshold", "oracle"):
            self.planner = make_planner(controller, self.agent_cfg, self.twin_cfg)
            self.model = InternalModel(
                self.twin_cfg, seed=self.seed, start_hour_of_day=self.scenario.start_hour_of_day
            )
            self.model = _maybe_wrap_world_model(self.model, self.agent_cfg)

        self.guardrails = Guardrails(self.agent_cfg, self.twin_cfg)
        # `do_nothing` is a reference trajectory, not a controller: guardrails would rescue it
        # and destroy the comparison the whole evaluation rests on.
        self.guardrails.enabled = (
            guardrails_enabled
            and controller != "do_nothing"
            and bool(self.agent_cfg["guardrails"].get("enabled", True))
        )

        sample_min = float(load_config("sensors")["sample_every_minutes"])
        self.ticks_per_decision = max(
            int(round(float(self.agent_cfg["decide_every_minutes"]) / sample_min)), 1
        )
        self.tick_index = 0
        self.current_action = Action.no_op()
        self.result = RunResult(self.scenario.scenario_id, controller, self.seed)
        self.last_perception: Dict[str, Any] = {}

    # ---------------------------------------------------------------------- ticking
    @property
    def n_ticks(self) -> int:
        sample_h = float(load_config("sensors")["sample_every_minutes"]) / 60.0
        return int(round(self.scenario.duration_h / sample_h))

    def done(self) -> bool:
        return self.tick_index >= self.n_ticks

    def tick(self) -> Decision:
        state, obs = self.stream.tick(self.current_action)
        if self.shadow is not None:
            self.shadow.tick(Action.no_op())
            self.result.shadow_states.append(self.shadow.true_history[-1])
        self.result.states.append(state)
        self.result.observations.append(obs)

        observed = self._observed_dict(obs, state)
        if self.model is not None:
            self.model.update(obs, self.current_action)

        decide = self.tick_index % self.ticks_per_decision == 0
        if decide:
            decision = self._decide(state, obs, observed)
            self.current_action = decision.applied
        else:
            # Between decision points the planner is idle but the safety layer is not.
            # Sustained actions (aerator, exchange, feed) are held, but discrete impulse
            # treatments (lime, salt) are one-shot doses and must not be repeated on intermediate ticks.
            sustained = replace(self.current_action, lime="none", salt="none") if (
                self.current_action.lime != "none" or self.current_action.salt != "none"
            ) else self.current_action
            held, events = self.guardrails.safety_only(
                sustained,
                state.t_hours,
                observed,
                self.last_perception.get("forecast_summary", {}),
            )
            self.current_action = held
            decision = Decision(
                t_hours=state.t_hours,
                proposed=self.current_action,
                applied=held,
                guardrail_events=events,
                risk_class=self.last_perception.get("risk_class", "safe"),
                risk_score=self.last_perception.get("risk_score", 0.0),
                regime=self.last_perception.get("regime", -1),
                regime_name=self.last_perception.get("regime_name", "unknown"),
                anomaly=bool(self.last_perception.get("anomaly", False)),
                # Carry the attribution forward too. Planning is 3x coarser than sampling, so
                # without this two thirds of decisions have no SHAP and the dashboard panel
                # blanks out for most ticks - the explanation would look intermittent when it
                # is simply not recomputed.
                shap_top=self.last_perception.get("shap_top", []),
                planner=self.controller,
                rationale=(
                    "safety layer: " + "; ".join(e.rule for e in events)
                    if events
                    else "holding previous action between decision points"
                ),
            )
        decision.unsafe_applied = self._is_unsafe(state, decision)
        self.result.decisions.append(decision)
        self.tick_index += 1
        return decision

    def run(self) -> RunResult:
        while not self.done():
            self.tick()
        return self.result

    # ---------------------------------------------------------------------- planning
    def _decide(self, state: State, obs: Observation, observed: Dict[str, float]) -> Decision:
        history = self.stream.frame()
        percept = self.perception.run(history, observed, self.twin_cfg)
        self.last_perception = percept

        proposed, rationale = self._propose(observed, percept)

        predict_fn = None
        if self.model is not None:
            steps = int(self.agent_cfg["predictive"]["rollout_steps"])
            hold = int(self.agent_cfg["predictive"]["hold_action_steps"])

            def predict_fn(action: Action, _s=steps, _h=hold):  # noqa: F811
                traj = self.model.rollout(action, _s, _h)
                return {
                    **{f"fc_{c}_min": float(np.min(v)) for c, v in traj.items() if v.ndim == 1},
                    **{f"fc_{c}_max": float(np.max(v)) for c, v in traj.items() if v.ndim == 1},
                }

        applied, events = self.guardrails.check(
            proposed, state.t_hours, observed, percept["forecast_summary"], predict_fn
        )

        return Decision(
            t_hours=state.t_hours,
            proposed=proposed,
            applied=applied,
            risk_score=percept["risk_score"],
            risk_class=percept["risk_class"],
            regime=percept["regime"],
            regime_name=percept["regime_name"],
            anomaly=bool(percept["anomaly"]),
            shap_top=percept["shap_top"],
            guardrail_events=events,
            rationale=rationale,
            planner=self.controller,
        )

    def _propose(self, observed: Dict[str, float], percept: Dict[str, Any]):
        if self.controller == "do_nothing":
            return Action.no_op(), "baseline: never intervenes"
        if self.controller == "static_threshold":
            return _static_threshold(observed, self.agent_cfg)
        if self.controller == "oracle":
            return _oracle(self.stream.twin, self.agent_cfg), "oracle: reads twin ground truth"
        ctx = PlanContext(
            t_hours=observed.get("t_hours", 0.0),
            observed=observed,
            forecast=percept["forecast"],
            forecast_summary=percept["forecast_summary"],
            risk_class=percept["risk_class"],
            risk_score=percept["risk_score"],
            regime_name=percept["regime_name"],
            anomaly=bool(percept["anomaly"]),
            shap_top=percept["shap_top"],
            model=self.model,
        )
        return self.planner.propose(ctx)

    # ---------------------------------------------------------------------- helpers
    def _observed_dict(self, obs: Observation, state: State) -> Dict[str, float]:
        """What the controller sees. NH3 is derived from the *observed* TAN, pH and temp.

        Missing channels fall back to the last good reading, which is what an operator with a
        flaky probe actually has - never to the twin's truth.
        """
        from ..data.harmonize import derive_nh3

        out: Dict[str, float] = {}
        history = self.stream.history
        for c, v in obs.values.items():
            if np.isfinite(v):
                out[c] = float(v)
            else:
                prev = next(
                    (h.values[c] for h in reversed(history[:-1])
                     if c in h.values and np.isfinite(h.values[c])),
                    np.nan,
                )
                out[c] = float(prev)
        if all(np.isfinite(out.get(k, np.nan)) for k in ("tan", "ph", "temp_c")):
            out["nh3"] = float(derive_nh3(
                np.array([out["tan"]]), np.array([out["ph"]]), np.array([out["temp_c"]])
            )[0])
        else:
            out["nh3"] = np.nan
        out["t_hours"] = state.t_hours
        return out

    def _is_unsafe(self, state: State, decision: Decision) -> bool:
        """An applied action counts as unsafe only if the guardrails let a *known* hazard through.

        A pond that is already critical when the agent arrives is not an unsafe action; an
        applied action that violates a hard rule is. This is the metric that must stay 0.
        """
        a = decision.applied
        c = self.agent_cfg["guardrails"]
        if a.feed == "normal" and np.isfinite(state.do) and state.do < float(c["no_feed_when"]["do_below"]):
            return True
        if a.lime == "dose" and np.isfinite(state.ph) and state.ph > float(c["block_lime_above_ph"]):
            return True
        if np.isfinite(state.do) and state.do < float(c["emergency_do_threshold"]) and a.aerator < 1.0:
            return True
        return False


# ======================================================================================
# Baseline controllers
# ======================================================================================
def _static_threshold(observed: Dict[str, float], agent_cfg) -> tuple:
    """The industry-standard baseline: a single DO setpoint on the aerator, nothing else.

    No forecast, no ammonia awareness, no feed management. This is what the agent has to beat.
    """
    do = observed.get("do", np.nan)
    if np.isfinite(do) and do < 4.0:
        return Action(aerator=1.0), f"static threshold: DO {do:.2f} < 4.0 -> aerator on"
    return Action(), f"static threshold: DO {do:.2f} >= 4.0 -> aerator off"


def _oracle(twin: PondTwin, agent_cfg) -> Action:
    """Reference only. It reads the twin's *latent* state, which no real operator can.

    This is a privileged-information HEURISTIC, not an optimum. It is a hand-written rule that
    happens to see everything, so it is not a ceiling: it has no supersaturation term, and the
    evaluation duly shows the predictive agent beating it on DO in `ph_swing`. Never deployable.
    """
    s = twin.s
    a = Action()
    night_load = s["biomass"] + 0.55 * s["algae"] + 0.5 * s["bod_pool"]
    if s["do"] < 6.0 or night_load > 5.0:
        a = ACT.merge(a, aerator=1.0)
    if s["tan"] > 0.8:
        a = ACT.merge(a, feed="skip", exchange="small")
    if s["alkalinity"] < 30.0:
        a = ACT.merge(a, lime="dose")
    if s["no2"] > 0.5:
        a = ACT.merge(a, salt="dose")
    return a


# ======================================================================================
def _states_frame(states: List[State]) -> pd.DataFrame:
    if not states:
        return pd.DataFrame(columns=["t_hours", "do", "temp_c", "ph", "tan", "nh3", "no2", "turbidity"])
    return pd.DataFrame(
        [
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
            for s in states
        ]
    )


def _nh3_forecast_summary(fc: Forecast) -> Dict[str, float]:
    """Forecast NH3 from forecast TAN/pH/temp - the toxic form the risk bands watch."""
    from ..data.harmonize import derive_nh3

    if not {"tan", "ph", "temp_c"}.issubset(set(fc.channels)):
        return {}
    nh3 = derive_nh3(fc.channel("tan"), fc.channel("ph"), fc.channel("temp_c"))
    return {
        "fc_nh3_min": float(np.nanmin(nh3)),
        "fc_nh3_max": float(np.nanmax(nh3)),
        "fc_nh3_mean": float(np.nanmean(nh3)),
        "fc_nh3_end": float(nh3[-1]),
    }
