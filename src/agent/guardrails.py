"""Hard safety constraints. Always on, for every planner.

This is the project's actual contribution: not that a model predicts water chemistry, but
that no proposal, however it was generated, reaches the pond without passing a fixed,
auditable set of checks. Every difference between proposed and applied is logged as a
``GuardrailEvent``, and ``unsafe_applied`` must stay 0 across the whole evaluation grid.

Rules, in the order they fire:
  1. dosing caps and minimum intervals   (a dose the pond has not finished absorbing)
  2. rate limits                          (no step change in aeration)
  3. contextual blocks                    (no lime into an alkaline pond, no feeding at low DO,
                                           no large exchange into worse or thermally-shocking water)
  4. predicted-critical rejection         (any action forecast to push a channel critical)
  5. emergency override                   (DO critical now or predicted -> aerator to max)

The override is last so that it wins over every restriction above it: when the pond is
drowning, nothing outranks air.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.runtime import load_config
from ..common.schema import Action, GuardrailEvent
from ..twin import risk_bands as RB


class Guardrails:
    """Stateful across a run: it remembers dosing history and the last applied action."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, twin_cfg: Optional[Dict[str, Any]] = None):
        agent_cfg = cfg or load_config("agent")
        self.cfg = agent_cfg["guardrails"]
        self.twin_cfg = twin_cfg or load_config("twin")
        self.enabled = bool(self.cfg.get("enabled", True))
        self.margin = self.cfg.get("sensor_margin", {}) or {}
        self.reset()

    def reset(self) -> None:
        self.last_action = Action.no_op()
        self.dose_times: Dict[str, List[float]] = {"lime": [], "salt": [], "exchange": []}
        self.exchange_hours_today = 0.0
        self._day = 0
        self.events_log: List[GuardrailEvent] = []

    # ------------------------------------------------------------------------------
    def check(
        self,
        proposed: Action,
        t_hours: float,
        observed: Dict[str, float],
        forecast_summary: Optional[Dict[str, float]] = None,
        predict_fn=None,
    ) -> Tuple[Action, List[GuardrailEvent]]:
        """Return (applied_action, events). ``predict_fn(action) -> summary`` is optional."""
        events: List[GuardrailEvent] = []
        if not self.enabled:
            self.last_action = proposed
            return proposed, events

        self._roll_day(t_hours)
        action = proposed
        fc = forecast_summary or {}

        action = self._apply_dosing_caps(action, t_hours, events)
        action = self._apply_rate_limits(action, events)
        action = self._apply_context_blocks(action, observed, fc, events)
        action = self._reject_predicted_critical(action, observed, predict_fn, events)
        action = self._emergency_override(action, observed, fc, events)

        self._record(action, t_hours)
        self.last_action = action
        self.events_log.extend(events)
        return action, events

    def safety_only(
        self,
        action: Action,
        t_hours: float,
        observed: Dict[str, float],
        forecast_summary: Optional[Dict[str, float]] = None,
    ) -> Tuple[Action, List[GuardrailEvent]]:
        """The subset of rules that must run at *sensor* rate, not planning rate.

        Planning is expensive, so the agent only re-plans every ``decide_every_minutes``. A
        pond can cross into critical in far less than that, so the contextual blocks and the
        emergency override run on every single tick against the action currently being held.
        This split - slow planner, fast safety layer - is the standard supervisory-control
        arrangement and it is what keeps unsafe-applied at zero between decisions.
        """
        events: List[GuardrailEvent] = []
        if not self.enabled:
            return action, events
        out = self._apply_context_blocks(action, observed, forecast_summary or {}, events)
        out = self._emergency_override(out, observed, forecast_summary or {}, events)
        # Discrete impulse treatments (lime, salt) are one-shot doses; holding between decision points
        # must never re-trigger them.
        if out.lime != "none" or out.salt != "none":
            out = replace(out, lime="none", salt="none")
        if out != action:
            self.last_action = out
            self.events_log.extend(events)
        return out, events

    # ------------------------------------------------------------------------------
    def _roll_day(self, t_hours: float) -> None:
        day = int(t_hours // 24)
        if day != self._day:
            self._day = day
            self.exchange_hours_today = 0.0
            cutoff = day * 24.0
            for k in self.dose_times:
                self.dose_times[k] = [t for t in self.dose_times[k] if t >= cutoff]

    def _apply_dosing_caps(self, a: Action, t: float, ev: List[GuardrailEvent]) -> Action:
        c = self.cfg
        intervals = c["min_interval_minutes"]

        if a.lime == "dose":
            if len(self.dose_times["lime"]) >= int(c["max_lime_doses_per_day"]):
                ev.append(GuardrailEvent("lime_daily_cap",
                                         f"already {len(self.dose_times['lime'])} lime doses today",
                                         "block"))
                a = replace(a, lime="none")
            elif self._too_soon("lime", t, intervals["lime"]):
                ev.append(GuardrailEvent("lime_min_interval",
                                         f"last lime dose < {intervals['lime']} min ago", "block"))
                a = replace(a, lime="none")

        if a.salt == "dose":
            if len(self.dose_times["salt"]) >= int(c["max_salt_doses_per_day"]):
                ev.append(GuardrailEvent("salt_daily_cap", "salt dose cap reached today", "block"))
                a = replace(a, salt="none")
            elif self._too_soon("salt", t, intervals["salt"]):
                ev.append(GuardrailEvent("salt_min_interval",
                                         f"last salt dose < {intervals['salt']} min ago", "block"))
                a = replace(a, salt="none")

        if a.exchange != "none":
            if self._too_soon("exchange", t, intervals["exchange"]):
                ev.append(GuardrailEvent("exchange_min_interval",
                                         "exchange still within its minimum interval", "block"))
                a = replace(a, exchange="none")
            elif self.exchange_hours_today >= float(c["max_exchange_hours_per_day"]):
                ev.append(GuardrailEvent("exchange_daily_cap",
                                         f"{self.exchange_hours_today:.1f} h exchanged today", "block"))
                a = replace(a, exchange="none")
        return a

    def _apply_rate_limits(self, a: Action, ev: List[GuardrailEvent]) -> Action:
        limit = float(self.cfg["aerator_rate_limit_per_tick"])
        delta = a.aerator - self.last_action.aerator
        if abs(delta) > limit + 1e-9:
            capped = self.last_action.aerator + np.sign(delta) * limit
            ev.append(GuardrailEvent("aerator_rate_limit",
                                     f"aerator {a.aerator:.2f} -> {capped:.2f} "
                                     f"(max step {limit})", "clamp"))
            a = replace(a, aerator=float(np.clip(capped, 0.0, 1.0)))
        return a

    def _apply_context_blocks(
        self, a: Action, obs: Dict[str, float], fc: Dict[str, float], ev: List[GuardrailEvent]
    ) -> Action:
        c = self.cfg
        ph = self._pessimistic(obs, "ph", low=False)
        if a.lime == "dose" and np.isfinite(ph) and ph > float(c["block_lime_above_ph"]):
            ev.append(GuardrailEvent("lime_into_alkaline_pond",
                                     f"pH {ph:.2f} > {c['block_lime_above_ph']}", "block"))
            a = replace(a, lime="none")

        nf = c["no_feed_when"]
        do = self._pessimistic(obs, "do", low=True)
        nh3 = self._pessimistic(obs, "nh3", low=False)
        fc_do = fc.get("fc_do_min", np.nan)
        if a.feed == "normal":
            reasons = []
            if np.isfinite(do) and do < float(nf["do_below"]):
                reasons.append(f"DO {do:.2f} < {nf['do_below']}")
            if np.isfinite(nh3) and nh3 > float(nf["nh3_above"]):
                reasons.append(f"NH3 {nh3:.3f} > {nf['nh3_above']}")
            if np.isfinite(fc_do) and fc_do < float(nf["forecast_do_below"]):
                reasons.append(f"forecast DO {fc_do:.2f} < {nf['forecast_do_below']}")
            if reasons:
                ev.append(GuardrailEvent("feeding_blocked", "; ".join(reasons), "clamp"))
                a = replace(a, feed="skip")

        if a.exchange == "large":
            inflow = c["inflow_quality"]
            temp = obs.get("temp_c", np.nan)
            delta = abs(temp - float(inflow["temp_c"])) if np.isfinite(temp) else 0.0
            if delta > float(c["block_large_exchange_above_temp_delta"]):
                ev.append(GuardrailEvent("large_exchange_thermal_shock",
                                         f"|pond {temp:.1f} - inflow {inflow['temp_c']:.1f}| "
                                         f"= {delta:.1f} C", "clamp"))
                a = replace(a, exchange="small")
            elif c.get("require_better_inflow_for_large_exchange", True):
                if not self._inflow_is_better(obs, inflow):
                    ev.append(GuardrailEvent("large_exchange_inflow_not_better",
                                             "replacement water is not known to be better", "clamp"))
                    a = replace(a, exchange="small")
        return a

    def _reject_predicted_critical(
        self, a: Action, obs: Dict[str, float], predict_fn, ev: List[GuardrailEvent]
    ) -> Action:
        """Roll the proposal forward; if it drives a channel critical, fall back to no_op.

        Deliberately conservative: an action that would *cause* a critical excursion is never
        preferable to doing nothing, and the emergency override below still gets the last word.
        """
        if predict_fn is None or a.is_no_op():
            return a
        try:
            summary = predict_fn(a)
        except Exception:
            return a
        if summary is None:
            return a
        level = RB.worst_forecast_risk(summary, self.twin_cfg)
        if level != "critical":
            return a
        base = predict_fn(Action.no_op())
        if base is not None and RB.worst_forecast_risk(base, self.twin_cfg) == "critical":
            return a  # the pond is already heading critical; the action is not the cause
        ev.append(GuardrailEvent("predicted_critical",
                                 f"'{a.label()}' predicted to drive a channel critical", "block"))
        return Action.no_op()

    def _emergency_override(
        self, a: Action, obs: Dict[str, float], fc: Dict[str, float], ev: List[GuardrailEvent]
    ) -> Action:
        c = self.cfg
        do = self._pessimistic(obs, "do", low=True)
        fc_do = fc.get("fc_do_min", np.nan)
        trigger = None
        if np.isfinite(do) and do < float(c["emergency_do_threshold"]):
            trigger = f"DO {do:.2f} < {c['emergency_do_threshold']}"
        elif np.isfinite(fc_do) and fc_do < float(c["emergency_forecast_do"]):
            trigger = f"forecast min DO {fc_do:.2f} < {c['emergency_forecast_do']}"
        if trigger is None:
            return a
        # The rate limit is deliberately bypassed here: an emergency is exactly the case the
        # limit exists to be overruled by.
        if a.aerator < 1.0 or a.feed == "normal":
            ev.append(GuardrailEvent("emergency_do_override",
                                     f"{trigger} -> aerator max, feeding stopped", "override"))
        return replace(a, aerator=1.0, feed="skip")

    # ------------------------------------------------------------------------------
    def _pessimistic(self, obs: Dict[str, float], channel: str, low: bool) -> float:
        """The reading, moved by the sensor margin in whichever direction is less safe."""
        v = obs.get(channel, np.nan)
        m = float(self.margin.get(channel, 0.0))
        if not np.isfinite(v) or m == 0.0:
            return v
        return v - m if low else v + m

    def _too_soon(self, kind: str, t: float, min_minutes: float) -> bool:
        times = self.dose_times[kind]
        return bool(times) and (t - times[-1]) * 60.0 < float(min_minutes)

    def _inflow_is_better(self, obs: Dict[str, float], inflow: Dict[str, float]) -> bool:
        """'Known better' = the operator's stated inflow beats the pond on DO and ammonia."""
        do_ok = obs.get("do", np.inf) < float(inflow["do"]) - 0.5
        tan_ok = obs.get("tan", 0.0) > float(inflow["tan"]) + 0.1
        return bool(do_ok or tan_ok)

    def _record(self, a: Action, t: float) -> None:
        if a.lime == "dose":
            self.dose_times["lime"].append(t)
        if a.salt == "dose":
            self.dose_times["salt"].append(t)
        if a.exchange != "none":
            self.dose_times["exchange"].append(t)
            self.exchange_hours_today += float(self.twin_cfg["actions"]["exchange_duration_h"])


def is_unsafe(observed: Dict[str, float], twin_cfg=None) -> bool:
    """Did the pond actually reach a critical band? Used for the unsafe-applied counter."""
    level, _ = RB.state_risk(observed, twin_cfg)
    return level == "critical"
