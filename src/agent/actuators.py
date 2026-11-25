"""The action space, its cost model, and the candidate set the planners search over.

The twin already knows how to *apply* an ``Action`` (``PondTwin._apply_discrete``). This
module owns the other half: which actions are worth considering, and what each one costs the
farm. Keeping cost here rather than in the planner means ``reactive`` and ``predictive`` are
scored on the same currency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..common.runtime import load_config
from ..common.schema import Action


def candidates(cfg: Optional[Dict[str, Any]] = None) -> List[Action]:
    """A curated candidate set, not the full cross product.

    The full space is 3x3x3x2x2 = 108 actions; rolling a forecast forward for each on every
    tick would be both slow and pointless, since most combinations are incoherent (liming and
    exchanging at once) or dominated. These 14 cover every lever and every escalation level,
    which keeps the learned-MPC planner explainable: you can read the candidate list and know
    exactly what the agent was allowed to consider.
    """
    cfg = cfg or load_config("agent")
    levels = cfg["action_space"]["aerator"]
    lo, mid, hi = (levels + [1.0, 1.0, 1.0])[:3]

    return [
        Action(),                                              # no_op
        Action(aerator=mid),
        Action(aerator=hi),
        Action(aerator=hi, feed="skip"),
        Action(feed="reduced"),
        Action(feed="skip"),
        Action(exchange="small"),
        Action(exchange="small", feed="reduced"),
        Action(exchange="large", feed="skip"),
        Action(exchange="small", aerator=mid),
        Action(lime="dose"),
        Action(lime="dose", aerator=mid),
        Action(salt="dose"),
        Action(salt="dose", exchange="small"),
    ]


def action_cost(action: Action, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Relative operating cost of one agent tick's action, in arbitrary but consistent units."""
    cfg = cfg or load_config("agent")
    c = cfg["action_costs"]
    total = float(c["aerator_per_unit"]) * float(action.aerator)
    if action.exchange == "small":
        total += float(c["exchange_small"])
    elif action.exchange == "large":
        total += float(c["exchange_large"])
    if action.lime == "dose":
        total += float(c["lime_dose"])
    if action.salt == "dose":
        total += float(c["salt_dose"])
    return total


def feed_penalty(action: Action, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Withholding feed costs growth. Without this the agent starves the fish to stay safe."""
    cfg = cfg or load_config("agent")
    reduced = float(cfg["action_costs"].get("feed_reduced_penalty", 0.4))
    return {"normal": 0.0, "reduced": reduced, "skip": 1.0}[action.feed]


def describe(action: Action) -> str:
    """Operator-readable phrasing for the intervention log."""
    if action.is_no_op():
        return "hold - no intervention"
    bits = []
    if action.aerator > 0:
        pct = int(round(action.aerator * 100))
        bits.append(f"run aerator at {pct}%")
    if action.exchange != "none":
        bits.append(f"{action.exchange} water exchange")
    if action.feed == "reduced":
        bits.append("reduce feed")
    elif action.feed == "skip":
        bits.append("skip feeding")
    if action.lime == "dose":
        bits.append("dose lime")
    if action.salt == "dose":
        bits.append("dose salt")
    return "; ".join(bits)


def merge(base: Action, **changes) -> Action:
    from dataclasses import replace

    return replace(base, **changes)
