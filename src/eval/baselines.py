"""The controllers the agent is measured against.

    do_nothing        no intervention, ever. Guardrails are OFF for this one - it is a
                      reference trajectory, not a controller, and letting the safety layer
                      rescue it would destroy the comparison it exists to provide.
    static_threshold  the industry-standard aerator thermostat: DO below 4 mg/L -> aerator on.
                      No forecast, no ammonia awareness, no feed management. This is the bar.
    reactive          the project's own transparent rule set, with the forecast available.
    predictive        the learned-MPC agent. The thing being evaluated.
    oracle            reads the twin's LATENT state (algae, biomass, BOD pool) - information
                      no probe can provide. It is a privileged-information HEURISTIC, not an
                      optimum: it is a hand-written rule that happens to see everything, and
                      the evaluation shows the agent beating it on DO in `ph_swing`, where its
                      rule never addresses supersaturation. Read it as "what a good rule could
                      do with perfect knowledge", not as a ceiling. Never deployable.

All five are driven by the same ``AgentRun``, so any difference is the policy and not the
harness. The implementations live in ``agent/loop.py``; this module is the registry, the
descriptions, and the honest labelling of what each one is allowed to see.
"""

from __future__ import annotations

from typing import Dict, List

CONTROLLERS: Dict[str, Dict[str, str]] = {
    "do_nothing": {
        "label": "Do nothing",
        "kind": "baseline",
        "sees": "nothing",
        "note": "Reference trajectory. Guardrails disabled by design.",
    },
    "static_threshold": {
        "label": "Static DO threshold",
        "kind": "baseline",
        "sees": "current DO",
        "note": "The conventional aerator thermostat. The bar the agent must clear.",
    },
    "reactive": {
        "label": "Agent (reactive rules)",
        "kind": "agent",
        "sees": "current channels + forecast summary",
        "note": "Transparent utility rules over every channel, not just DO.",
    },
    "predictive": {
        "label": "Agent (predictive, learned MPC)",
        "kind": "agent",
        "sees": "current channels + forecast + internal pond model",
        "note": "Scores candidate actions by rolled-forward consequences. The contribution.",
    },
    "oracle": {
        "label": "Oracle (latent state)",
        "kind": "reference",
        "sees": "twin latent state - unavailable to any real farm",
        "note": ("Privileged-information heuristic, NOT an optimum or an upper bound - it is a "
                 "hand-written rule with perfect knowledge, and the agent beats it on DO in "
                 "ph_swing. Not deployable; never claimed as a result."),
    },
}

DEFAULT_GRID: List[str] = ["do_nothing", "static_threshold", "reactive", "predictive", "oracle"]
COMPARISON_PAIRS = [("predictive", "static_threshold"), ("predictive", "do_nothing"),
                    ("predictive", "reactive")]


def describe(name: str) -> str:
    c = CONTROLLERS.get(name, {})
    return f"{c.get('label', name)} - sees {c.get('sees', '?')}. {c.get('note', '')}"


def is_deployable(name: str) -> bool:
    """Whether a controller could exist on a real farm. The oracle cannot."""
    return CONTROLLERS.get(name, {}).get("kind") != "reference"
