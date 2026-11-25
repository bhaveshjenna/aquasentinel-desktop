"""Guardrail units. The safety claim is only as good as these tests.

Every rule gets a case that would violate it, and the assertion is that the *applied* action
does not. The final test is the one that matters most: across every scenario and controller,
unsafe-applied stays at zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agent.guardrails import Guardrails
from src.agent.loop import AgentRun
from src.common.runtime import load_config, set_seed
from src.common.schema import Action


@pytest.fixture(autouse=True)
def _seeded():
    set_seed(20240101)


@pytest.fixture
def gr():
    return Guardrails()


SAFE = {"do": 7.0, "ph": 7.6, "temp_c": 28.0, "tan": 0.3, "nh3": 0.005, "no2": 0.05}


def obs(**over):
    return {**SAFE, **over}


# ---------------------------------------------------------------------------- emergency
def test_critical_do_forces_aerator_to_max(gr):
    applied, events = gr.check(Action(aerator=0.0), 0.0, obs(do=1.8))
    assert applied.aerator == 1.0
    assert applied.feed == "skip"
    assert any(e.rule == "emergency_do_override" for e in events)


def test_forecast_critical_do_also_triggers_the_override(gr):
    applied, events = gr.check(Action(), 0.0, obs(do=6.0), {"fc_do_min": 1.5})
    assert applied.aerator == 1.0
    assert any(e.rule == "emergency_do_override" for e in events)


def test_emergency_beats_the_rate_limit(gr):
    """A rate limit that could delay an emergency response would be a bug, not a safeguard."""
    gr.check(Action(aerator=0.0), 0.0, obs())
    applied, _ = gr.check(Action(aerator=0.0), 0.5, obs(do=1.0))
    assert applied.aerator == 1.0


# ---------------------------------------------------------------------------- dosing
def test_lime_daily_cap(gr):
    """Attempt a dose every interval for one whole day; the cap must hold within that day."""
    g = load_config("agent")["guardrails"]
    cap = int(g["max_lime_doses_per_day"])
    interval_h = g["min_interval_minutes"]["lime"] / 60.0
    applied_doses = 0
    t = 0.0
    while t < 23.5:
        applied, _ = gr.check(Action(lime="dose"), t, obs(ph=6.2))
        applied_doses += int(applied.lime == "dose")
        t += interval_h + 0.1
    assert applied_doses == cap, f"expected exactly {cap} doses in a day, got {applied_doses}"


def test_lime_cap_resets_on_the_next_day(gr):
    g = load_config("agent")["guardrails"]
    cap = int(g["max_lime_doses_per_day"])
    interval_h = g["min_interval_minutes"]["lime"] / 60.0
    for i in range(cap):
        gr.check(Action(lime="dose"), i * (interval_h + 0.1), obs(ph=6.2))
    blocked, _ = gr.check(Action(lime="dose"), 20.0, obs(ph=6.2))
    fresh, _ = gr.check(Action(lime="dose"), 30.0, obs(ph=6.2))
    assert blocked.lime == "none"
    assert fresh.lime == "dose"


def test_lime_minimum_interval(gr):
    a1, _ = gr.check(Action(lime="dose"), 0.0, obs(ph=6.2))
    a2, ev = gr.check(Action(lime="dose"), 0.5, obs(ph=6.2))
    assert a1.lime == "dose"
    assert a2.lime == "none"
    assert any(e.rule == "lime_min_interval" for e in ev)


def test_no_lime_into_an_alkaline_pond(gr):
    applied, ev = gr.check(Action(lime="dose"), 0.0, obs(ph=9.0))
    assert applied.lime == "none"
    assert any(e.rule == "lime_into_alkaline_pond" for e in ev)


def test_salt_daily_cap(gr):
    a1, _ = gr.check(Action(salt="dose"), 0.0, obs(no2=1.0))
    a2, _ = gr.check(Action(salt="dose"), 13.0, obs(no2=1.0))
    assert a1.salt == "dose"
    assert a2.salt == "none"


# ---------------------------------------------------------------------------- feeding
def test_feeding_blocked_at_low_do(gr):
    applied, ev = gr.check(Action(feed="normal"), 0.0, obs(do=3.0))
    assert applied.feed == "skip"
    assert any(e.rule == "feeding_blocked" for e in ev)


def test_feeding_blocked_at_high_ammonia(gr):
    applied, ev = gr.check(Action(feed="normal"), 0.0, obs(nh3=0.09))
    assert applied.feed == "skip"


def test_feeding_blocked_by_a_bad_forecast(gr):
    applied, _ = gr.check(Action(feed="normal"), 0.0, obs(do=6.0), {"fc_do_min": 2.5})
    assert applied.feed == "skip"


# ---------------------------------------------------------------------------- exchange
def test_large_exchange_downgraded_on_thermal_shock(gr):
    applied, ev = gr.check(Action(exchange="large"), 0.0, obs(temp_c=34.0))
    assert applied.exchange == "small"
    assert any("thermal_shock" in e.rule for e in ev)


def test_large_exchange_downgraded_when_inflow_is_not_better(gr):
    applied, ev = gr.check(Action(exchange="large"), 0.0, obs(do=8.5, tan=0.02, temp_c=26.5))
    assert applied.exchange == "small"


# ---------------------------------------------------------------------------- rate limits
def test_aerator_rate_limit_is_respected():
    cfg = load_config("agent")
    cfg = {**cfg, "guardrails": {**cfg["guardrails"], "aerator_rate_limit_per_tick": 0.25}}
    g = Guardrails(cfg)
    g.check(Action(aerator=0.0), 0.0, obs())
    applied, ev = g.check(Action(aerator=1.0), 0.25, obs())
    assert applied.aerator == pytest.approx(0.25)
    assert any(e.rule == "aerator_rate_limit" for e in ev)


# ---------------------------------------------------------------------------- predicted critical
def test_an_action_predicted_to_go_critical_is_rejected(gr):
    def predict(action):
        if action.is_no_op():
            return {"fc_do_min": 6.0}
        return {"fc_do_min": 1.0}     # this action would drown the pond

    applied, ev = gr.check(Action(exchange="large"), 0.0, obs(), {}, predict)
    assert any(e.rule == "predicted_critical" for e in ev)


def test_an_already_critical_pond_does_not_blame_the_action(gr):
    """If doing nothing is also critical, the action is not the cause and must not be blocked."""
    def predict(action):
        return {"fc_do_min": 1.0}

    applied, ev = gr.check(Action(aerator=0.5), 0.0, obs(do=6.0), {}, predict)
    assert not any(e.rule == "predicted_critical" for e in ev)


# ---------------------------------------------------------------------------- disabled
def test_disabled_guardrails_pass_everything_through():
    g = Guardrails()
    g.enabled = False
    proposal = Action(aerator=0.0, feed="normal", lime="dose")
    applied, ev = g.check(proposal, 0.0, obs(do=0.5, ph=10.0))
    assert applied == proposal and ev == []


# ---------------------------------------------------------------------------- the claim
@pytest.mark.parametrize("controller", ["static_threshold", "reactive", "predictive", "oracle"])
def test_no_unsafe_action_is_ever_applied(controller):
    """The headline safety result, as a test rather than a sentence."""
    for scenario in ("night_do_crash", "overfeeding_ammonia", "aerator_failure"):
        run = AgentRun(scenario, controller=controller, seed=4242, with_shadow=False).run()
        unsafe = sum(int(d.unsafe_applied) for d in run.decisions)
        assert unsafe == 0, f"{controller} applied {unsafe} unsafe actions in {scenario}"


def test_turning_the_guardrails_off_is_measurably_unsafe():
    """The claim "the guardrails are load-bearing" as a test rather than an assertion.

    The scenario is `aerator_failure` on purpose, and that is itself a finding. Across the
    ablation grid the predictive planner is safe *without* guardrails in most scenarios - it
    plans well enough on its own under ordinary stress. The place the safety layer actually
    earns its keep is when the actuator it chose does not deliver: the planner commands
    aeration, the fault swallows it, DO falls, and nothing in the planner's own logic reacts
    to the discrepancy. The emergency override does.
    """
    guarded = AgentRun("aerator_failure", controller="predictive", seed=4242,
                       with_shadow=False, guardrails_enabled=True).run()
    unguarded = AgentRun("aerator_failure", controller="predictive", seed=4242,
                         with_shadow=False, guardrails_enabled=False).run()

    assert guarded.metrics()["unsafe_applied"] == 0
    assert unguarded.metrics()["unsafe_applied"] > 0,         "if the agent is safe without guardrails here, the guardrails are not doing anything"


def _cfg_with(world_model: bool):
    import copy
    from src.common.runtime import load_config
    cfg = copy.deepcopy(load_config("agent"))
    cfg["predictive"]["world_model"] = world_model
    return cfg


def test_a_well_behaved_planner_needs_no_rescuing_in_ordinary_stress():
    """The other half of the same finding, so the previous test is not read as more than it is.

    Under a DO crash with a working aerator the planner driven by the **hand-written pond
    model** is already safe on its own. The guardrails are a backstop for the cases it cannot
    see, not a crutch it leans on constantly.

    This is pinned to the physics model deliberately. The learned world model does *not* have
    this property - see the test directly below, which is the finding rather than an exception.
    """
    unguarded = AgentRun("night_do_crash", controller="predictive", seed=4242,
                         with_shadow=False, guardrails_enabled=False,
                         agent_cfg=_cfg_with(world_model=False)).run()
    assert unguarded.metrics()["unsafe_applied"] == 0


def test_the_learned_world_model_is_not_safe_without_guardrails():
    """The finding that justifies the safety layer existing at all under learned dynamics.

    Swapping the hand-written pond model for the action-conditioned LSTM turns a planner that
    was safe unguarded into one that is not: it proposes actions the hard limits then have to
    catch. A learned dynamics model is only approximately right, and the planner optimises
    against it as though it were exactly right - so prediction error becomes unsafe commands.

    Both halves are asserted so neither can be overstated: unguarded it proposes unsafe
    actions, and guarded it applies none of them. That is the whole argument for keeping the
    guardrails outside the learned component rather than folding them into its cost function.
    """
    from src.common.runtime import MODELS_DIR
    if not (MODELS_DIR / "world_model_federated.pt").exists():
        pytest.skip("no world model; run scripts/pretrain_world_model.py")

    cfg = _cfg_with(world_model=True)
    unguarded = AgentRun("night_do_crash", controller="predictive", seed=4242,
                         with_shadow=False, guardrails_enabled=False, agent_cfg=cfg).run()
    guarded = AgentRun("night_do_crash", controller="predictive", seed=4242,
                       with_shadow=False, guardrails_enabled=True, agent_cfg=cfg).run()

    assert unguarded.metrics()["unsafe_applied"] > 0,         "if the learned planner were safe unguarded here, this finding would have gone away"
    assert guarded.metrics()["unsafe_applied"] == 0,         "whatever the learned model proposes, the guardrails must still apply nothing unsafe"


def test_do_nothing_is_deliberately_unguarded():
    """The reference trajectory must NOT be rescued, or it stops being a reference."""
    run = AgentRun("night_do_crash", controller="do_nothing", seed=4242, with_shadow=False).run()
    assert all(d.applied.is_no_op() for d in run.decisions)
    assert sum(int(d.unsafe_applied) for d in run.decisions) > 0
