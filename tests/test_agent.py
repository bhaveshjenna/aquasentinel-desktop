"""Agent loop, planners, counterfactual integrity and evaluation reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from src.agent import actuators as ACT
from src.agent.loop import AgentRun
from src.agent.policy import InternalModel, PlanContext, PredictivePlanner, ReactivePlanner
from src.common.runtime import load_config, set_seed
from src.common.schema import Action
from src.eval import metrics as M


@pytest.fixture(autouse=True)
def _seeded():
    set_seed(20240101)


# ---------------------------------------------------------------------------- actuators
def test_candidates_cover_every_lever():
    cands = ACT.candidates()
    assert any(c.aerator == 1.0 for c in cands)
    assert any(c.exchange == "large" for c in cands)
    assert any(c.lime == "dose" for c in cands)
    assert any(c.salt == "dose" for c in cands)
    assert any(c.feed == "skip" for c in cands)
    assert any(c.is_no_op() for c in cands), "holding must always be an option"


def test_doing_nothing_is_free_and_acting_is_not():
    assert ACT.action_cost(Action()) == 0.0
    assert ACT.action_cost(Action(aerator=1.0)) > 0
    assert ACT.action_cost(Action(exchange="large")) > ACT.action_cost(Action(exchange="small"))


def test_withholding_feed_is_penalised():
    assert ACT.feed_penalty(Action(feed="normal")) == 0.0
    assert ACT.feed_penalty(Action(feed="skip")) > ACT.feed_penalty(Action(feed="reduced")) > 0


# ---------------------------------------------------------------------------- planners
def _ctx(**observed):
    base = {"do": 7.0, "temp_c": 28.0, "ph": 7.6, "tan": 0.3, "nh3": 0.005, "no2": 0.05}
    return PlanContext(t_hours=0.0, observed={**base, **observed})


def test_reactive_holds_when_everything_is_fine():
    action, why = ReactivePlanner().propose(_ctx())
    assert action.is_no_op()
    assert "within thresholds" in why


def test_reactive_aerates_on_low_do():
    action, why = ReactivePlanner().propose(_ctx(do=3.5))
    assert action.aerator == 1.0 and action.feed == "skip"
    assert "DO" in why


def test_reactive_responds_to_ammonia_and_nitrite():
    assert ReactivePlanner().propose(_ctx(nh3=0.04))[0].exchange == "small"
    assert ReactivePlanner().propose(_ctx(no2=1.2))[0].salt == "dose"


def test_reactive_uses_the_forecast_not_only_the_present():
    ctx = _ctx(do=8.0)
    ctx.forecast_summary = {"fc_do_min": 3.0}
    action, _ = ReactivePlanner().propose(ctx)
    assert action.aerator > 0, "a comfortable now with a bad forecast must still act"


def test_predictive_prefers_acting_when_the_pond_is_heading_down():
    model = InternalModel(seed=1, start_hour_of_day=20.0)
    model.twin.s.update({"do": 3.4, "algae": 4.0, "biomass": 3.5, "bod_pool": 12.0})
    ctx = _ctx(do=3.4)
    ctx.model = model
    action, why = PredictivePlanner().propose(ctx)
    assert action.aerator > 0
    assert "candidates" in why and "min DO" in why


def test_predictive_holds_in_a_healthy_pond():
    model = InternalModel(seed=1, start_hour_of_day=11.0)
    ctx = _ctx()
    ctx.model = model
    action, _ = PredictivePlanner().propose(ctx)
    assert ACT.action_cost(action) <= ACT.action_cost(Action(aerator=0.5)), \
        "a healthy pond must not provoke expensive intervention"


def test_internal_model_never_reads_the_scenario_schedule():
    """The agent's belief must not know what the operator is about to do to the pond."""
    model = InternalModel(seed=1)
    traj = model.rollout(Action(), steps=12, hold_steps=0)
    assert model.twin.scenario is None
    assert np.isfinite(traj["do"]).all()


# ---------------------------------------------------------------------------- the loop
def test_run_produces_one_decision_per_tick():
    run = AgentRun("healthy_baseline", controller="reactive", seed=1, with_shadow=False)
    result = run.run()
    assert len(result.decisions) == len(result.states) == run.n_ticks


def test_shadow_pond_is_a_true_counterfactual():
    """Same seed, same perturbations, same noise: only the actions differ."""
    result = AgentRun("night_do_crash", controller="predictive", seed=3).run()
    assert len(result.shadow_states) == len(result.states)
    live_min = min(s.do for s in result.states)
    shadow_min = min(s.do for s in result.shadow_states)
    assert live_min > shadow_min, "the agent must actually change the outcome"


def test_do_nothing_matches_its_own_shadow():
    """The strongest check on the counterfactual: with no actions the two ponds are identical."""
    result = AgentRun("night_do_crash", controller="do_nothing", seed=11).run()
    live = [s.do for s in result.states]
    shadow = [s.do for s in result.shadow_states]
    assert np.allclose(live, shadow), "an unacted pond must equal its no-action shadow exactly"


def test_the_agent_never_reads_twin_ground_truth():
    """It is fed the noisy probe, so its view must differ from the truth."""
    run = AgentRun("healthy_baseline", controller="reactive", seed=5, with_shadow=False)
    for _ in range(30):
        run.tick()
    truth = run.result.states[-1].do
    observed = run.result.observations[-1].values["do"]
    assert observed != pytest.approx(truth, abs=1e-9)


def test_missing_readings_fall_back_to_history_not_to_truth():
    run = AgentRun("sensor_fault", controller="reactive", seed=7, with_shadow=False).run()
    for d in run.decisions:
        assert np.isfinite(d.risk_score)


def test_runs_are_reproducible():
    a = AgentRun("night_do_crash", controller="predictive", seed=42, with_shadow=False).run()
    b = AgentRun("night_do_crash", controller="predictive", seed=42, with_shadow=False).run()
    assert a.metrics() == b.metrics()


def test_different_controllers_give_different_outcomes():
    nothing = AgentRun("night_do_crash", controller="do_nothing", seed=8, with_shadow=False).run()
    agent = AgentRun("night_do_crash", controller="predictive", seed=8, with_shadow=False).run()
    assert agent.metrics()["time_critical_h"] < nothing.metrics()["time_critical_h"]


def test_metrics_are_complete_and_finite():
    m = AgentRun("healthy_baseline", controller="predictive", seed=9).run().metrics()
    for key in ("time_in_risk_h", "time_critical_h", "interventions", "unsafe_applied",
                "intervention_cost", "min_do", "risk_h_do"):
        assert key in m
        assert np.isfinite(float(m[key])) if isinstance(m[key], (int, float)) else True


# ---------------------------------------------------------------------------- statistics
def test_wilcoxon_refuses_to_test_too_few_seeds():
    res = M.wilcoxon([1, 2, 3], [2, 3, 4])
    assert np.isnan(res["p"])
    assert "too few seeds" in res["note"]


def test_wilcoxon_detects_a_consistent_difference():
    a = list(range(10))
    b = [x + 5 for x in range(10)]
    res = M.wilcoxon(a, b)
    assert res["p"] < 0.05
    assert res["median_diff"] == pytest.approx(-5.0)
    assert res["n"] == 10


def test_wilcoxon_handles_identical_inputs():
    res = M.wilcoxon([1.0] * 8, [1.0] * 8)
    assert res["p"] == 1.0 and "exactly zero" in res["note"]


def test_anomaly_recall_is_scored_per_event():
    times = list(np.arange(0, 20, 0.5))
    flags = [10.0 <= t <= 10.5 for t in times]
    res = M.anomaly_recall(flags, [(10.0, 12.0), (16.0, 18.0)], times)
    assert res["events"] == 2 and res["detected"] == 1
    assert res["recall"] == pytest.approx(0.5)


def test_anomaly_false_positive_rate_excludes_the_fault_windows():
    times = list(np.arange(0, 20, 0.5))
    flags = [t < 2.0 for t in times]
    res = M.anomaly_recall(flags, [(10.0, 12.0)], times)
    assert res["detected"] == 0 and res["false_positive_rate"] > 0


# ---------------------------------------------------------------------------- eval harness
def test_run_returns_a_result_not_the_runner():
    """`AgentRun.run()` returns the RunResult. Confusing the two shipped a crash into the
    evaluation harness that only surfaced 90 minutes into a grid, after every expensive stage
    had already completed - so it is worth one cheap assertion."""
    from src.agent.loop import RunResult

    result = AgentRun("healthy_baseline", controller="reactive", seed=1,
                      with_shadow=False).run()
    assert isinstance(result, RunResult)
    assert not hasattr(result, "result")
    assert result.decisions and result.states


def test_anomaly_eval_runs_end_to_end():
    """Smoke test for the stage that crashed: it must return one scored row per seed."""
    from src.agent.loop import Perception
    from src.eval.run_eval import run_anomaly_eval
    from src.perception.regimes import RegimeModel
    from src.common.runtime import MODELS_DIR

    path = MODELS_DIR / "regimes.pkl"
    if not path.exists():
        pytest.skip("no pretrained regime model; run scripts/pretrain_federated.py")

    perception = Perception(regimes=RegimeModel.load(path))
    out = run_anomaly_eval([11, 12], perception, verbose=False)
    assert len(out) == 2
    for col in ("events", "detected", "recall", "false_positive_rate"):
        assert col in out.columns
    assert (out["recall"] >= 0).all() and (out["recall"] <= 1).all()


def test_impulse_actions_not_repeated_across_cadence():
    """Verify that impulse actions (lime, salt) are executed only once across a multi-tick cadence."""
    run = AgentRun("healthy_baseline", controller="reactive", seed=42, with_shadow=False)
    # Force first decision to be a lime dose
    forced_action = Action(lime="dose", aerator=0.5)

    # Tick 0: initial tick + plan
    run.tick()
    run.current_action = forced_action

    initial_lime_pool = run.stream.twin._lime_pool

    # Advance through the 3 ticks of the decision interval (15 minutes)
    for _ in range(run.ticks_per_decision):
        run.tick()

    # The twin's lime pool should have received exactly one dose (25 kg), not 3 doses (75 kg)
    lime_dose_kg = float(run.twin_cfg["actions"]["lime_dose_kg"])
    # Some lime dissolves during these steps, but total added to pool initially was exactly 1 dose
    # Before dissolution: added was lime_dose_kg. Check that it didn't accumulate 3 * lime_dose_kg
    assert run.stream.twin._lime_pool < 2.0 * lime_dose_kg


