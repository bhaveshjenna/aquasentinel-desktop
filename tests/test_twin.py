"""Twin sanity: the physics has to move in the right direction before anything else matters."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.runtime import load_config, set_seed
from src.common.schema import Action
from src.twin import dynamics as D
from src.twin.pond import PondTwin
from src.twin.scenarios import get_scenario, load_scenarios


@pytest.fixture(autouse=True)
def _seeded():
    set_seed(20240101)


def _quiet_cfg():
    """Process noise off, so a directional assertion is about the dynamics, not the RNG."""
    cfg = load_config("twin")
    cfg = {**cfg, "process_noise": False}
    return cfg


def test_do_saturation_falls_with_temperature():
    assert D.do_saturation(10.0) > D.do_saturation(25.0) > D.do_saturation(35.0)
    assert 7.0 < D.do_saturation(28.0) < 8.5     # standard table value is ~7.8 mg/L


def test_aeration_strips_oxygen_from_a_supersaturated_pond():
    """The aerator drives DO toward saturation from BOTH sides - it is not an oxygen tap."""
    cfg = _quiet_cfg()
    base = PondTwin(cfg=cfg, start_hour_of_day=15.0)
    aerated = PondTwin(cfg=cfg, start_hour_of_day=15.0)
    for t in (base, aerated):
        t.s["do"] = 18.0          # a blooming afternoon pond, well above saturation
    for _ in range(24):
        base.step(Action())
        aerated.step(Action(aerator=1.0))
    assert aerated.s["do"] < base.s["do"]


def test_aeration_raises_do():
    base = PondTwin(cfg=_quiet_cfg(), start_hour_of_day=22.0)   # night: no photosynthesis
    aerated = PondTwin(cfg=_quiet_cfg(), start_hour_of_day=22.0)
    for _ in range(72):                                          # 6 h
        base.step(Action())
        aerated.step(Action(aerator=1.0))
    assert aerated.s["do"] > base.s["do"] + 0.5


def test_exchange_lowers_ammonia():
    cfg = _quiet_cfg()
    base = PondTwin(cfg=cfg)
    base.s["tan"] = 2.0
    swapped = PondTwin(cfg=cfg)
    swapped.s["tan"] = 2.0
    for i in range(24):
        base.step(Action())
        swapped.step(Action(exchange="large") if i == 0 else Action())
    assert swapped.s["tan"] < base.s["tan"]


def test_lime_raises_alkalinity_and_ph():
    cfg = _quiet_cfg()
    base = PondTwin(cfg=cfg)
    limed = PondTwin(cfg=cfg)
    for i in range(48):
        base.step(Action())
        limed.step(Action(lime="dose") if i == 0 else Action())
    assert limed.s["alkalinity"] > base.s["alkalinity"]


def test_feeding_raises_ammonia_and_skipping_does_not():
    cfg = _quiet_cfg()
    fed = PondTwin(cfg=cfg, start_hour_of_day=7.0)
    starved = PondTwin(cfg=cfg, start_hour_of_day=7.0)
    for _ in range(int(6 / (5 / 60))):
        fed.step(Action(feed="normal"))
        starved.step(Action(feed="skip"))
    assert fed.s["tan"] > starved.s["tan"]


def test_photosynthesis_is_zero_at_night():
    cfg = load_config("twin")
    assert D.light_fraction(2.0, cfg["light"]) == 0.0
    assert D.light_fraction(23.0, cfg["light"]) == 0.0
    assert D.light_fraction(12.0, cfg["light"]) > 0.5


def test_nh3_fraction_rises_with_ph_and_temperature():
    assert D.nh3_fraction(9.0, 28.0) > D.nh3_fraction(7.0, 28.0)
    assert D.nh3_fraction(8.0, 32.0) > D.nh3_fraction(8.0, 22.0)
    # Sanity against the standard tables: ~0.7 % un-ionised at pH 7.0, 25 C.
    assert 0.001 < D.nh3_fraction(7.0, 25.0) < 0.02


def test_determinism_same_seed_same_trajectory():
    a = PondTwin(seed=99).run(12.0)
    b = PondTwin(seed=99).run(12.0)
    assert np.allclose([s.do for s in a], [s.do for s in b])


def test_different_seeds_differ():
    a = PondTwin(seed=1).run(12.0)
    b = PondTwin(seed=2).run(12.0)
    assert not np.allclose([s.do for s in a], [s.do for s in b])


def test_copy_is_independent_but_shares_the_noise_stream():
    """The counterfactual depends on this: a fork must diverge only because of actions."""
    twin = PondTwin(seed=7)
    twin.run(6.0)
    fork = twin.copy()
    live = [twin.step(Action()).do for _ in range(48)]
    shadow = [fork.step(Action()).do for _ in range(48)]
    assert np.allclose(live, shadow), "identical actions must give identical trajectories"

    # Aerate at NIGHT. In the afternoon a blooming pond is supersaturated and the aerator
    # correctly strips oxygen toward equilibrium - a real property of the model, not a bug,
    # but not the direction this test is about.
    twin2 = PondTwin(seed=7, start_hour_of_day=21.0)
    twin2.run(3.0)
    fork2 = twin2.copy()
    acted = [twin2.step(Action(aerator=1.0)).do for _ in range(48)]
    held = [fork2.step(Action()).do for _ in range(48)]
    assert acted[-1] > held[-1], "the action must be the only thing that separates them"


def test_state_stays_physical_in_every_scenario():
    for sid in load_scenarios():
        scenario = get_scenario(sid).reset()
        twin = PondTwin(scenario=scenario, seed=scenario.seed,
                        start_hour_of_day=scenario.start_hour_of_day)
        for state in twin.run(min(scenario.duration_h, 36.0)):
            assert 0.0 <= state.do < 30.0, sid
            assert 3.0 <= state.ph <= 12.0, sid
            assert state.tan >= 0.0 and state.no2 >= 0.0, sid
            assert state.turbidity >= 0.0, sid
            assert np.isfinite(state.nh3), sid


def test_scenarios_produce_their_advertised_stress():
    """Each scenario must actually do what its name says, or the evaluation is theatre."""
    def run(sid):
        s = get_scenario(sid).reset()
        t = PondTwin(scenario=s, seed=s.seed, start_hour_of_day=s.start_hour_of_day)
        return t.run(s.duration_h)

    healthy = run("healthy_baseline")
    assert min(s.do for s in healthy) > 4.0, "the healthy baseline must not need rescuing"

    crash = run("night_do_crash")
    assert min(s.do for s in crash) < 2.0, "the DO crash must actually cross critical"

    ammonia = run("overfeeding_ammonia")
    assert max(s.nh3 for s in ammonia) > 0.05, "overfeeding must push NH3 past critical"

    bloom = run("algal_bloom")
    assert max(s.do for s in bloom) > 15.0, "a bloom must supersaturate by day"
    assert min(s.do for s in bloom) < 4.0, "and the die-off must take DO down"

    heat = run("heatwave")
    assert max(s.temp_c for s in heat) > 33.0

    swing = run("ph_swing")
    ph = [s.ph for s in swing]
    assert max(ph) - min(ph) > 1.5, "a buffer crash must make the pH swing volatile"


def test_aerator_fault_delivers_nothing():
    scenario = get_scenario("aerator_failure").reset()
    twin = PondTwin(scenario=scenario, seed=scenario.seed,
                    start_hour_of_day=scenario.start_hour_of_day)
    delivered = []
    for _ in range(int(20 / twin.dt_h)):
        twin.step(Action(aerator=1.0))
        delivered.append((twin.t_hours, twin.delivered_aerator))
    during = [d for t, d in delivered if 9.0 < t < 25.0]
    outside = [d for t, d in delivered if t < 7.0]
    assert max(during) == 0.0, "the fault must swallow the command"
    assert min(outside) == 1.0, "and only while it is active"
