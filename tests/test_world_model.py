"""The action-conditioned world model: shapes, action sensitivity, and the planner seam.

The claim this file defends is narrow and specific: when the learned model is in place, the
trajectory the planner scores comes from the network, the network actually responds to the
action it is given, and nothing it can output is allowed to be physically absurd.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agent import actuators as ACT
from src.agent.policy import InternalModel, LearnedWorldModel
from src.common.runtime import MODELS_DIR, set_seed
from src.common.schema import Action, Observation
from src.perception.world_model import CHANNELS, ActionWorldModel, action_features


@pytest.fixture(autouse=True)
def _seeded():
    set_seed(20240101)


def flat_window(model, do=6.0):
    return np.tile(np.array([do, 28.0, 7.6, 0.4, 0.05, 25.0], dtype=np.float32), (model.window, 1))


# ------------------------------------------------------------------ action encoding
def test_action_features_separate_every_lever():
    """If two different actions encoded identically the model could not tell them apart."""
    seen = {tuple(action_features(a, 1.0)) for a in ACT.candidates()}
    assert len(seen) == len(ACT.candidates()), "candidate actions must encode distinctly"


def test_action_features_carry_the_hold_fraction():
    """The planner holds a candidate then releases it; a model not told so over-predicts."""
    full = action_features(Action(aerator=1.0), 1.0)
    half = action_features(Action(aerator=1.0), 0.5)
    assert full[-1] == 1.0 and half[-1] == 0.5
    assert np.array_equal(full[:-1], half[:-1]), "only the hold fraction may differ"


def test_no_op_encodes_as_all_zeros_but_the_hold():
    f = action_features(Action.no_op(), 1.0)
    assert not f[:-1].any(), "no_op must not look like an intervention to the network"


# ------------------------------------------------------------------ the network
def test_predict_returns_every_plan_channel_with_the_right_shape():
    m = ActionWorldModel()
    out = m.predict(flat_window(m), Action(aerator=1.0))
    assert set(out) == set(CHANNELS)
    for c in CHANNELS:
        assert out[c].shape == (m.horizon,)
        assert np.isfinite(out[c]).all()


def test_predict_batch_matches_predict_one_at_a_time():
    """The planner uses the batched path; it must not be a different model."""
    m = ActionWorldModel()
    w = flat_window(m)
    cands = ACT.candidates()[:5]
    batch = m.predict_batch(w, cands, 0.5)
    for a, got in zip(cands, batch):
        one = m.predict(w, a, 0.5)
        for c in CHANNELS:
            assert np.allclose(got[c], one[c], atol=1e-5)


def test_the_network_actually_responds_to_the_action():
    """An untrained net still has to *depend* on the action input, or the head is dead wiring.

    This is deliberately a structural claim, not a physical one: it asserts the action reaches
    the output, which is what makes the model action-conditioned at all. Whether the response
    is correct is what training decides and what the evaluation measures.
    """
    m = ActionWorldModel()
    w = flat_window(m)
    a = m.predict(w, Action.no_op())
    b = m.predict(w, Action(aerator=1.0, exchange="large"))
    assert not np.allclose(a["do"], b["do"]), "the action must change the prediction"


def test_encoder_warm_start_from_the_federated_forecaster():
    """The link back to the 42 real sites: the encoder is the federated one, not random."""
    path = MODELS_DIR / "forecaster_federated.pt"
    if not path.exists():
        pytest.skip("no federated forecaster; run scripts/pretrain_federated.py")
    m = ActionWorldModel()
    before = m.get_weights()["lstm.weight_ih_l0"].copy()
    assert m.warm_start_encoder(path) is True
    after = m.get_weights()["lstm.weight_ih_l0"]
    assert not np.allclose(before, after), "warm start must actually replace the encoder"


def test_warm_start_degrades_instead_of_raising_when_the_file_is_missing():
    m = ActionWorldModel()
    assert m.warm_start_encoder(MODELS_DIR / "definitely_not_here.pt") is False


def test_weights_round_trip_for_fedavg():
    """FedAvg averages these dicts; set(get(w)) must be exact or federation is lossy."""
    m = ActionWorldModel()
    w = m.get_weights()
    m.set_weights(w)
    for k, v in m.get_weights().items():
        assert np.allclose(v, w[k])


def test_save_and_load_preserves_predictions(tmp_path):
    m = ActionWorldModel()
    w = flat_window(m)
    before = m.predict(w, Action(aerator=0.5))
    m2 = ActionWorldModel.load(m.save(tmp_path / "wm.pt"))
    after = m2.predict(w, Action(aerator=0.5))
    for c in CHANNELS:
        assert np.allclose(before[c], after[c], atol=1e-6)


# ------------------------------------------------------------------ the planner seam
def _obs(do=6.0):
    return Observation(t_hours=0.0, values={"do": do, "temp_c": 28.0, "ph": 7.6,
                                            "tan": 0.4, "no2": 0.05, "turbidity": 25.0})


def test_learned_model_falls_back_until_a_full_window_exists():
    """The first ticks of a run have no window. Falling back is correct, not a failure."""
    wm = ActionWorldModel()
    lm = LearnedWorldModel(wm, InternalModel(seed=1))
    traj = lm.rollout(Action(aerator=1.0), 24, 12)
    assert lm.used_fallback == 1 and lm.used_learned == 0
    assert set(("do", "nh3", "chloride")) <= set(traj)

    for _ in range(wm.window):
        lm.update(_obs(), Action.no_op())
    lm.rollout(Action(aerator=1.0), 24, 12)
    assert lm.used_learned == 1, "with a full window the network must answer the rollout"


def test_learned_rollout_supplies_everything_the_planner_reads():
    wm = ActionWorldModel()
    lm = LearnedWorldModel(wm, InternalModel(seed=1))
    for _ in range(wm.window):
        lm.update(_obs(), Action.no_op())
    traj = lm.rollout(Action(aerator=1.0), 24, 12)
    for key in ("do", "temp_c", "ph", "tan", "no2", "turbidity", "nh3", "chloride"):
        assert key in traj and len(traj[key]) == 24
        assert np.isfinite(traj[key]).all()


def test_learned_rollout_is_clipped_to_physically_possible_values():
    """An unclipped net can predict negative DO, which would dominate the cost function and
    drive a nonsense action. The clamp is load-bearing, so it is asserted."""
    wm = ActionWorldModel()
    lm = LearnedWorldModel(wm, InternalModel(seed=1), clip=True)
    for _ in range(wm.window):
        lm.update(_obs(), Action.no_op())
    traj = lm.rollout(Action(aerator=1.0), 24, 12)
    assert (traj["do"] >= 0).all() and (traj["do"] <= 25).all()
    assert (traj["ph"] >= 4).all() and (traj["ph"] <= 11).all()
    assert (traj["nh3"] >= 0).all()


def test_learned_model_falls_back_when_the_last_observation_is_missing():
    """The delta encoding is measured from the last row; if a probe is out there, the origin
    is undefined and the physics model is the honest answer."""
    wm = ActionWorldModel()
    lm = LearnedWorldModel(wm, InternalModel(seed=1))
    for _ in range(wm.window - 1):
        lm.update(_obs(), Action.no_op())
    lm.update(Observation(t_hours=0.0, values={"do": float("nan"), "temp_c": 28.0, "ph": 7.6,
                                               "tan": 0.4, "no2": 0.05, "turbidity": 25.0}),
              Action.no_op())
    before = lm.used_fallback
    lm.rollout(Action(aerator=1.0), 24, 12)
    assert lm.used_fallback == before + 1


def test_learned_model_keeps_the_internal_twin_available():
    """`evaluate` reads `ctx.model.twin.dt_h`; the seam must not break that."""
    lm = LearnedWorldModel(ActionWorldModel(), InternalModel(seed=1))
    assert lm.twin.dt_h > 0
