"""Client splitting and data-poor designation - the bookkeeping the federated claim rests on.

These run against stub clients rather than real sites: `split_clients` and `apply_data_poor`
touch only `n_samples`, the window arrays and the stratification key, so building 42 real
`PondClient`s (each of which windows a full series and instantiates an LSTM) would buy nothing
but minutes. The regimes themselves are exercised by `scripts/pretrain_federated.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.runtime import load_config
from src.federated.simulate import apply_data_poor, split_clients


class StubClient:
    """Just enough of a `PondClient` for the split and designation code paths."""

    def __init__(self, client_id: str, source: str, n: int) -> None:
        self.client_id = client_id
        self.source = source
        self.region = "test"
        self.data_fraction = 1.0
        self.X = np.zeros((n, 4, 3), dtype=np.float32)
        self.Y = np.zeros((n, 3), dtype=np.float32)
        self.M = np.ones((n, 3), dtype=np.float32)

    @property
    def n_samples(self) -> int:
        return int(len(self.X))


def make_clients(n_usgs: int = 26, n_ukea: int = 16):
    """Sizes ascend across the whole set, so the thinnest clients are USGS by construction."""
    out = []
    for i in range(n_usgs):
        out.append(StubClient(f"usgs_{i:03d}", "usgs", 300 + 10 * i))
    for i in range(n_ukea):
        out.append(StubClient(f"ukea_{i:03d}", "ukea", 1000 + 10 * i))
    return out


@pytest.fixture
def fed_cfg():
    return load_config("federated")


# ------------------------------------------------------------------ the split
def test_split_holds_out_sites_and_keeps_them_disjoint(fed_cfg):
    clients = make_clients()
    train, holdout = split_clients(clients, fed_cfg)

    assert len(train) + len(holdout) == len(clients)
    train_ids = {c.client_id for c in train}
    hold_ids = {c.client_id for c in holdout}
    assert not (train_ids & hold_ids), "a held-out site must never also be trained on"
    assert len(holdout) >= int(fed_cfg["split"]["holdout_min"])


def test_split_keeps_both_sources_on_both_sides(fed_cfg):
    """Stratification is what makes the holdout a test of generalisation rather than of
    'does a model trained on English rivers work on English rivers'."""
    train, holdout = split_clients(make_clients(), fed_cfg)
    assert {c.source for c in holdout} == {"usgs", "ukea"}
    assert {c.source for c in train} == {"usgs", "ukea"}


def test_split_is_deterministic_under_the_configured_seed(fed_cfg):
    a, _ = split_clients(make_clients(), fed_cfg)
    b, _ = split_clients(make_clients(), fed_cfg)
    assert [c.client_id for c in a] == [c.client_id for c in b]


# ------------------------------------------------------ data-poor designation
def test_data_poor_clients_all_stay_in_the_training_split(fed_cfg):
    """The regression this file exists for.

    `apply_data_poor` used to run over ALL clients, before the split. A designated client could
    then land in the holdout, where no regime trains it, so it never reached the data-poor table
    - the headline claim was scored on fewer clients than were designated, silently.
    """
    clients = make_clients()
    train, holdout = split_clients(clients, fed_cfg)
    poor_ids = apply_data_poor(train, fed_cfg, verbose=False)

    n_designated = int(fed_cfg["data_poor"]["n_clients"])
    assert len(poor_ids) == n_designated
    train_ids = {c.client_id for c in train}
    assert set(poor_ids) <= train_ids, "every designated client must be one that actually trains"
    assert not (set(poor_ids) & {c.client_id for c in holdout})


def test_data_poor_truncates_history_to_the_configured_fraction(fed_cfg):
    train, _ = split_clients(make_clients(), fed_cfg)
    before = {c.client_id: c.n_samples for c in train}
    poor_ids = apply_data_poor(train, fed_cfg, verbose=False)

    frac = float(fed_cfg["data_poor"]["data_fraction"])
    for c in train:
        if c.client_id in poor_ids:
            assert c.data_fraction == pytest.approx(frac)
            assert c.n_samples == max(int(before[c.client_id] * frac), 1)
            assert len(c.Y) == len(c.X) == len(c.M), "X/Y/M must stay aligned after truncation"
        else:
            assert c.n_samples == before[c.client_id]
            assert c.data_fraction == 1.0


def test_data_poor_picks_the_thinnest_training_clients(fed_cfg):
    """`select: fewest_windows` should choose the clients least able to train alone - which is
    the population the federated claim is about."""
    assert fed_cfg["data_poor"]["select"] == "fewest_windows"
    train, _ = split_clients(make_clients(), fed_cfg)
    sizes = {c.client_id: c.n_samples for c in train}
    poor_ids = apply_data_poor(train, fed_cfg, verbose=False)

    expected = sorted(sizes, key=lambda k: sizes[k])[: len(poor_ids)]
    assert set(poor_ids) == set(expected)


def test_data_poor_is_a_no_op_when_disabled(fed_cfg):
    cfg = dict(fed_cfg)
    cfg["data_poor"] = dict(fed_cfg["data_poor"], n_clients=0)
    train, _ = split_clients(make_clients(), fed_cfg)
    before = [c.n_samples for c in train]
    assert apply_data_poor(train, cfg, verbose=False) == []
    assert [c.n_samples for c in train] == before
