"""Three training regimes over one identical client set: independent-local, FedAvg, central.

    independent-local  each client trains only on its own site; nothing is ever shared.
                       This is "independent models learning from different ponds", and it is
                       the baseline the federated claim has to beat.
    federated (FedAvg) each round every selected client does E local epochs on its own data;
                       the server averages the weight tensors (optionally weighted by sample
                       count) and broadcasts the average back. In-process, deterministic,
                       CPU-only, no network - the coupling is the only thing being studied.
    central            all clients' windows pooled into one model. An upper reference that
                       violates the privacy premise; reported, never recommended.

Total local work is matched across regimes (independent-local epochs == rounds * local_epochs)
so a difference cannot be explained by one regime simply training longer.

Evaluation is leave-some-sites-out: held-out sites are never trained on by any regime, so the
question being answered is "does this transfer to a pond we have never seen", not "did it
memorise the ponds we have".
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.runtime import OUTPUTS_DIR, load_config, set_seed
from ..data import registry as R
from ..perception.forecaster import LSTMForecaster
from .client import PondClient


# ======================================================================================
# Client construction and splitting
# ======================================================================================
def build_clients(
    fed_cfg: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[PondClient]:
    fed_cfg = fed_cfg or load_config("federated")
    ccfg = fed_cfg["clients"]
    sites = R.load_all(min_channels=int(ccfg["min_channels"]))[: int(ccfg["max_clients"])]
    if not sites:
        raise RuntimeError("no cached sites - run scripts/collect_data.py first")

    clients: List[PondClient] = []
    for site in sites:
        c = PondClient(
            site,
            model_cfg=model_cfg,
            seed=int(fed_cfg["seed"]),
            stride=int(ccfg["window_stride"]),
            val_fraction=float(fed_cfg["split"]["val_fraction"]),
        )
        if c.n_samples < int(ccfg["min_windows"]):
            if verbose:
                print(f"  skip {c.client_id}: only {c.n_samples} training windows")
            continue
        clients.append(c)
    if verbose:
        print(f"  {len(clients)} clients built from {len(sites)} cached sites")
    return clients


def apply_data_poor(clients: List[PondClient], fed_cfg: Dict[str, Any], verbose: bool = True) -> List[str]:
    """Designate the data-poor clients - the ones the federated claim is actually about."""
    dp = fed_cfg["data_poor"]
    n = int(dp["n_clients"])
    if n <= 0 or not clients:
        return []
    if str(dp.get("select", "fewest_windows")) == "random":
        rng = np.random.default_rng(int(fed_cfg["seed"]))
        chosen = list(rng.choice(len(clients), size=min(n, len(clients)), replace=False))
    else:
        chosen = list(np.argsort([c.n_samples for c in clients])[:n])

    ids = []
    frac = float(dp["data_fraction"])
    for i in chosen:
        c = clients[i]
        keep = max(int(len(c.X) * frac), 1)
        c.X, c.Y, c.M = c.X[:keep], c.Y[:keep], c.M[:keep]
        c.data_fraction = frac
        ids.append(c.client_id)
        if verbose:
            print(f"  data-poor: {c.client_id} -> {keep} training windows ({frac:.0%})")
    return ids


def split_clients(clients: List[PondClient], fed_cfg: Dict[str, Any]) -> Tuple[List[PondClient], List[PondClient]]:
    """Leave-some-sites-out, stratified so both sources appear on both sides."""
    scfg = fed_cfg["split"]
    rng = np.random.default_rng(int(fed_cfg["seed"]))
    n_hold = max(int(len(clients) * float(scfg["holdout_fraction"])), int(scfg["holdout_min"]))
    n_hold = min(n_hold, max(len(clients) - 2, 0))

    by_key: Dict[str, List[int]] = {}
    key_of = str(scfg.get("stratify_by", "source"))
    for i, c in enumerate(clients):
        by_key.setdefault(getattr(c, key_of, "all"), []).append(i)

    holdout_idx: List[int] = []
    keys = sorted(by_key)
    while len(holdout_idx) < n_hold:
        progressed = False
        for k in keys:
            pool = [i for i in by_key[k] if i not in holdout_idx]
            if not pool or len(holdout_idx) >= n_hold:
                continue
            holdout_idx.append(int(rng.choice(pool)))
            progressed = True
        if not progressed:
            break

    holdout = [clients[i] for i in sorted(holdout_idx)]
    train = [c for i, c in enumerate(clients) if i not in holdout_idx]
    return train, holdout


# ======================================================================================
# Weight aggregation
# ======================================================================================
def fedavg(
    weight_sets: List[Dict[str, np.ndarray]],
    sample_counts: Optional[List[int]] = None,
    dp_cfg: Optional[Dict[str, Any]] = None,
    global_weights: Optional[Dict[str, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Weighted average of client weights, with optional DP clipping + Gaussian noise."""
    if not weight_sets:
        raise ValueError("nothing to aggregate")
    counts = np.asarray(sample_counts if sample_counts else [1] * len(weight_sets), dtype=float)
    w = counts / counts.sum()

    if dp_cfg and dp_cfg.get("enabled"):
        if global_weights is None:
            raise ValueError("DP-FedAvg needs the previous global weights to form updates")
        weight_sets = _clip_updates(weight_sets, global_weights, float(dp_cfg["clip_norm"]))

    out: Dict[str, np.ndarray] = {}
    for key in weight_sets[0]:
        stack = np.stack([ws[key].astype(np.float64) for ws in weight_sets], axis=0)
        avg = np.tensordot(w, stack, axes=(0, 0))
        if dp_cfg and dp_cfg.get("enabled"):
            rng = rng or np.random.default_rng(0)
            sigma = float(dp_cfg["noise_multiplier"]) * float(dp_cfg["clip_norm"]) / len(weight_sets)
            avg = avg + rng.normal(0.0, sigma, size=avg.shape)
        out[key] = avg.astype(np.float32)
    return out


def _clip_updates(weight_sets, global_weights, clip_norm: float):
    """Clip each client's update (not its weights) to a fixed L2 norm - the DP-SGD recipe."""
    clipped = []
    for ws in weight_sets:
        delta = {k: ws[k].astype(np.float64) - global_weights[k].astype(np.float64) for k in ws}
        norm = np.sqrt(sum(float((d**2).sum()) for d in delta.values()))
        scale = min(1.0, clip_norm / (norm + 1e-12))
        clipped.append({k: global_weights[k] + delta[k] * scale for k in ws})
    return clipped


# ======================================================================================
# The three regimes
# ======================================================================================
def run_independent_local(train: List[PondClient], fed_cfg: Dict[str, Any], verbose=True) -> Dict[str, Dict[str, np.ndarray]]:
    cfg = fed_cfg["regimes"]["independent_local"]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c in train:
        c.reset_model(seed=int(fed_cfg["seed"]))
        info = c.local_train(epochs=int(cfg["epochs"]))
        out[c.client_id] = c.get_weights()
        if verbose:
            print(f"    {c.client_id:24s} n={c.n_samples:5d}  loss {info['loss']:.5f}")
    return out


def run_federated(
    train: List[PondClient], fed_cfg: Dict[str, Any], verbose=True
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    cfg = fed_cfg["regimes"]["federated"]
    dp_cfg = fed_cfg.get("differential_privacy", {})
    rounds = int(cfg["rounds"])
    local_epochs = int(cfg["local_epochs"])
    frac = float(cfg["client_fraction"])
    rng = np.random.default_rng(int(fed_cfg["seed"]))

    for c in train:
        c.reset_model(seed=int(fed_cfg["seed"]))
    # Every client starts from the SAME initialisation - otherwise averaging weights from
    # differently-initialised networks averages across permuted hidden units and is meaningless.
    global_weights = train[0].get_weights()
    for c in train:
        c.set_weights(global_weights)

    log: List[Dict[str, Any]] = []
    for rnd in range(rounds):
        k = max(int(round(len(train) * frac)), 1)
        selected = list(rng.choice(len(train), size=k, replace=False)) if k < len(train) else range(len(train))
        weights, counts, losses = [], [], []
        for i in selected:
            c = train[i]
            c.set_weights(global_weights)
            info = c.local_train(epochs=local_epochs)
            weights.append(c.get_weights())
            counts.append(c.n_samples)
            losses.append(info["loss"])
        global_weights = fedavg(
            weights,
            counts if cfg.get("weighted_by_samples", True) else None,
            dp_cfg=dp_cfg,
            global_weights=global_weights,
            rng=rng,
        )
        row = {
            "round": rnd + 1,
            "clients": len(weights),
            "mean_local_loss": float(np.mean(losses)),
            "total_samples": int(sum(counts)),
        }
        log.append(row)
        if verbose:
            print(f"    round {rnd + 1:2d}/{rounds}  clients={len(weights):2d}  "
                  f"mean local loss {row['mean_local_loss']:.5f}")
    return global_weights, pd.DataFrame(log)


def run_central(train: List[PondClient], fed_cfg: Dict[str, Any], model_cfg=None, verbose=True) -> Dict[str, np.ndarray]:
    """Pool every client's windows into one model. The privacy premise is deliberately broken."""
    cfg = fed_cfg["regimes"]["central"]
    X = np.concatenate([c.X for c in train if len(c.X)])
    Y = np.concatenate([c.Y for c in train if len(c.X)])
    M = np.concatenate([c.M for c in train if len(c.X)])
    model = LSTMForecaster(model_cfg, seed=int(fed_cfg["seed"]))
    info = model.fit(X, Y, M, epochs=int(cfg["epochs"]))
    if verbose:
        print(f"    pooled {len(X)} windows from {len(train)} clients, loss {info['loss']:.5f}")
    return model.get_weights()


# ======================================================================================
# The comparison
# ======================================================================================
def compare(
    fed_cfg: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all three regimes and score them on held-out real sites. Returns everything."""
    fed_cfg = fed_cfg or load_config("federated")
    set_seed(int(fed_cfg["seed"]))

    if verbose:
        print("Building clients (one real monitoring site = one pond) ...")
    clients = build_clients(fed_cfg, model_cfg, verbose)
    # Split BEFORE designating data-poor clients. The other order lets a designated client land
    # in the holdout, where it never trains and so never appears in the data-poor table - the
    # claim federation is actually about would be scored on fewer clients than were designated.
    train, holdout = split_clients(clients, fed_cfg)
    poor_ids = apply_data_poor(train, fed_cfg, verbose)
    if verbose:
        print(f"  train sites: {len(train)}   held-out sites: {len(holdout)} "
              f"({[c.client_id for c in holdout]})")

    results: Dict[str, Any] = {
        "clients": pd.DataFrame([c.summary() for c in clients]),
        "data_poor_ids": poor_ids,
        "train_ids": [c.client_id for c in train],
        "holdout_ids": [c.client_id for c in holdout],
    }

    if verbose:
        print("\n[independent-local] each site trains alone, nothing shared")
    local_weights = run_independent_local(train, fed_cfg, verbose)

    if verbose:
        print("\n[federated] FedAvg over the same clients")
    fed_weights, fed_log = run_federated(train, fed_cfg, verbose)
    results["fed_log"] = fed_log

    central_weights = None
    if fed_cfg["regimes"]["central"].get("enabled", True):
        if verbose:
            print("\n[central] pooled upper reference")
        central_weights = run_central(train, fed_cfg, model_cfg, verbose)

    results["weights"] = {
        "federated": fed_weights,
        "central": central_weights,
        "independent_local": local_weights,
    }

    if verbose:
        print("\nScoring on held-out real sites ...")
    results["holdout"] = _score_holdout(holdout, local_weights, fed_weights, central_weights)
    results["data_poor"] = _score_data_poor(train, poor_ids, local_weights, fed_weights, central_weights)
    return results


def _score_holdout(holdout, local_weights, fed_weights, central_weights) -> pd.DataFrame:
    """Score every regime on sites no regime has trained on.

    ``independent-local`` has no single model to test, so it is represented by the *mean over
    its per-site models* - which is the honest question: how well does a pond that trained
    alone do on a pond it has never seen? Reporting only its best member would be cherry-picking.
    """
    rows: List[Dict[str, Any]] = []
    for c in holdout:
        for cid, w in local_weights.items():
            m = c.evaluate_weights(w, on="val")
            m.update({"regime": "independent_local", "source_model": cid, "holdout_site": c.client_id})
            rows.append(m)
        m = c.evaluate_weights(fed_weights, on="val")
        m.update({"regime": "federated", "source_model": "global", "holdout_site": c.client_id})
        rows.append(m)
        if central_weights is not None:
            m = c.evaluate_weights(central_weights, on="val")
            m.update({"regime": "central", "source_model": "pooled", "holdout_site": c.client_id})
            rows.append(m)
    return pd.DataFrame(rows)


def _score_data_poor(train, poor_ids, local_weights, fed_weights, central_weights) -> pd.DataFrame:
    """The headline claim: does federation help the pond that cannot train well alone?

    Each data-poor client is scored on ITS OWN held-out tail under (a) the model it trained
    alone on its thin slice, and (b) the federated global model it helped produce without
    sharing a single row.
    """
    rows: List[Dict[str, Any]] = []
    for c in train:
        if c.client_id not in poor_ids:
            continue
        if c.client_id in local_weights:
            m = c.evaluate_weights(local_weights[c.client_id], on="val")
            m.update({"regime": "independent_local", "site": c.client_id, "data_fraction": c.data_fraction})
            rows.append(m)
        m = c.evaluate_weights(fed_weights, on="val")
        m.update({"regime": "federated", "site": c.client_id, "data_fraction": c.data_fraction})
        rows.append(m)
        if central_weights is not None:
            m = c.evaluate_weights(central_weights, on="val")
            m.update({"regime": "central", "site": c.client_id, "data_fraction": c.data_fraction})
            rows.append(m)
    return pd.DataFrame(rows)


def summarize(results: Dict[str, Any], channels=("do", "temp_c", "ph")) -> pd.DataFrame:
    """Per-regime, per-channel RMSE on held-out sites - the table the README quotes."""
    df = results["holdout"]
    if df.empty:
        return df
    keep = ["regime", "holdout_site"] + [f"rmse_{c}" for c in channels if f"rmse_{c}" in df.columns]
    agg = df[keep].groupby(["regime", "holdout_site"]).mean().reset_index()
    return agg.groupby("regime").mean(numeric_only=True).reset_index()


def write_outputs(results: Dict[str, Any], prefix: str = "federated") -> Dict[str, str]:
    paths: Dict[str, str] = {}
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, key in (("comparison", "holdout"), ("data_poor", "data_poor"), ("clients", "clients")):
        df = results.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            p = OUTPUTS_DIR / f"{prefix}_{name}.csv"
            df.to_csv(p, index=False)
            paths[name] = str(p)
    if isinstance(results.get("fed_log"), pd.DataFrame):
        p = OUTPUTS_DIR / f"{prefix}_rounds.csv"
        results["fed_log"].to_csv(p, index=False)
        paths["rounds"] = str(p)
    summary = summarize(results)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        p = OUTPUTS_DIR / f"{prefix}_summary.csv"
        summary.to_csv(p, index=False)
        paths["summary"] = str(p)
    return paths
