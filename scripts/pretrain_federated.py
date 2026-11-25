"""Phase 6: federated pretraining across independent per-pond clients. Fully offline.

    python scripts/pretrain_federated.py                 # all three regimes + models/ + outputs/
    python scripts/pretrain_federated.py --skip-central
    python scripts/pretrain_federated.py --rounds 4 --local-epochs 1   # quick pass
    python scripts/pretrain_federated.py --dp                          # DP-FedAvg stretch

Writes to models/:
    forecaster_federated.pt        the weights the dashboard runs by default
    forecaster_central.pt          upper reference
    forecaster_local_<site>.pt     one per client, for the independent-local switch
    regimes.pkl                    HDBSCAN regime model (real sites + twin scenarios)
    risk_model.pkl                 LightGBM risk model over forecast summaries
    manifest.json                  what was trained, on what, when

Writes to outputs/: federated_comparison.csv, federated_data_poor.csv, federated_summary.csv,
federated_rounds.csv, federated_clients.csv, regime_metrics.csv, risk_metrics.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.runtime import MODELS_DIR, OUTPUTS_DIR, load_config, set_seed  # noqa: E402
from src.data import registry as R  # noqa: E402
from src.federated import simulate as FS  # noqa: E402
from src.perception.forecaster import LSTMForecaster  # noqa: E402
from src.perception.regimes import RegimeModel  # noqa: E402
from src.perception.risk import RiskModel, build_training_set  # noqa: E402
from src.twin.scenarios import load_scenarios  # noqa: E402


def twin_frames(n_scenarios: int = 8, seeds=(1, 2, 3, 4)) -> tuple:
    """Twin trajectories for the regime and risk models.

    The forecaster is trained on real data only. The regime and risk models are fitted on real
    data AND twin runs, because they have to operate in the twin's regime space at run time and
    the real sites contain no aerator, no feeding and almost no ammonia excursions. This is
    stated rather than hidden: the regime clusters are partly a property of the simulation.
    """
    from src.agent.loop import AgentRun

    frames, groups = [], []
    for sid in list(load_scenarios())[:n_scenarios]:
        for seed in seeds:
            run = AgentRun(sid, controller="do_nothing", seed=seed, with_shadow=False).run()
            frames.append(run.frame())
            groups.append(seed)          # the split boundary: an unseen noise realisation
    return frames, groups


def main() -> int:
    ap = argparse.ArgumentParser(description="Federated pretraining (offline).")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-epochs", type=int, default=None)
    ap.add_argument("--skip-central", action="store_true")
    ap.add_argument("--skip-perception", action="store_true", help="forecaster only")
    ap.add_argument("--dp", action="store_true", help="enable DP-FedAvg (clip + Gaussian noise)")
    ap.add_argument("--only-perception", action="store_true",
                    help="refit the regime and risk models against existing forecaster weights")
    args = ap.parse_args()

    fed_cfg = json.loads(json.dumps(load_config("federated")))  # deep copy
    if args.rounds is not None:
        fed_cfg["regimes"]["federated"]["rounds"] = args.rounds
        fed_cfg["regimes"]["independent_local"]["epochs"] = args.rounds * fed_cfg["regimes"]["federated"]["local_epochs"]
        fed_cfg["regimes"]["central"]["epochs"] = fed_cfg["regimes"]["independent_local"]["epochs"]
    if args.local_epochs is not None:
        fed_cfg["regimes"]["federated"]["local_epochs"] = args.local_epochs
    if args.skip_central:
        fed_cfg["regimes"]["central"]["enabled"] = False
    if args.dp:
        fed_cfg["differential_privacy"]["enabled"] = True

    set_seed(int(fed_cfg["seed"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("AquaSentinel - federated pretraining across independent pond clients")
    print("=" * 78)
    info = R.summary()
    print(f"  real data: {info['n_sites']} sites, {info['n_rows']:,} rows, "
          f"{info['sources']}, {info['date_range'][0][:10]} -> {info['date_range'][1][:10]}")
    print(f"  matched work: independent-local {fed_cfg['regimes']['independent_local']['epochs']} epochs "
          f"== {fed_cfg['regimes']['federated']['rounds']} rounds x "
          f"{fed_cfg['regimes']['federated']['local_epochs']} local epochs\n")

    if args.only_perception:
        return _perception_only(fed_cfg)

    results = FS.compare(fed_cfg)
    paths = FS.write_outputs(results)

    # ---------------------------------------------------------------- save weights
    model = LSTMForecaster()
    model.set_weights(results["weights"]["federated"])
    model.save(MODELS_DIR / "forecaster_federated.pt")
    if results["weights"]["central"] is not None:
        model.set_weights(results["weights"]["central"])
        model.save(MODELS_DIR / "forecaster_central.pt")
    for cid, w in results["weights"]["independent_local"].items():
        model.set_weights(w)
        model.save(MODELS_DIR / f"forecaster_local_{cid}.pt")
    print(f"\n  weights -> {MODELS_DIR}")

    summary = FS.summarize(results)
    print("\nHeld-out real sites (RMSE, lower is better):")
    print(summary.to_string(index=False))
    if not results["data_poor"].empty:
        print("\nData-poor clients on their own held-out tail:")
        cols = ["regime", "site", "rmse_do", "rmse_temp_c", "rmse_ph", "n_train"]
        dp = results["data_poor"]
        print(dp[[c for c in cols if c in dp.columns]].to_string(index=False))

    manifest = {
        "trained_at": datetime.now().replace(microsecond=0).isoformat(),
        "n_clients": int(len(results["clients"])),
        "train_sites": results["train_ids"],
        "holdout_sites": results["holdout_ids"],
        "data_poor_sites": results["data_poor_ids"],
        "federated": fed_cfg["regimes"]["federated"],
        "differential_privacy": (
            {**fed_cfg["differential_privacy"], "epsilon": _epsilon(fed_cfg)}
            if fed_cfg["differential_privacy"]["enabled"] else {"enabled": False}
        ),
        "real_data": info,
        "outputs": paths,
    }

    # ---------------------------------------------------------------- perception
    if not args.skip_perception:
        fedmodel = LSTMForecaster.load(MODELS_DIR / "forecaster_federated.pt")

        print("\n[regimes] fitting HDBSCAN on real sites + twin scenarios ...")
        regimes = RegimeModel()
        feats = []
        for site in R.load_all(min_channels=3):
            arr = site.matrix(regimes.channels)
            f = regimes.featurize_series(arr, stride=12)
            feats.append(f[np.isfinite(f).all(axis=1)])
        tframes, tgroups = twin_frames()
        for fr in tframes:
            arr = np.column_stack([fr[c].to_numpy() for c in regimes.channels])
            feats.append(regimes.featurize_series(arr, stride=6))
        F = np.concatenate([f for f in feats if len(f)])
        regimes.fit(F, verbose=True)
        regimes.save(MODELS_DIR / "regimes.pkl")
        pd.DataFrame([regimes.metrics_]).to_csv(OUTPUTS_DIR / "regime_metrics.csv", index=False)
        # Evidence for the "the density structure is not there" claim in regimes.py.
        from src.perception.regimes import sweep as regime_sweep

        regime_sweep(F).to_csv(OUTPUTS_DIR / "regime_sweep.csv", index=False)
        manifest["regimes"] = regimes.metrics_
        manifest["regime_names"] = {str(k): v for k, v in regimes.names.items()}

        print("\n[risk] building the rule-derived training set over twin trajectories ...")
        rows, labels, groups = build_training_set(tframes, fedmodel, stride=2, groups=tgroups)
        risk, metrics = _fit_risk(rows, labels, groups)
        risk.save(MODELS_DIR / "risk_model.pkl")
        pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "risk_metrics.csv", index=False)
        print(f"  risk model: {metrics}")
        manifest["risk"] = metrics

    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nDone. models/ and outputs/ written; manifest at {MODELS_DIR / 'manifest.json'}")
    return 0


def _perception_only(fed_cfg) -> int:
    """Refit regimes + risk against the forecaster weights already in models/."""
    fedmodel = LSTMForecaster.load(MODELS_DIR / "forecaster_federated.pt")

    print("[regimes] fitting HDBSCAN on real sites + twin scenarios ...")
    regimes = RegimeModel()
    feats = []
    for site in R.load_all(min_channels=3):
        arr = site.matrix(regimes.channels)
        f = regimes.featurize_series(arr, stride=12)
        feats.append(f[np.isfinite(f).all(axis=1)])
    tframes, tgroups = twin_frames()
    for fr in tframes:
        arr = np.column_stack([fr[c].to_numpy() for c in regimes.channels])
        feats.append(regimes.featurize_series(arr, stride=6))
    F = np.concatenate([f for f in feats if len(f)])
    regimes.fit(F, verbose=True)
    regimes.save(MODELS_DIR / "regimes.pkl")
    pd.DataFrame([regimes.metrics_]).to_csv(OUTPUTS_DIR / "regime_metrics.csv", index=False)
    from src.perception.regimes import sweep as regime_sweep

    regime_sweep(F).to_csv(OUTPUTS_DIR / "regime_sweep.csv", index=False)

    print()
    print("[risk] building the rule-derived training set over twin trajectories ...")
    rows, labels, groups = build_training_set(tframes, fedmodel, stride=2, groups=tgroups)
    risk, metrics = _fit_risk(rows, labels, groups)
    risk.save(MODELS_DIR / "risk_model.pkl")
    pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "risk_metrics.csv", index=False)
    print(f"  risk model: {metrics}")

    import json as _json

    mp = MODELS_DIR / "manifest.json"
    manifest = _json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    manifest["regimes"] = regimes.metrics_
    manifest["regime_names"] = {str(k): v for k, v in regimes.names.items()}
    manifest["risk"] = metrics
    mp.write_text(_json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print("Perception models refreshed.")
    return 0


def _fit_risk(rows, labels, groups):
    """Train on all but the last seed; test on that seed across EVERY scenario.

    Holding out a seed rather than a slice of the array means the test set contains the same
    mix of scenarios as the training set, differing only in the noise realisation. Slicing the
    array positionally would hold out whole scenarios instead and silently turn an ordinary
    holdout into a transfer test.
    """
    groups = np.asarray(groups)
    held = sorted(set(groups.tolist()))[-1]
    train_idx = np.where(groups != held)[0]
    test_idx = np.where(groups == held)[0]
    print(f"  split: train on seeds {sorted(set(groups.tolist()) - {held})}, "
          f"test on seed {held} ({len(train_idx)} / {len(test_idx)} rows)")

    risk = RiskModel()
    risk.fit([rows[i] for i in train_idx], [labels[i] for i in train_idx], verbose=True)
    metrics = risk.evaluate([rows[i] for i in test_idx], [labels[i] for i in test_idx])
    metrics["held_out_seed"] = int(held)
    return risk, metrics


def _epsilon(fed_cfg) -> str:
    """Report a real accountant value or say UNACCOUNTED. Never guess an epsilon."""
    dp = fed_cfg["differential_privacy"]
    try:
        from dp_accounting import dp_event, rdp

        accountant = rdp.RdpAccountant()
        rounds = int(fed_cfg["regimes"]["federated"]["rounds"])
        event = dp_event.SelfComposedDpEvent(
            dp_event.GaussianDpEvent(float(dp["noise_multiplier"])), rounds
        )
        accountant.compose(event)
        return f"{accountant.get_epsilon(float(dp['delta'])):.3f}"
    except Exception:
        return "UNACCOUNTED (install dp-accounting for an RDP epsilon)"


if __name__ == "__main__":
    raise SystemExit(main())
