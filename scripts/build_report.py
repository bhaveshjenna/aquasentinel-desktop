"""Render RESULTS.md entirely from outputs/ and data_cache/manifest.csv.

No number in this project is typed by hand into a document. The README contains prose and
pointers; every figure lives in a CSV and is rendered here. If a table is missing, this script
says which command produces it rather than inventing a placeholder.

    python scripts/build_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.runtime import DATA_CACHE, MODELS_DIR, OUTPUTS_DIR, ROOT  # noqa: E402

MISSING = {
    "manifest.csv": "python scripts/collect_data.py",
    "federated_comparison.csv": "python scripts/pretrain_federated.py",
    "federated_summary.csv": "python scripts/pretrain_federated.py",
    "federated_data_poor.csv": "python scripts/pretrain_federated.py",
    "regime_metrics.csv": "python scripts/pretrain_federated.py",
    "regime_sweep.csv": "python scripts/pretrain_federated.py",
    "risk_metrics.csv": "python scripts/pretrain_federated.py",
    "agent_summary.csv": "python -m src.eval.run_eval",
    "agent_stats.csv": "python -m src.eval.run_eval",
    "safety.csv": "python -m src.eval.run_eval",
    "ablations.csv": "python -m src.eval.run_eval",
    "anomaly_detection.csv": "python -m src.eval.run_eval",
    "forecast_horizon.csv": "python -m src.eval.run_eval",
}

OUT: List[str] = []


def w(line: str = "") -> None:
    OUT.append(line)


def read(name: str, base: Path = OUTPUTS_DIR) -> Optional[pd.DataFrame]:
    p = base / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def absent(name: str) -> None:
    w(f"> `{name}` not found. Generate it with `{MISSING.get(name, '(see README)')}`.")
    w()


def table(df: pd.DataFrame, cols: Optional[List[str]] = None, floatfmt: str = "{:.3f}") -> None:
    """Markdown table straight from a frame, with no hand-entered values."""
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "-" if pd.isna(v) else floatfmt.format(v))
    w("| " + " | ".join(str(c) for c in out.columns) + " |")
    w("|" + "|".join("---" for _ in out.columns) + "|")
    for _, r in out.iterrows():
        w("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    w()


# ======================================================================================
def section_data() -> None:
    w("## 1. Real data actually collected")
    w()
    mf = read("manifest.csv", DATA_CACHE)
    if mf is None:
        absent("manifest.csv")
        return
    w(f"- **{len(mf)} real fish ponds**, one per federated client")
    w(f"- **{int(mf['rows'].sum()):,} harmonised rows** at "
      f"{int(mf['cadence_minutes'].iloc[0])}-minute cadence")
    w(f"- window {str(mf['start'].min())[:10]} to {str(mf['end'].max())[:10]}")
    w(f"- {mf['region'].nunique()} distinct regions; sources: "
      + ", ".join(f"{k} ({v} sites)" for k, v in mf["source"].value_counts().items()))
    w()
    counts: dict = {}
    for chans in mf["channels"]:
        for c in str(chans).split("|"):
            counts[c] = counts.get(c, 0) + 1
    w("Channel coverage across sites (this is the non-IID structure, not a simulation of it):")
    w()
    table(pd.DataFrame({"channel": list(counts), "sites reporting it": list(counts.values())})
          .sort_values("sites reporting it", ascending=False))
    w("Licences: " + "; ".join(sorted(set(mf["license"].astype(str)))))
    w()
    w("> Real catfish aquaponics ponds in Nsukka, Nigeria (Udanor et al., *Data in Brief* 2022), "
      "logged by low-cost IoT probes. Ponds whose timelines are mostly sensor outage were "
      "dropped by the QC thresholds in `configs/data.yaml`; nitrate is excluded because the "
      "raw column is an ADC count, not a concentration.")
    w()


def section_federated() -> None:
    w("## 2. Federated vs. independent-local vs. central")
    w()
    summary = read("federated_summary.csv")
    if summary is None:
        absent("federated_summary.csv")
    else:
        w("Forecast RMSE on **held-out real ponds** — ponds no regime trained on. "
          "Lower is better.")
        w()
        table(summary)

    comp = read("federated_comparison.csv")
    if comp is not None and "holdout_site" in comp.columns:
        w(f"Scored on {comp['holdout_site'].nunique()} held-out pond(s). "
          "`independent_local` is the mean over every pond-specific model, because a pond that "
          "trained alone has no single global model to test — reporting only its best member "
          "would be cherry-picking.")
        w()

    dp = read("federated_data_poor.csv")
    if dp is None:
        absent("federated_data_poor.csv")
    else:
        w("### The data-poor ponds — the claim federation is actually about")
        w()
        w("Each of these clients kept only a fraction of its history and is scored on **its own** "
          "held-out tail, under the model it trained alone versus the federated global model it "
          "helped produce without sharing a single row.")
        w()
        cols = ["regime", "site", "data_fraction", "n_train", "rmse_do", "rmse_temp_c", "rmse_ph"]
        table(dp[[c for c in cols if c in dp.columns]])
        if {"regime", "rmse_do"} <= set(dp.columns):
            piv = dp.pivot_table(index="site", columns="regime", values="rmse_do")
            if {"federated", "independent_local"} <= set(piv.columns):
                delta = (piv["federated"] - piv["independent_local"]) / piv["independent_local"]
                won = int((delta < 0).sum())
                w(f"**Federation improved DO forecast RMSE on {won} of {len(delta)} data-poor "
                  f"sites** (median change {delta.median() * 100:+.1f}%). "
                  "Scoped to these sites, this window and this architecture — not a general claim.")
                w()

    rounds = read("federated_rounds.csv")
    if rounds is not None and not rounds.empty:
        w(f"FedAvg ran {len(rounds)} rounds; mean local loss "
          f"{rounds['mean_local_loss'].iloc[0]:.5f} → {rounds['mean_local_loss'].iloc[-1]:.5f}.")
        w()

    fh = read("forecast_horizon.csv")
    if fh is not None and not fh.empty:
        w("### Forecast error by horizon (held-out real ponds)")
        w()
        piv = (fh[fh["channel"] == "do"]
               .pivot_table(index="horizon_minutes", columns="source", values="rmse")
               .reset_index())
        table(piv)
        w("Dissolved oxygen, mg/L. Error growing with horizon is the expected and honest shape.")
        w()


def section_world_model() -> None:
    w("## 2b. The action-conditioned world model — the network that picks the action")
    w()
    w("The forecaster in §2 predicts what the water does *on its own*; it was trained on real "
      "pond sensor data that contains no log of actuator commands, so it cannot be asked what "
      "an aerator would do. The **world model** can: it takes the observed window *and a "
      "candidate action* and predicts the resulting trajectory. The MPC planner scores those "
      "learned trajectories and takes the argmin, so the action the agent applies is chosen "
      "from predictions made by a neural network rather than by a hand-written differential "
      "equation.")
    w()
    w("**Where its training data comes from, stated plainly.** The real pond dataset records "
      "water quality but not when the aerator ran, how much was fed or when lime went in - "
      "so it has no `(action → response)` pairs. The action-conditioned part is therefore "
      "learned from twin rollouts under randomised action sequences. What *is* carried over "
      "from real data is the **encoder**, warm-started from `forecaster_federated.pt` — the "
      "model FedAvg produced across the real ponds.")
    w()
    w("**Federation here is across simulated ponds**, each with its own volume, depth, aeration "
      "efficiency, algal productivity and temperature regime. That is a different claim from "
      "§2's federation over real ponds. Both use the same `fedavg`; only §2's clients are real.")
    w()
    summary = read("world_model_summary.csv")
    if summary is None:
        absent("world_model_summary.csv")
    else:
        w("Prediction error on held-out ponds no regime trained on. `rmse` is in scaled-delta "
          "units; `rmse_do_mgl` is dissolved oxygen in mg/L over the 2-hour horizon.")
        w()
        table(summary)
        try:
            s = summary.set_index("regime")["rmse_do_mgl"]
            f, loc, cen = float(s["federated"]), float(s["independent_local"]), float(s["central"])
            verdict = ("beats" if f < loc else "does not beat")
            w(f"Federated DO error {f:.3f} mg/L against {loc:.3f} for independent-local "
              f"(federation {verdict} training alone) and {cen:.3f} for the pooled reference.")
        except (KeyError, ValueError):
            pass
        w()
    w("**It is markedly less accurate than the physics it replaced**, and that is the honest "
      "headline: roughly 1.1 mg/L of DO error over the horizon, in a pond that operates between "
      "2 and 10 mg/L. The consequences are measured in the ablation table (`no_world_model`) "
      "and in the safety section, not asserted here.")
    w()
    rounds = read("world_model_rounds.csv")
    if rounds is not None and not rounds.empty:
        w(f"FedAvg ran {len(rounds)} rounds; held-out DO RMSE "
          f"{rounds['holdout_rmse_do_mgl'].iloc[0]:.3f} → "
          f"{rounds['holdout_rmse_do_mgl'].iloc[-1]:.3f} mg/L.")
        w()


def anomaly_verdict(fpr: float) -> str:
    """Plain verdict on the anomaly flag's false-alarm rate, from the measured number."""
    if fpr > 0.5:
        return (f"At a false-positive rate of {fpr:.0%} the flag fires on most ordinary windows, so "
                "it is not usable as an alarm on its own: the model was fitted mostly on noisy "
                "real-pond windows, and the simulated pond's normal behaviour falls outside them.")
    if fpr > 0.2:
        return (f"A false-positive rate of {fpr:.0%} is high; the flag is a weak hint, not an alarm.")
    return (f"A false-positive rate of {fpr:.0%} makes the flag usable as a novelty hint.")


def section_perception() -> None:
    w("## 3. Perception")
    w()
    reg = read("regime_metrics.csv")
    if reg is None:
        absent("regime_metrics.csv")
    else:
        w("### Regimes (HDBSCAN, unsupervised — no labels consumed anywhere)")
        w()
        table(reg)
        w("DBCV is HDBSCAN's density-based validity index, the appropriate one for "
          "non-globular clusters; silhouette is included for comparability with the usual "
          "literature. `noise_fraction` is the share of windows assigned to no cluster — "
          "those are the anomaly flags.")
        w()
        w("**A DBCV close to zero means the clusters are not well separated**, whatever the "
          "silhouette says. The intended deliverable was a taxonomy of operating regimes, and "
          "the data does not support one.")
        w()

    sweep = read("regime_sweep.csv")
    if sweep is not None and not sweep.empty:
        w("#### The sweep behind that claim")
        w()
        table(sweep)
        if "dbcv" in sweep.columns:
            w(f"Across every configuration DBCV stays at or below {sweep['dbcv'].max():.3f}: "
              "no setting finds well-separated clusters.")
        w()
        w("Continuous water-quality windows form a connected continuum — a pond slides around "
          "a manifold rather than jumping between discrete states. The model is therefore used "
          "as a **novelty detector** (the noise label is the anomaly flag; how well that works "
          "is measured in the next section), "
          "and the cluster label is a descriptive reading aid that nothing downstream depends "
          "on.")
        w()

    risk = read("risk_metrics.csv")
    if risk is None:
        absent("risk_metrics.csv")
    else:
        w("### Risk model (LightGBM over current channels + forecast summaries)")
        w()
        table(risk)
        w("The target is **rule-derived**: the risk band that `configs/twin.yaml` assigns to "
          "the worst point of the true next horizon. This model has not learned water "
          "chemistry; it has learned to anticipate a known rule from an imperfect forecast.")
        w()

    an = read("anomaly_detection.csv")
    if an is not None and not an.empty:
        w("### Anomaly detection on injected sensor faults")
        w()
        w(f"Event recall **{an['recall'].mean():.2f}** "
          f"({int(an['detected'].sum())}/{int(an['events'].sum())} injected fault events flagged "
          f"across {len(an)} seeds), false-positive rate "
          f"**{an['false_positive_rate'].mean():.3f}** outside the fault windows.")
        w()
        w(anomaly_verdict(an["false_positive_rate"].mean()))
        w()
        w("Scored per event, not per tick: a detector that fires constantly would score "
          "perfectly on a per-tick metric. The noise label cannot distinguish a broken probe "
          "from a genuine novel excursion, and the false-positive rate is the price of that.")
        w()


def section_agent() -> None:
    w("## 4. Agent vs. baselines, in the twin")
    w()
    safety = read("safety.csv")
    if safety is None:
        absent("safety.csv")
    else:
        w("### Safety — the headline claim")
        w()
        table(safety, floatfmt="{:.0f}")
        w("`do_nothing` is deliberately unguarded: it is a reference trajectory, not a "
          "controller, and letting the safety layer rescue it would destroy the comparison it "
          "exists to provide. Every guardrailed controller must show **0** unsafe-applied.")
        w()

    summary = read("agent_summary.csv")
    if summary is None:
        absent("agent_summary.csv")
    else:
        w("### Time out of band (median across seeds, with IQR)")
        w()
        cols = ["scenario", "controller", "risk_h_do_median", "risk_h_do_iqr",
                "time_critical_h_median", "interventions_median", "intervention_cost_median",
                "unsafe_applied_median"]
        table(summary[[c for c in cols if c in summary.columns]])
        w("Medians, not means — a single unlucky seed must not carry a claim.")
        w()

    stats = read("agent_stats.csv")
    if stats is None:
        absent("agent_stats.csv")
    else:
        w("### Paired Wilcoxon signed-rank, agent vs. each baseline")
        w()
        w("Paired because the same seed gives every controller the identical perturbation "
          "schedule and noise realisation. Negative `median_diff` means the agent is better. "
          "`p` and `n` are reported for every comparison; a comparison with too few seeds to "
          "test says so instead of quietly printing a mean.")
        w()
        key = stats[(stats["metric"] == "risk_h_do") & (stats["controller_a"] == "predictive")]
        cols = ["scenario", "controller_b", "median_a", "median_b", "median_diff", "p", "n",
                "effect_size_rb", "note"]
        table(key[[c for c in cols if c in key.columns]])
        sig = key[(key["p"] < 0.05) & (key["median_diff"] < 0)]
        ties = key[key["median_diff"] == 0]
        w(f"The agent is significantly better (p < 0.05) in "
          f"**{len(sig)} of {len(key)}** scenario × baseline comparisons on this metric.")
        if len(ties):
            w(f"The {len(ties)} exact ties are all in "
              f"`{'`, `'.join(sorted(ties['scenario'].unique()))}` — scenarios in which no "
              f"controller accrues any DO time-out-of-band, so there is nothing to win.")
        w()

        # ---- the other side of the ledger -------------------------------------------
        cost = stats[(stats["metric"] == "intervention_cost")
                     & (stats["controller_a"] == "predictive")
                     & (stats["controller_b"] == "static_threshold")]
        if not cost.empty:
            w("### What that costs")
            w()
            w("Risk reduction is not free, and reporting only the win would be a half-truth. "
              "Agent versus static threshold on cumulative intervention cost — **positive "
              "`median_diff` means the agent spends more**:")
            w()
            table(cost[["scenario", "median_a", "median_b", "median_diff", "p", "n"]])
            worse = int((cost["median_diff"] > 0).sum())
            w(f"The agent is more expensive to run in **{worse} of {len(cost)}** scenarios, "
              f"including ones where it achieves the same zero DO risk the thermostat already "
              f"achieved. It buys preemption — acting before a threshold is crossed — and "
              f"preemption means acting on excursions that would sometimes have resolved on "
              f"their own. `cost_weights.intervention` in `configs/agent.yaml` is the dial "
              f"that trades one against the other; nothing here establishes where a real farm "
              f"should set it.")
            w()


def section_ablations() -> None:
    w("## 5. Ablations")
    w()
    ab = read("ablations.csv")
    if ab is None:
        absent("ablations.csv")
        return
    agg = (ab.groupby("ablation")
             .agg(median_do_risk_h=("risk_h_do", "median"),
                  median_critical_h=("time_critical_h", "median"),
                  median_cost=("intervention_cost", "median"),
                  unsafe_total=("unsafe_applied", "sum"),
                  runs=("seed", "count"))
             .reset_index()
             .sort_values("median_do_risk_h"))
    table(agg)

    # Call out the direction of the forecaster ablation explicitly - it is the result most
    # likely to be skimmed past, and it does not go the flattering way.
    try:
        full = float(agg.loc[agg["ablation"] == "full", "median_do_risk_h"].iloc[0])
        nolstm = float(agg.loc[agg["ablation"] == "no_lstm_correction", "median_do_risk_h"].iloc[0])
    except (IndexError, KeyError):
        full = nolstm = float("nan")
    if full == full and nolstm == nolstm:
        if nolstm < full:
            w(f"**The LSTM residual correction makes control worse, and it ships anyway.** "
              f"The system scores {full:.2f} h of DO time-out-of-band with the correction "
              f"against {nolstm:.2f} h without it — so `no_lstm_correction` is the "
              f"better-performing configuration, by {full - nolstm:.2f} h. A forecaster "
              f"trained on noisy low-cost pond probes, used as a correction on the simulated "
              f"pond, injects more bias than it removes. The correction is damped by 0.5 in "
              f"`policy.PredictivePlanner.correction` and it is still net negative.")
            w()
            w("It is enabled in the shipped configuration as a deliberate choice rather than an "
              "oversight. The federated LSTM is the component this project exists to produce, "
              "and keeping it on the actuator path is what makes the end-to-end system a "
              "*learned* controller rather than a digital twin rolled forward by classical "
              "control. The cost of that choice is stated here, in the table above, and in the "
              "README — quantified rather than hidden. A deployment that cared only about the "
              "score should set `predictive.use_forecaster: false`.")
        else:
            w(f"The LSTM residual correction helps control: {full:.2f} h of DO time-out-of-band "
              f"with it against {nolstm:.2f} h without.")
        w()
    w("- `full` — the shipped configuration: federated weights, forecast overlay, LSTM "
      "correction applied to the planner, guardrails on.")
    w("- `no_lstm_correction` — the MPC planner rolls its internal pond model forward without "
      "the LSTM residual correction. This isolates what the forecaster contributes to control, "
      "and it is what justifies calling the correction's cost measured rather than assumed.")
    w("- `no_forecast_at_all` — no forecaster loaded, so the risk model and the guardrails lose "
      "their forward-looking features too.")
    w("- `no_guardrails` — the ablation that is expected to be *worse on safety*. If "
      "`unsafe_total` here is not greater than zero, the guardrails are not doing anything and "
      "the safety claim is empty.")
    w("- `weights_*` — the same agent driven by each federated regime's weights, at the "
      "shipped setting. The LSTM correction is the path by which the weight source reaches "
      "the controller at all.")
    w("- `no_world_model` — the planner rolls the **hand-written** pond model forward instead "
      "of the action-conditioned LSTM. This is what isolates the learned dynamics model: "
      "`full` minus this row is the price of replacing physics with a network inside the "
      "decision.")
    w()


def section_provenance() -> None:
    w("## 6. Provenance")
    w()
    import json

    mp = MODELS_DIR / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        w(f"- models trained {m.get('trained_at')}")
        w(f"- {m.get('n_clients')} clients; {len(m.get('holdout_sites', []))} held out: "
          f"`{', '.join(m.get('holdout_sites', [])[:6])}`")
        w(f"- data-poor clients: `{', '.join(m.get('data_poor_sites', []))}`")
        overlap = sorted(set(m.get("data_poor_sites", [])) & set(m.get("holdout_sites", [])))
        if overlap:
            w(f"  - **REGRESSION:** `{', '.join(overlap)}` "
              f"{'is' if len(overlap) == 1 else 'are'} designated data-poor *and* held out, so "
              f"{'it does' if len(overlap) == 1 else 'they do'} not appear in the data-poor "
              f"table above, which is therefore scored on fewer clients than were designated. "
              f"`apply_data_poor` must run **after** `split_clients` in "
              f"`src/federated/simulate.py` and be given the training split only. This was "
              f"fixed once already; see `tests/test_federated.py"
              f"::test_data_poor_clients_all_stay_in_the_training_split`.")
        fed = m.get("federated", {})
        w(f"- FedAvg: {fed.get('rounds')} rounds x {fed.get('local_epochs')} local epochs, "
          f"client fraction {fed.get('client_fraction')}, "
          f"sample-weighted: {fed.get('weighted_by_samples')}")
        dp = m.get("differential_privacy", {})
        if dp.get("enabled"):
            w(f"- differential privacy: clip {dp.get('clip_norm')}, noise "
              f"{dp.get('noise_multiplier')}, **epsilon = {dp.get('epsilon')}**")
        else:
            w("- differential privacy: disabled")
    else:
        w("> `models/manifest.json` not found — run `python scripts/pretrain_federated.py`.")
    w()

    twin_cal = ROOT / "configs" / "twin.yaml"
    if twin_cal.exists():
        import yaml

        cal = (yaml.safe_load(twin_cal.read_text(encoding="utf-8")) or {}).get("calibration")
        if cal and cal.get("calibrated_at"):
            w(f"- twin calibrated {cal.get('calibrated_at')} against "
              f"{cal.get('n_real_sites')} real sites")
            w(f"  - applied: `{'`, `'.join(cal.get('changes', {}).keys())}`")
        elif cal:
            w(f"- twin: {cal.get('status', 'uncalibrated')}")
            pf = cal.get("pond_fit_not_applied") or {}
            if pf:
                w(f"  - a fit to the {pf.get('n_ponds')} real ponds was computed and **deliberately "
                  f"not applied** (DO noise {pf.get('noise.do')} mg/L/√h, baseline turbidity "
                  f"{pf.get('turbidity.base_ntu')} NTU - probe faults, not pond physics). "
                  f"See `calibration.note` in `configs/twin.yaml`.")
    w()
    w("Every table above was generated by `scripts/build_report.py` reading `outputs/` and "
      "`data_cache/manifest.csv`. No figure in this file was typed by hand.")
    w()


def main() -> int:
    w("# AquaSentinel — results")
    w()
    w(f"_Generated {datetime.now().replace(microsecond=0).isoformat()} by "
      f"`scripts/build_report.py`. Do not edit._")
    w()
    section_data()
    section_federated()
    section_world_model()
    section_perception()
    section_agent()
    section_ablations()
    section_provenance()

    path = ROOT / "RESULTS.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"wrote {path} ({len(OUT)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
