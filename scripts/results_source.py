"""Every number the Review-2 report and slides quote, read from outputs/ at build time.

The Review-2 documents used to carry their tables and figures as typed-in literals, which is
exactly how a document drifts from the code: the data source changed and every literal went
stale at once. Now both ``make_review2_report.py`` and ``make_review2_ppt.py`` call ``load()``
and format what it returns, so a re-run of the pipeline is a re-run of the documents.

Verdicts ("federation beats training alone", "the correction helps or hurts") are computed
here too, from the same numbers, so the prose cannot claim a direction the data does not show.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _csv(name: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the pipeline (see README) before building docs")
    return pd.read_csv(p)


def f2(x: float) -> str:
    return f"{float(x):.2f}"


def f3(x: float) -> str:
    return f"{float(x):.3f}"


def pct(x: float) -> str:
    return f"{float(x) * 100:+.1f} %"


CHANNEL_NAMES = {
    "do": "Dissolved oxygen", "temp_c": "Temperature", "ph": "pH",
    "tan": "Total ammonia nitrogen", "turbidity": "Turbidity", "conductivity": "Conductivity",
    "no3": "Nitrate", "no2": "Nitrite", "chlorophyll": "Chlorophyll", "bga": "Blue-green algae",
}


@lru_cache(maxsize=1)
def load() -> Dict[str, Any]:
    r: Dict[str, Any] = {}

    # ------------------------------------------------------------------ data
    mf = pd.read_csv(ROOT / "data_cache" / "manifest.csv")
    r["n_ponds"] = int(len(mf))
    r["n_rows"] = int(mf["rows"].sum())
    r["start"] = str(mf["start"].min())[:10]
    r["end"] = str(mf["end"].max())[:10]
    r["ponds"] = [c.replace("kaggle_pond_", "") for c in mf["client_id"]]
    counts: Dict[str, int] = {}
    for chans in mf["channels"]:
        for c in str(chans).split("|"):
            counts[c] = counts.get(c, 0) + 1
    r["coverage"] = [[CHANNEL_NAMES.get(c, c), str(n)]
                     for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    r["pond_rows"] = [[c.replace("kaggle_pond_", ""), f"{int(n):,}", str(s)[:10], str(e)[:10],
                       f"{float(m) * 100:.0f} %"]
                      for c, n, s, e, m in zip(mf["client_id"], mf["rows"], mf["start"],
                                               mf["end"], mf["missing_frac"])]

    mp = ROOT / "models" / "manifest.json"
    man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    strip = lambda xs: [x.replace("kaggle_pond_", "") for x in xs]  # noqa: E731
    r["train_ponds"] = strip(man.get("train_sites", []))
    r["holdout_ponds"] = strip(man.get("holdout_sites", []))
    r["poor_ponds"] = strip(man.get("data_poor_sites", []))
    fed_cfg = man.get("federated", {})
    r["rounds"] = int(fed_cfg.get("rounds", 0))
    r["local_epochs"] = int(fed_cfg.get("local_epochs", 0))

    # ------------------------------------------------------------------ federated forecaster
    fs = _csv("federated_summary.csv").set_index("regime")
    r["fed"] = {k: fs.loc[k].to_dict() for k in fs.index}
    label = {"central": "Central (pools all data; breaks the privacy premise)",
             "federated": "Federated (FedAvg, no raw data shared)",
             "independent_local": "Independent-local (each pond alone)"}
    r["fed_rows"] = [[label[k], f3(fs.loc[k, "rmse_do"]), f3(fs.loc[k, "rmse_temp_c"]),
                      f3(fs.loc[k, "rmse_ph"])] for k in ("central", "federated",
                                                          "independent_local")]
    fd, loc, cen = (fs.loc[k, "rmse_do"] for k in ("federated", "independent_local", "central"))
    r["fed_do"], r["loc_do"], r["cen_do"] = float(fd), float(loc), float(cen)
    r["fed_vs_local"] = (fd - loc) / loc
    r["fed_beats_local_all"] = bool(all(fs.loc["federated", c] <= fs.loc["independent_local", c]
                                        for c in ("rmse_do", "rmse_temp_c", "rmse_ph")))
    rounds = _csv("federated_rounds.csv")
    r["loss_first"] = float(rounds["mean_local_loss"].iloc[0])
    r["loss_last"] = float(rounds["mean_local_loss"].iloc[-1])

    dp = _csv("federated_data_poor.csv")
    piv = dp.pivot_table(index="site", columns="regime", values="rmse_do")
    r["poor_rows"] = []
    for site, row in piv.iterrows():
        r["poor_rows"].append([site.replace("kaggle_pond_", ""),
                               f3(row["independent_local"]), f3(row["federated"]),
                               f3(row["central"]),
                               pct((row["federated"] - row["independent_local"])
                                   / row["independent_local"])])
    delta = (piv["federated"] - piv["independent_local"]) / piv["independent_local"]
    r["poor_won"], r["poor_n"] = int((delta < 0).sum()), int(len(delta))
    r["poor_median"] = float(delta.median())
    # With only a couple of data-poor ponds a median hides the split, so keep each change.
    r["poor_changes"] = ", ".join(
        f"{s.replace('kaggle_pond_', '')} {pct(v)}" for s, v in delta.items())
    r["poor_ntrain"] = int(dp["n_train"].min())
    r["poor_frac"] = float(dp["data_fraction"].iloc[0])
    r["n_train_ponds"] = len(r["train_ponds"])
    r["n_holdout"] = len(r["holdout_ponds"])
    import yaml
    r["max_missing"] = float(yaml.safe_load((ROOT / "configs" / "data.yaml")
                                            .read_text(encoding="utf-8"))["targets"]["max_missing_frac"])

    # ------------------------------------------------------------------ world model
    wm = _csv("world_model_summary.csv").set_index("regime")
    r["wm_rows"] = [[k.replace("_", "-").capitalize(), f3(wm.loc[k, "rmse"]),
                     f3(wm.loc[k, "rmse_do_mgl"])]
                    for k in ("central", "federated", "independent_local")]
    r["wm_fed"], r["wm_loc"], r["wm_cen"] = (float(wm.loc[k, "rmse_do_mgl"])
                                             for k in ("federated", "independent_local", "central"))

    # ------------------------------------------------------------------ perception
    risk = _csv("risk_metrics.csv").iloc[0]
    r["risk"] = {k: float(risk[k]) for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                            "critical_recall", "critical_precision")}
    r["risk"]["n"] = int(risk["n"])
    reg = _csv("regime_metrics.csv").iloc[0]
    r["regime"] = {"n_clusters": int(reg["n_clusters"]), "noise": float(reg["noise_fraction"]),
                   "dbcv": float(reg["dbcv"]), "silhouette": float(reg["silhouette"]),
                   "n_points": int(reg["n_points"])}
    an = _csv("anomaly_detection.csv")
    r["anomaly"] = {"recall": float(an["recall"].mean()), "detected": int(an["detected"].sum()),
                    "events": int(an["events"].sum()), "seeds": int(len(an)),
                    "fpr": float(an["false_positive_rate"].mean())}

    # ------------------------------------------------------------------ agent
    saf = _csv("safety.csv").set_index("controller")
    names = {"do_nothing": "do_nothing (deliberately unguarded reference)",
             "static_threshold": "static_threshold", "reactive": "reactive",
             "predictive": "predictive (the contribution)",
             "oracle": "oracle (privileged information)"}
    r["safety_rows"] = [[names[c], str(int(saf.loc[c, "runs_n"])),
                         str(int(saf.loc[c, "unsafe_applied_total"])),
                         str(int(saf.loc[c, "guardrail_events_total"]))]
                        for c in names if c in saf.index]
    guarded = [c for c in saf.index if c != "do_nothing"]
    r["guarded_unsafe"] = int(saf.loc[guarded, "unsafe_applied_total"].sum())
    r["runs_per_controller"] = int(saf["runs_n"].iloc[0])
    r["do_nothing_unsafe"] = int(saf.loc["do_nothing", "unsafe_applied_total"])

    summ = _csv("agent_summary.csv")
    pv = summ.pivot_table(index="scenario", columns="controller", values="risk_h_do_median")
    r["risk_rows"] = [[s] + [f2(pv.loc[s, c]) for c in
                             ("do_nothing", "static_threshold", "reactive", "predictive")]
                      for s in pv.index]
    runs = _csv("agent_runs.csv")
    r["n_scenarios"] = int(runs["scenario"].nunique())
    r["n_seeds"] = int(runs["seed"].nunique())
    r["n_controllers"] = int(runs["controller"].nunique())

    st = _csv("agent_stats.csv")
    key = st[(st["metric"] == "risk_h_do") & (st["controller_a"] == "predictive")]
    r["wins"] = int(((key["p"] < 0.05) & (key["median_diff"] < 0)).sum())
    r["losses"] = int(((key["p"] < 0.05) & (key["median_diff"] > 0)).sum())
    r["comparisons"] = int(len(key))
    ties = key[key["median_diff"] == 0]
    r["tie_n"] = int(len(ties))
    r["tie_scenarios"] = sorted(ties["scenario"].unique().tolist())
    # Are the ties genuinely "nobody had any risk"? Check rather than assert.
    zero = pv[(pv[["do_nothing", "static_threshold", "reactive", "predictive"]] == 0).all(axis=1)]
    r["tie_all_zero"] = bool(set(r["tie_scenarios"]) <= set(zero.index))
    r["n_sig_other"] = r["comparisons"] - r["wins"] - r["tie_n"]
    cost = st[(st["metric"] == "intervention_cost") & (st["controller_a"] == "predictive")
              & (st["controller_b"] == "static_threshold")]
    r["cost_worse"], r["cost_n"] = int((cost["median_diff"] > 0).sum()), int(len(cost))

    # ------------------------------------------------------------------ ablations
    ab = _csv("ablations.csv")
    agg = (ab.groupby("ablation")
             .agg(do=("risk_h_do", "median"), cost=("intervention_cost", "median"),
                  unsafe=("unsafe_applied", "sum"), runs=("seed", "count"))
             .sort_values("do"))
    r["ablation_rows"] = [[("full (shipped configuration)" if k == "full" else k), f2(v["do"]),
                          f2(v["cost"]), str(int(v["unsafe"])), str(int(v["runs"]))]
                         for k, v in agg.iterrows()]
    g = lambda k, c: float(agg.loc[k, c]) if k in agg.index else float("nan")  # noqa: E731
    r["ab_full"], r["ab_nolstm"] = g("full", "do"), g("no_lstm_correction", "do")
    r["ab_nowm"] = g("no_world_model", "do")
    r["ab_noguard_unsafe"] = int(g("no_guardrails", "unsafe"))
    r["ab_full_unsafe"] = int(g("full", "unsafe"))
    r["ab_runs"] = int(agg["runs"].iloc[0])
    return r


@lru_cache(maxsize=1)
def demo(scenario: str = "night_do_crash", seed: int = 1002) -> Dict[str, Any]:
    """Re-run the dashboard demonstration headless, so its numbers are measured, not recalled."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.agent.loop import AgentRun
    from src.common.runtime import set_seed
    from src.eval.run_eval import load_perception

    set_seed(seed)
    run = AgentRun(scenario, controller="predictive", seed=seed,
                   perception=load_perception("federated"))
    res = run.run()
    m = res.metrics()
    min_live = min(s.do for s in res.states)
    min_shadow = min(s.do for s in res.shadow_states)
    return {"scenario": scenario, "seed": seed, "ticks": len(res.states),
            "unsafe": int(m["unsafe_applied"]), "critical_min": m["time_critical_h"] * 60,
            "risk_h": m["time_in_risk_h"], "shadow_risk_h": m["shadow_time_in_risk_h"],
            "min_do_gap": min_live - min_shadow}


@lru_cache(maxsize=1)
def test_count() -> int:
    """How many tests the suite collects - counted, not remembered."""
    import re
    import subprocess
    import sys

    # pytest.ini already adds -q; a second -q would suppress the summary line read here.
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only",
                          "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True,
                         env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    m = re.search(r"(\d+) tests? collected|collected (\d+) items?", out.stdout)
    if not m:
        raise RuntimeError("could not count tests: " + out.stdout[-500:] + out.stderr[-500:])
    return int(m.group(1) or m.group(2))
