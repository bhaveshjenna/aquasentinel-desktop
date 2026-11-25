"""Build report/review2/AquaSentinel-Novelty-and-Base-Papers.pdf. Offline.

    python scripts/make_novelty_pdf.py

Argument order is deliberate and is the order the document reads in: the base papers first,
because they fix what already existed; then the gap between them; then the novelty, which is
only meaningful as a difference from that baseline; then the architecture, which shows where
each novel piece physically sits.

Citation numbers are the numbers used in Chapter 5 of the draft report - the reference list is
imported from ``scripts/make_review2_report.py`` rather than retyped, so the two documents
cannot drift apart. Every quantitative claim is transcribed from RESULTS.md, which is itself
generated from ``outputs/*.csv``.

Printing uses the same headless-Chrome path as ``scripts/make_report_pdf.py``.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "report" / "review2" / "figures"
OUT_PDF = ROOT / "report" / "review2" / "AquaSentinel-Novelty-and-Base-Papers.pdf"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_report = _load("_r2report", Path(__file__).with_name("make_review2_report.py"))
_pdf = _load("_r2pdf", Path(__file__).with_name("make_report_pdf.py"))
REFERENCES = _report.REFERENCES


# --------------------------------------------------------------------------- helpers
def e(text: str) -> str:
    return html.escape(str(text))


def cite(*nums: int) -> str:
    return "[" + ", ".join(str(n) for n in nums) + "]"


def table(headers, rows, cls="t", widths=None):
    colgroup = ""
    if widths:
        colgroup = "<colgroup>" + "".join(f"<col style='width:{w}'>" for w in widths) + "</colgroup>"
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<table class='{cls}'>{colgroup}<thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>")


def figure(name: str, caption: str, cls="fig") -> str:
    path = FIGS / f"{name}.svg"
    if not path.exists():
        return f"<p class='missing'>[missing figure {name} — run scripts/make_review2_figures.py]</p>"
    svg = path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    return f"<figure class='{cls}'>{svg}<figcaption>{caption}</figcaption></figure>"


# =========================================================================== content
CSS = """
@page { size: A4; margin: 1.5cm 1.4cm 1.4cm 1.4cm; }
* { box-sizing: border-box; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 9.6pt; line-height: 1.48;
       color: #14140f; margin: 0; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 2mm; letter-spacing: -0.01em; }
h2 { font-size: 13pt; margin: 9mm 0 2.5mm; padding-bottom: 1.2mm;
     border-bottom: 1.6pt solid #14140f; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 5.5mm 0 1.8mm; page-break-after: avoid; }
h4 { font-size: 9.8pt; margin: 4mm 0 1.2mm; font-style: italic; font-weight: normal;
     color: #52514e; page-break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: justify; hyphens: auto; }
.lead { font-size: 10.4pt; color: #33322e; }
.sub { color: #6a6862; font-size: 9pt; margin: 0 0 5mm; }
.rule { border: 0; border-top: 0.6pt solid #d8d7cf; margin: 6mm 0; }
code, .mono { font-family: "Consolas", "DejaVu Sans Mono", monospace; font-size: 8.4pt; }
/* Tables flow across pages - forcing a long one to stay whole just pushes it to the next
   page and leaves half the previous one blank. Rows stay atomic and the header repeats. */
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm; font-size: 8.2pt;
        line-height: 1.34; page-break-inside: auto; table-layout: fixed; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
th { background: #eceae2; text-align: left; font-weight: bold; padding: 1.5mm 1.8mm;
     border: 0.5pt solid #b9b7ad; vertical-align: bottom; }
td { padding: 1.5mm 1.8mm; border: 0.5pt solid #cfcdc3; vertical-align: top; }
/* Fixed layout clips long unbroken tokens - file paths are exactly that, so let them wrap.
   Ordinary prose gets the gentle rule; only the monospace paths get the aggressive one. */
td, th { overflow-wrap: break-word; }
td .mono, td code { overflow-wrap: anywhere; word-break: break-all; }
tbody tr:nth-child(even) td { background: #f7f6f1; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.id { font-weight: bold; white-space: nowrap; }
figure { margin: 4mm 0 5mm; page-break-inside: avoid; text-align: center; }
figure svg { width: 100%; height: auto; }
figcaption { font-size: 8.2pt; color: #52514e; margin-top: 2mm; text-align: left;
             line-height: 1.4; }
.page { page-break-before: always; }
.claim { border-left: 2.4pt solid #eb6834; padding: 0 0 0 3.5mm; margin: 4mm 0 4mm; }
.claim h3 { margin-top: 0; }
.tag { display: inline-block; font-size: 7.6pt; letter-spacing: 0.06em; text-transform: uppercase;
       color: #8a8880; margin-right: 2mm; font-family: "Consolas", monospace; }
.warn { border-left: 2.4pt solid #d03b3b; padding: 0 0 0 3.5mm; margin: 4mm 0; }
.note { background: #f4f3ef; border: 0.5pt solid #d8d7cf; padding: 2.6mm 3mm; margin: 3.5mm 0;
        font-size: 9pt; }
ol.refs { padding-left: 0; list-style: none; counter-reset: none; font-size: 8.4pt;
          line-height: 1.42; }
ol.refs li { margin: 0 0 1.6mm; padding-left: 9mm; text-indent: -9mm; }
ul.tight { margin: 0 0 2.6mm; padding-left: 5mm; }
ul.tight li { margin-bottom: 1.1mm; }
.foot { margin-top: 8mm; font-size: 8.2pt; color: #6a6862; border-top: 0.6pt solid #d8d7cf;
        padding-top: 2.5mm; }
"""

# ---- base papers: the ones whose method is actually implemented -----------------
BASE = [
    (24, "McMahan et al. (2017) — Federated Averaging",
     "The aggregation rule itself: clients train locally for E epochs, the server averages "
     "weights in proportion to sample count, raw data never moves.",
     "One <em>real monitoring site</em> is one client, so the non-IID structure is inherited "
     "rather than manufactured. Run twice over two different client populations — 42 real "
     "sites and 8 simulated ponds — and scored on a control task, not only on held-out error.",
     "src/federated/simulate.py"),
    (18, "Hochreiter and Schmidhuber (1997) — LSTM",
     "The recurrent architecture used for both the forecaster (M1) and the world model's "
     "encoder (M2).",
     "Predicts per-channel <em>deltas</em> divided by fixed, unfitted scales, and consumes a "
     "value-plus-mask tensor so clients carrying different probes remain comparable.",
     "src/perception/forecaster.py"),
    (49, "Ha and Schmidhuber (2018) — world models",
     "Action-conditioned latent dynamics as the substrate a planner rolls forward.",
     "The encoder is warm-started from the federated forecaster, so the representation the "
     "decision is computed on comes from real data while only the action head is learned in "
     "simulation.",
     "src/perception/world_model.py"),
    (46, "Camacho and Bordons (2007) — model predictive control",
     "Roll each candidate action forward through a model, score the trajectories, apply the "
     "first action of the best one.",
     "The model rolled forward is the learned M2 rather than a differential equation, and the "
     "candidate set is a curated 14 of the 108-action cross product so the deliberation stays "
     "readable.",
     "src/agent/policy.py — PredictivePlanner"),
    (48, "Wabersich and Zeilinger (2021) — predictive safety filter",
     "The central design decision: hard constraints live <em>outside</em> the learned policy "
     "and filter its output, rather than being folded into its cost function.",
     "Implemented as a five-stage stack that runs on every sensor tick while the planner "
     "decides every 15 minutes, and the counterfactual — how many unsafe actions the same "
     "planner issues with the filter removed — is measured rather than argued.",
     "src/agent/guardrails.py"),
    (47, "Hewing et al. (2020) — learning-based MPC",
     "The failure analysis that motivates the above: an optimiser treats its model as exact "
     "and will exploit wherever the model is wrong.",
     "Turned into a measurement by running the identical ablation against a learned planner "
     "and a hand-written one on the same grid and seeds.",
     "outputs/ablations.csv — no_guardrails"),
    (36, "Campello, Moulavi and Sander (2013) — HDBSCAN",
     "Density-based clustering that may decline to assign a point, which makes the "
     "unassigned label usable as a novelty flag.",
     "Run on dimensionless derived features so a 12 °C river window and a 28 °C pond window "
     "can occupy one space; the outcome was negative and is reported as such.",
     "src/perception/regimes.py"),
    (38, "Moulavi et al. (2014) — DBCV",
     "The density-based validity index appropriate to non-globular clusters.",
     "This is the instrument that produced the project's clearest negative result: DBCV 0.000 "
     "beside a healthy silhouette of 0.525, across a nine-configuration sweep.",
     "outputs/regime_sweep.csv"),
    (42, "Ke et al. (2017) — LightGBM",
     "Gradient-boosted trees over tabular features, cheap enough to score on every tick.",
     "Consumes current channels plus forecast summaries, so the risk head is forward-looking; "
     "its target is stated to be a rule rather than learned chemistry.",
     "src/perception/risk.py"),
    (43, "Lundberg and Lee (2017) — SHAP",
     "Additive per-prediction attribution.",
     "Wired to the operator panel and to the agent's own stated rationale, so a recommendation "
     "arrives with the channels that drove it.",
     "src/perception/explain.py"),
    (2, "Boyd and Tucker (1998) — pond water quality",
     "The pond oxygen budget, the diel cycle, and the safe operating bands for DO, pH, "
     "ammonia and nitrite in warm-water culture.",
     "Encoded as the twin's coefficients and risk bands in configs/twin.yaml, and labelled "
     "throughout as calibratable defaults rather than measured facts.",
     "src/twin/dynamics.py · configs/twin.yaml"),
    (5, "Emerson et al. (1975) — aqueous ammonia equilibrium",
     "The temperature-and-pH dependence of the un-ionised ammonia fraction.",
     "Makes TAN dangerous only in combination with a rising pH, which is what gives the "
     "overfeeding and pH-swing scenarios their coupling.",
     "src/twin/dynamics.py"),
    (6, "Benson and Krause (1984) — oxygen saturation",
     "The saturation relation for dissolved oxygen in freshwater.",
     "Used to express DO as a percentage of saturation, which is the transformation that lets "
     "river and pond windows share one regime feature space.",
     "src/perception/regimes.py"),
    (57, "Wilcoxon (1945) — signed-rank test",
     "The paired non-parametric test used for every controller comparison.",
     "Paired on seed, because the same seed gives every controller the identical perturbation "
     "schedule and noise realisation; the harness refuses to test below six seeds rather than "
     "quietly printing a mean.",
     "src/eval/metrics.py"),
    (34, "Abadi et al. (2016) and Mironov (2017) — DP-SGD, RDP",
     "Per-update clipping plus Gaussian noise, with Rényi accounting for the privacy budget.",
     "Implemented as an opt-in path; ε is reported from an accountant when one is installed "
     "and as UNACCOUNTED otherwise, never guessed.",
     "src/federated/simulate.py — --dp"),
]

# ---- closest prior work: what the novelty is measured against ------------------
CLOSEST = [
    (13, 14, "Liu et al. (2013); Ta and Wei (2018) — DO prediction for ponds and RAS",
     "Single site, centrally trained, evaluated as forecast error.",
     "Neither carries the prediction through to an actuator, so the value of the forecast to "
     "control is assumed rather than measured. This project measures it — and finds it "
     "negative."),
    (16, 17, "Barzegar et al. (2020); Hu et al. (2019) — CNN-LSTM and LSTM water quality",
     "The strongest reported architecture family; absolute-value targets, one operator's data.",
     "Absolute targets do not transfer across water bodies, and there is no privacy-preserving "
     "training path. The delta parameterisation and the federated client boundary here exist "
     "for exactly those two reasons."),
    (32, 33, "Durrant et al. (2022); Friha et al. (2022) — cross-silo FL in agri-food",
     "The closest sector precedent: farms as silos unwilling to pool raw records.",
     "Yield prediction and intrusion detection respectively — neither closes a control loop, "
     "so neither asks whether federated weights are a safe basis for an action."),
    (28, 30, "Zhao et al. (2018); Nguyen et al. (2021) — non-IID FL, FL for IoT",
     "Quantify and survey the cost of client heterogeneity.",
     "Heterogeneity is produced by partitioning a benchmark by label. Here it is inherited "
     "from 42 real stations that differ in climate, season, catchment and even in which probes "
     "they carry."),
    (52, 47, "García and Fernández (2015); Hewing et al. (2020) — safe RL and learning-based MPC",
     "Taxonomy separating constrained optimisation from an external safety layer, and the "
     "theory of why a learned model is dangerous inside an optimiser.",
     "Both argue the case; neither reports the counterfactual on one system — how many unsafe "
     "commands the same planner issues once the filter is removed. N3 is that measurement."),
]

# ---- novelty claims -------------------------------------------------------------
NOVELTY = [
    ("N1", "Two-stage federation with a cross-domain encoder transfer",
     "The forecaster (M1) is federated over 42 <em>real</em> monitoring sites. The "
     "action-conditioned world model (M2) then warm-starts its encoder from that federated "
     "model and federates its action head over 8 <em>simulated</em> ponds. The split is forced "
     "by the data and not chosen for convenience: rivers contain no aerator, no feeding and no "
     "liming, so they contain zero (action &rarr; response) pairs, and no amount of federation "
     "over them can teach a model what an aerator does. The consequence is that the "
     "representation the control decision is computed on is federated over real data, while "
     "only the part real data cannot supply is learned in simulation.",
     "On held-out simulated ponds the federated world model reaches 1.126 mg/L DO error "
     "against 1.292 for independent-local and 1.075 for the pooled reference — the same "
     "ordering as the real-site forecaster, now for the model that chooses the action. "
     "<span class='mono'>outputs/world_model_comparison.csv</span>"),

    ("N2", "Federated learning scored on the control task, not only on held-out error — "
           "and the ordering reverses",
     "The three weight sources are swapped into the identical controller and the resulting "
     "<em>control</em> outcome is measured, which converts “does federation lower RMSE” into "
     "“does federation improve the decision”. This is the claim the project was built to test, "
     "and the answer does not flatter it.",
     "On forecast RMSE the order is central 0.338 &le; federated 0.342 &lt; independent-local "
     "0.421 mg/L. On dissolved-oxygen time-out-of-band under the shipped controller the order "
     "<em>reverses</em>: independent-local 8.83 h &lt; federated 10.00 h &lt; central 11.75 h. "
     "Better held-out forecast accuracy did not produce better control on this system. "
     "<span class='mono'>outputs/federated_comparison.csv</span>, "
     "<span class='mono'>outputs/ablations.csv</span>"),

    ("N3", "The safety layer's contribution measured as a counterfactual, not asserted",
     "The literature argues that constraints are necessary and that an optimiser will exploit "
     "wherever its learned model is wrong " + cite(47, 48, 52) + ". Very few papers report the "
     "counterfactual that would demonstrate it: how many unsafe commands the <em>same</em> "
     "planner issues once the safety layer is removed, on the same grid and the same seeds. "
     "Here that cell is part of the standing ablation grid, so the claim is re-measured on "
     "every evaluation run rather than taken on trust.",
     "With the guardrails removed, the shipped learned planner applies <strong>1058</strong> "
     "unsafe actions across the ablation grid. With them in place it applies <strong>0</strong>, "
     "as does every other guardrailed controller, across 8 scenarios &times; 5 controllers "
     "&times; 10 seeds — 80 runs each. "
     "<span class='mono'>outputs/ablations.csv</span>, "
     "<span class='mono'>outputs/safety.csv</span>"),

    ("N4", "A negative clustering result, reported with the index that exposes it",
     "The intended deliverable was a taxonomy of pond operating regimes. Swept across cluster "
     "sizes, sample counts, both selection methods, a PCA variant and both pooled and "
     "twin-only data, the answer is always one dominant density mode. The contribution is the "
     "demonstration that the usual metric misleads here: silhouette flatters a single blob cut "
     "in two, and DBCV " + cite(38) + " does not.",
     "DBCV <strong>0.000</strong> beside silhouette <strong>0.525</strong> over 40,086 "
     "windows, with 21.8 % of windows unassigned. The model is therefore repurposed as a "
     "novelty detector — event recall 0.80 on injected sensor faults at a 0.113 false-positive "
     "rate — and nothing downstream depends on a cluster id. "
     "<span class='mono'>outputs/regime_sweep.csv</span>, "
     "<span class='mono'>outputs/anomaly_detection.csv</span>"),

    ("N5", "Counterfactual attribution built into the system rather than inferred",
     "A shadow pond is forked at t = 0 with the same seed, the same perturbation schedule and "
     "the same noise realisation, running with the agent disabled. Every gap between the two "
     "trajectories is therefore caused by an action and by nothing else, and the operator sees "
     "that gap live rather than being asked to trust a summary statistic.",
     "The strong form is asserted by test: with no actions the two trajectories are "
     "bit-for-bit identical "
     "(<span class='mono'>tests/test_agent.py::test_do_nothing_matches_its_own_shadow</span>). "
     "In the demonstration run — night_do_crash, seed 1002, 432/432 ticks — the agent's pond "
     "spends 16.2 h in risk against 28.1 h for its own shadow."),

    ("N6", "Domain-gap-aware transfer, including a calibration the project declines to apply",
     "Three design decisions exist only because the training data is rivers and the target is "
     "a pond. Targets are per-channel deltas divided by <em>fixed</em> scales — fixed rather "
     "than fitted, because fitting them would leak client statistics across the federated "
     "boundary. Regime features are dimensionless, so a river at 9 mg/L and 12 °C and a pond "
     "at 6.5 mg/L and 28 °C can both be “85 % saturated”. And the twin calibration transfers "
     "process-noise magnitude, baseline turbidity and relaxation pH, but <em>refuses</em> the "
     "diurnal cycle.",
     "The measured median daily temperature swing across the 42 sites is 0.51 °C — a deep "
     "flowing river's swing, not a 1.5 m earthen pond's 2–4 °C. Copying it would flatten the "
     "pond into an isothermal tank and delete the dawn oxygen sag the project is about. The "
     "fitted value is recorded in "
     "<span class='mono'>configs/twin.yaml</span> under "
     "<span class='mono'>calibration.fitted_diurnal_not_applied</span> so a reader can see "
     "exactly what was declined."),

    ("N7", "Both sides of the ledger reported for every claim",
     "Risk reduction is reported together with what it cost, on the same grid and with the "
     "same test. This is a reporting discipline rather than a technique, but it is the reason "
     "the results above can be read at face value.",
     "The agent significantly lowers DO time-out-of-band in <strong>18 of 24</strong> "
     "scenario &times; baseline comparisons (the other 6 are exact ties where no controller "
     "accrues any risk) — and is significantly <em>more expensive</em> than the static "
     "threshold in <strong>7 of 8</strong> scenarios, including ones where it achieves the "
     "same zero risk the thermostat had already achieved. Every comparison is a paired "
     "Wilcoxon signed-rank " + cite(57) + " with p and n reported."),
]

NOT_NOVEL = [
    ("Federated averaging itself", "FedAvg is used as published " + cite(24) +
     "; no aggregation rule is proposed."),
    ("LSTM forecasting of water quality", "Well established " + cite(14, 15, 16, 17) +
     "; the architecture here is deliberately small and ordinary."),
    ("MPC, SHAP, HDBSCAN, LightGBM, digital twins",
     "All used as published " + cite(36, 42, 43, 46, 55) + "."),
    ("Any real-pond validity", "The pond is a simulation and the 42 sites are rivers used as "
     "an acknowledged proxy. Every performance claim is relative — agent against baselines "
     "inside the twin, federated against local and central on held-out real sites."),
    ("Learned water chemistry", "The risk head's target is a band rule from "
     "<span class='mono'>configs/twin.yaml</span>. It has learned to anticipate a known rule "
     "from an imperfect forecast, which is a smaller claim and the correct one."),
    ("A tuned or optimal cost trade-off", "<span class='mono'>cost_weights.intervention</span> "
     "is exposed as a dial. Nothing here establishes where a real farm should set it."),
]

# ---- the model inventory --------------------------------------------------------
MODELS = [
    ("M1", "LSTM Forecaster", "Multi-step multivariate forecast of the 6 sensor channels: "
     "24 steps ahead from a 36-step window (6 h from 9 h on the 15-min real feed; "
     "2 h from 3 h in the 5-min twin)",
     "input 36&times;12 (6 values + 6 mask) &rarr; LSTM(hidden 64, 1 layer) &rarr; "
     "Linear(64 &rarr; 24&times;6); targets are per-channel &Delta; &divide; fixed scale",
     "FedAvg, 12 rounds &times; 2 local epochs, 42 real sites; also trained "
     "independent-local (24 epochs) and central (24 epochs)",
     "42 real monitoring sites, 729,542 rows",
     "forecaster_{federated, central, local&times;32}.pt"),
    ("M2", "Action-conditioned World Model", "Predicts the trajectory that follows a "
     "<em>candidate action</em> — the model the planner scores",
     "encoder LSTM(64) warm-started from federated M1, + 8 action features &rarr; "
     "MLP(72 &rarr; 64 &rarr; 24&times;6), clipped to physical bounds",
     "FedAvg, 12 rounds, 8 heterogeneous simulated ponds (+2 held out)",
     "twin rollouts under randomised action sequences",
     "world_model_{federated, local&times;8}.pt"),
    ("M3", "HDBSCAN novelty detector", "Flags novel operating conditions; the unassigned "
     "label is the anomaly flag",
     "16-d dimensionless features (4 derived channels &times; 4 window statistics); "
     "min_cluster_size 40, min_samples 10, eom, euclidean",
     "unsupervised — no label is consumed anywhere",
     "pooled real + twin windows (40,086)",
     "regimes.pkl"),
    ("M4", "LightGBM risk classifier", "Maps current channels + forecast summaries to an "
     "ordered risk band",
     "220 trees, 31 leaves, max depth 6, lr 0.06, min_child_samples 25 &rarr; "
     "{safe, warning, critical}",
     "supervised on a <em>rule-derived</em> target: the band configs/twin.yaml assigns to the "
     "worst point of the true next horizon",
     "twin rollouts, held-out seed",
     "risk_model.pkl"),
    ("M5", "SHAP TreeExplainer", "Per-tick attribution of M4's score to the input channels",
     "TreeExplainer over M4; no separate training",
     "n/a — derived from M4", "n/a", "computed live"),
    ("M6", "InternalModel", "The agent's own belief about the pond — deliberately imperfect",
     "a PondTwin instance from nominal configuration, nudged toward observations by a "
     "complementary filter, &alpha; = 0.35",
     "not trained; it never reads the real twin's latent state or the scenario schedule",
     "n/a", "n/a"),
    ("M7", "MPC Planner", "Chooses the action: rolls candidates through M2 and takes the "
     "minimum-cost trajectory",
     "14 curated candidates of the 108-action cross product &times; 24 rollout steps; "
     "cost = 1.0&middot;risk + 3.0&middot;critical + 0.06&middot;intervention + "
     "0.20&middot;stability + 0.35&middot;feed_loss",
     "not trained — it is an optimiser over M2 and M6",
     "n/a", "n/a"),
]

EVIDENCE = [
    ("N1", "World-model DO error on held-out ponds, federated vs. independent-local",
     "1.126 vs 1.292 mg/L", "outputs/world_model_comparison.csv"),
    ("N2", "Forecast RMSE ordering (held-out real sites)",
     "central 0.338 &le; fed 0.342 &lt; local 0.421", "outputs/federated_comparison.csv"),
    ("N2", "Control ordering under the same weights (median DO time-out-of-band)",
     "local 8.83 &lt; fed 10.00 &lt; central 11.75 h", "outputs/ablations.csv"),
    ("N3", "Unsafe actions applied, shipped planner unguarded", "1058", "outputs/ablations.csv"),
    ("N3", "Unsafe actions applied, every guardrailed controller", "0, over 80 runs each",
     "outputs/safety.csv"),
    ("N4", "Regime structure — DBCV against silhouette", "0.000 vs 0.525, n = 40,086",
     "outputs/regime_sweep.csv"),
    ("N4", "Novelty detector, injected sensor faults",
     "event recall 0.80, FPR 0.113", "outputs/anomaly_detection.csv"),
    ("N5", "Shadow pond identity with no actions", "bit-for-bit identical",
     "tests/test_agent.py"),
    ("N6", "Measured diurnal temperature swing across the 42 real sites",
     "0.51 °C — fitted, not applied", "configs/twin.yaml"),
    ("N7", "Agent vs. baselines on DO time-out-of-band", "18 of 24 significant at p &lt; 0.05",
     "outputs/agent_stats.csv"),
    ("N7", "Agent vs. static threshold on intervention cost",
     "significantly more expensive in 7 of 8", "outputs/agent_stats.csv"),
]

CITED = [2, 5, 6, 13, 14, 15, 16, 17, 18, 24, 28, 30, 32, 33, 34, 35, 36, 38, 42, 43, 46,
         47, 48, 49, 52, 55, 57]


# =========================================================================== build
def build_html() -> str:
    o: list[str] = []
    a = o.append

    a("<h1>AquaSentinel — base papers, research gap and novelty</h1>")
    a("<p class='sub'>BCSE497J Project-I &middot; Review-2 supplement &middot; "
      "Sanjeev Karnatapu (23BAI0003), Nishant Ray (23BAI0065), Soham Swarup Dutta "
      "(23BCE0549) &middot; Guide: Dr. Sivaranjani A, SCOPE</p>")

    a("<p class='lead'>This document answers three questions in the order they have to be "
      "answered. <strong>&sect;1</strong> sets out the papers the project is actually built "
      "on — not everything that was surveyed, but the works whose method is implemented in "
      "the code. <strong>&sect;2</strong> states where those works stop. <strong>&sect;3</strong> "
      "states what is new here, with the evidence for each claim and an explicit list of what "
      "is <em>not</em> new. <strong>&sect;4</strong> then shows the full pipeline and every "
      "model in it, so each novel piece can be located physically in the system.</p>")
    a("<p>Citation numbers are the numbers used in Chapter 5 of the draft report; the "
      "reference list at the end is the cited subset, imported from the same source so the "
      "two documents cannot drift apart. Every quantitative claim is transcribed from "
      "<span class='mono'>RESULTS.md</span>, which is generated from "
      "<span class='mono'>outputs/*.csv</span> — none of the numbers below were typed by "
      "hand into a results table.</p>")

    # ---------------------------------------------------------------- 1. base papers
    a("<h2>1. Base papers</h2>")
    a("<h3>1.1 The works whose method is implemented</h3>")
    a("<p>These fifteen entries are load-bearing: remove any one and a specific module stops "
      "working or stops being justified. For each, the table separates what the project takes "
      "unchanged from what it changes — which is where the contribution in &sect;3 begins.</p>")
    a(table(
        ["Ref.", "Base paper", "What the project takes", "What the project changes or adds",
         "Where it lives"],
        [(f"<span class='id'>[{n}]</span>", f"<strong>{t}</strong>", takes, changes,
          f"<span class='mono'>{e(where)}</span>")
         for n, t, takes, changes, where in BASE],
        widths=["7%", "22%", "26%", "31%", "14%"]))

    a("<h3>1.2 The closest prior work, and why the gap survives it</h3>")
    a("<p>These are the papers the novelty in &sect;3 has to be measured against — the ones "
      "doing the most similar thing. They are listed with what they establish and, plainly, "
      "the reason each stops short of the question this project asks.</p>")
    a(table(
        ["Ref.", "Closest prior work", "What it establishes", "Why the gap survives it"],
        [(f"<span class='id'>[{x}, {y}]</span>", f"<strong>{t}</strong>", est, gap)
         for x, y, t, est, gap in CLOSEST],
        widths=["8%", "24%", "31%", "37%"]))

    # ---------------------------------------------------------------- 2. gap
    a("<h2>2. The gap between them</h2>")
    a("<p>Five strands — aquaculture water-quality science, data-driven forecasting, federated "
      "learning, unsupervised structure and explanation, and safe learning-based control — "
      "have developed largely independently. Reading them together, five gaps stand out, and "
      "each one shapes a design decision rather than merely motivating the project.</p>")
    gaps = [
        ("G1", "Prediction is never carried through to action",
         "Water-quality forecasting papers stop at an error metric " + cite(13, 14, 15, 16, 17) +
         ". None closes the loop onto an actuator and then re-measures, so the value of the "
         "forecast <em>to the controller</em> is assumed rather than tested.",
         "Addressed by N2, which measures it — and finds it negative."),
        ("G2", "Federated learning has not been evaluated on an actuation task",
         "Federated studies report held-out accuracy " + cite(24, 28, 30, 32) + ". Whether a "
         "federated model is a safe basis for a decision that opens a valve is a different "
         "question and is unaddressed.",
         "Addressed by N1 and N2."),
        ("G3", "Client heterogeneity is manufactured, not inherited",
         "Non-IID structure is usually produced by partitioning a benchmark by label " +
         cite(28) + ". Real sites differ in climate, season, catchment and even in which "
         "probes they carry, and that structure cannot be simulated faithfully.",
         "Addressed by the client construction: one real station = one client."),
        ("G4", "Safety is argued, not measured",
         "Learning-based control papers argue that constraints are necessary " + cite(47, 48, 52) +
         ". Very few report the counterfactual — how many unsafe commands the same planner "
         "would have issued with the safety layer removed — which is the only evidence that "
         "the layer is load-bearing.",
         "Addressed by N3."),
        ("G5", "Scripted quality labels make results circular",
         "Most public water-quality datasets carry index labels computed from thresholds. A "
         "model trained on them and scored against them has learned the threshold, and the "
         "reported accuracy says nothing about chemistry.",
         "Addressed by consuming only continuous channels and unsupervised structure, and by "
         "stating that the risk head's target is a rule."),
    ]
    a(table(["", "Gap", "Why it is still open", "Where this project answers it"],
            [(f"<span class='id'>{g}</span>", f"<strong>{t}</strong>", why, ans)
             for g, t, why, ans in gaps],
            widths=["5%", "22%", "45%", "28%"]))

    # ---------------------------------------------------------------- 3. novelty
    a("<h2 class='page'>3. Novelty</h2>")
    a("<p>Seven claims. Each is stated as a difference from the base papers in &sect;1, and "
      "each is followed by the evidence that supports it — including where that evidence is "
      "unflattering. A claim with no file behind it is not made.</p>")
    for tag, title, what, evidence in NOVELTY:
        a("<div class='claim'>")
        a(f"<h3><span class='tag'>{tag}</span>{title}</h3>")
        a(f"<p>{what}</p>")
        a(f"<p><strong>Evidence.</strong> {evidence}</p>")
        a("</div>")

    a("<h3>3.8 What is deliberately not claimed as novel</h3>")
    a("<p>A novelty section is only credible with this list beside it.</p>")
    a(table(["Not claimed", "Why"],
            [(f"<strong>{t}</strong>", why) for t, why in NOT_NOVEL],
            widths=["30%", "70%"]))

    a("<h3>3.9 One claim that is not yet backed by the outputs</h3>")
    a("<div class='warn'><p><strong>Learned planner versus physics planner, both unguarded.</strong> "
      "<span class='mono'>README.md</span> and <span class='mono'>STATUS.md</span> state that "
      "the learned planner applies 1058 unsafe actions unguarded against 423 for the "
      "hand-written one — 2.5&times; more dangerous. That would be the sharpest form of N3, "
      "but it is <em>not currently reproducible</em> from "
      "<span class='mono'>outputs/ablations.csv</span>: the standing grid in "
      "<span class='mono'>src/eval/run_eval.py</span> contains "
      "<span class='mono'>no_guardrails</span> (guardrails off, world model on) and "
      "<span class='mono'>no_world_model</span> (guardrails on, world model off), but no cell "
      "with both off. The 423 is prose, not a generated figure, so it is left out of &sect;3 "
      "and out of the evidence index. Closing this is a one-line addition to the grid "
      "— a <span class='mono'>{guardrails: False, world_model: False}</span> variant — "
      "followed by a re-run of <span class='mono'>python -m src.eval.run_eval</span>.</p></div>")

    a("<div class='warn'><p><strong>The single most important caveat.</strong> The pond is a "
      "simulation whose coefficients are calibratable defaults taken from published "
      "aquaculture guidance " + cite(2, 5, 6) + ", not measurements from a physical pond; and "
      "the 42 real sites are rivers, streams and lakes used as an explicitly acknowledged "
      "proxy for pond dynamics. Aeration, feeding and stocking density have no analogue in a "
      "river. Every result in &sect;3 is therefore a <em>relative</em> comparison inside a "
      "stated setting, and none of them is a real-world guarantee.</p></div>")

    # ---------------------------------------------------------------- 4. architecture
    a("<h2>4. Architecture — the full pipeline and every model in it</h2>")
    a("<p>The system is drawn as two diagrams because one would have to be shrunk past "
      "legibility to fit a page. Figure 1 is everything that is <em>trained</em>: acquisition, "
      "windowing, and the two separate federations. Figure 2 is everything that <em>runs</em>: "
      "the perception heads, the closed control loop, the safety layer and the outputs. Both "
      "are drawn at their final print size, and both are also written as SVG in "
      "<span class='mono'>report/review2/figures/</span> if you want to zoom in.</p>")

    a(figure("fig8_pipeline_training",
             "<strong>Figure 1. Training and federation pipeline.</strong> Band A is the only "
             "part of the system that touches the network. Band C federates the forecaster "
             "M1 across 42 real monitoring sites under three regimes over the identical client "
             "set. Band D federates the action-conditioned world model M2 across 8 "
             "heterogeneous simulated ponds, with its encoder warm-started from the federated "
             "M1 — the dashed edge on the right. The reason for the second federation is the "
             "red box: rivers contain no actuator, so they contain no action-response pairs "
             "at all."))

    a(figure("fig9_pipeline_inference",
             "<strong>Figure 2. Inference, decision and safety pipeline.</strong> The planner "
             "M7 scores 14 candidate actions by rolling each through the learned world model "
             "M2; whatever it proposes passes through the five-stage guardrail stack before "
             "it can reach an actuator. The planner re-decides every 15 minutes while the "
             "guardrails run on every sensor tick. The shadow twin is forked at t = 0 with the "
             "same seed and the same noise draw, so the gap between the two ponds is caused by "
             "actions and by nothing else."))

    a("<h3>4.1 The seven models, in full</h3>")
    a("<p>M1 and M2 are the two federated networks; M3 and M4 are the perception heads; M5 is "
      "an explainer over M4; M6 and M7 are the agent's belief and its optimiser, and are not "
      "trained at all. Naming them apart matters, because “the model” in this system is "
      "seven different things with different training data and different claims attached.</p>")
    a(table(
        ["", "Model", "Job", "Architecture", "How it is trained", "Data", "Artefact"],
        [(f"<span class='id'>{m}</span>", f"<strong>{n}</strong>", job, arch, train, data,
          f"<span class='mono'>{art}</span>")
         for m, n, job, arch, train, data, art in MODELS],
        widths=["4%", "12%", "15%", "21%", "19%", "12%", "17%"]))

    a("<h3>4.2 Where each novelty claim sits in the architecture</h3>")
    a(table(["Claim", "Location in the figures"],
            [("<span class='id'>N1</span>",
              "Figure 1, bands C and D, and the dashed encoder warm-start edge between them"),
             ("<span class='id'>N2</span>",
              "Figure 1 band C produces the three weight sets; Figure 2 loads one of them at "
              "start-up, so the control metric changes with the weight source"),
             ("<span class='id'>N3</span>",
              "Figure 2 — the guardrail stack, and the fact that it sits between M7 and the "
              "actuators rather than inside M7's cost function"),
             ("<span class='id'>N4</span>", "Figure 2 band E, M3"),
             ("<span class='id'>N5</span>", "Figure 2 — the shadow twin and the fork edge"),
             ("<span class='id'>N6</span>",
              "Figure 1 band B — the fixed delta scales — and the dimensionless regime "
              "features feeding M3"),
             ("<span class='id'>N7</span>", "Figure 2 band G — the evaluation harness")],
            widths=["10%", "90%"]))

    # ---------------------------------------------------------------- 5. evidence
    a("<h2>5. Evidence index</h2>")
    a("<p>Every claim in &sect;3, the number that supports it, and the file that number comes "
      "from. The files are regenerated by <span class='mono'>python -m src.eval.run_eval</span> "
      "and rendered into <span class='mono'>RESULTS.md</span> by "
      "<span class='mono'>python scripts/build_report.py</span>.</p>")
    a(table(["Claim", "What is measured", "Value", "Source file"],
            [(f"<span class='id'>{c}</span>", what, f"<strong>{val}</strong>",
              f"<span class='mono'>{src}</span>")
             for c, what, val, src in EVIDENCE],
            widths=["8%", "44%", "24%", "24%"]))

    a("<div class='note'><strong>Reproducing all of it.</strong> "
      "<span class='mono'>python scripts/collect_data.py</span> (once, online) &rarr; "
      "<span class='mono'>calibrate_twin.py</span> &rarr; "
      "<span class='mono'>pretrain_federated.py</span> &rarr; "
      "<span class='mono'>pretrain_world_model.py</span> &rarr; "
      "<span class='mono'>python -m src.eval.run_eval</span> &rarr; "
      "<span class='mono'>build_report.py</span>. "
      "The test suite is <span class='mono'>python -m pytest tests/ -q</span> — 131 tests, "
      "all passing.</div>")

    # ---------------------------------------------------------------- references
    a("<h2>References</h2>")
    a("<p class='sub'>Numbering follows Chapter 5 of the draft report; this is the subset "
      "cited above.</p>")
    a("<ol class='refs'>")
    for n in CITED:
        a(f"<li>[{n}]&nbsp;&nbsp;{e(REFERENCES[n - 1])}</li>")
    a("</ol>")

    a("<p class='foot'>Generated by <span class='mono'>scripts/make_novelty_pdf.py</span>. "
      "Figures by <span class='mono'>scripts/make_review2_figures.py</span>. If any number "
      "here disagrees with <span class='mono'>RESULTS.md</span>, RESULTS.md is right and this "
      "document is stale.</p>")

    body = "\n".join(o)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>AquaSentinel — Novelty and Base Papers</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PDF))
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_pdf.with_suffix(".html")
    html_path.write_text(build_html(), encoding="utf-8")
    print(f"HTML  -> {html_path}  ({html_path.stat().st_size / 1024:.0f} KB)")

    out_pdf.unlink(missing_ok=True)
    proc = _pdf._print_pdf(html_path, out_pdf)
    if not out_pdf.exists():
        print("PDF was not produced.")
        if proc is not None:
            print((proc.stderr or proc.stdout or "")[-1200:])
        print(f"The HTML is at {html_path} - print it to PDF from any browser.")
        return 1
    print(f"PDF   -> {out_pdf}  ({out_pdf.stat().st_size / 1024:.0f} KB)")
    if not args.keep_html:
        html_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
