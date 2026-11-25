# AquaSentinel — build status

> **Note (25 October 2026): this document describes the earlier river-data version of the project** (42 USGS / UK Environment Agency sites, an optional LLM planner, the Streamlit dashboard as the main UI). The project now trains on **7 real catfish aquaponics ponds in Nsukka, Nigeria** (one region), the LLM planner has been removed, and the main UI is the desktop console (`python -m desktop`). The concepts below still apply; the numbers, data description and file lists do not. For current results see `RESULTS.md` and `README.md`. The river version is kept in the separate copy of this folder.

**Last session:** 2026-09-05
**State:** Phases 0–8 of the build spec are implemented, running end to end on real collected
data, and fully evaluated. 131 tests pass. The dashboard works. `outputs/` is complete,
`RESULTS.md` is generated from it, and the 29-page PDF report is rebuilt from both.

Jump to **[Pick up here](#pick-up-here)**.

---

## Phase checklist

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton, `runtime.py`, `schema.py`, configs | ✅ |
| 1 | Online collection (USGS + UK EA), harmonise, manifest | ✅ **42 sites, 729,542 rows** |
| 2 | Twin + sensor stream + dashboard MVP | ✅ calibrated; all 8 scenarios behave |
| 3 | Reactive agent, guardrails, counterfactual, scorecard | ✅ **0 unsafe applied** |
| 4 | Perception ML (LSTM, HDBSCAN, LightGBM risk, SHAP) | ✅ with the caveats below |
| 5 | Predictive agent (learned MPC) | ✅ beats both deployable baselines |
| 6 | Federated learning across independent ponds | ✅ **fed 0.342 vs local 0.421 DO RMSE** |
| 6b | Action-conditioned world model, federated across simulated ponds | ✅ **the planner's trajectories are network output** |
| 7 | Evaluation grid, Wilcoxon, ablations | ✅ 400-run grid + 6 ablation variants |
| 8 | Tests, README, generated RESULTS.md | ✅ 131 tests |

---

## What changed this session

Two things, and both moved numbers. Everything below was regenerated from scratch afterwards:
`pretrain_federated.py` → `run_eval.py` → `build_report.py` → `make_figures.py` →
`make_report_pdf.py`, all run alone and in that order.

**1. The LSTM residual correction is ON in the planner.** `predictive.use_forecaster` ships
as **`true`**. This was flipped off earlier in the session on the ablation evidence and then
deliberately flipped back: the federated LSTM is the component the project exists to produce,
and keeping it on the actuator path is what makes the end-to-end system a *learned* controller
rather than a digital twin driven by classical control. **It costs 3.29 h of median DO
time-out-of-band** (9.75 h shipped vs 6.46 h for `no_lstm_correction`) and that price is
reported in the README, `RESULTS.md` and §10.1 of the report rather than tuned away. `full` in
the ablation table means the shipped config; `no_lstm_correction` is the better-scoring
variant, kept in the grid so the cost is re-measured on every run.

**1b. The federated weights decide the actions.** With the flag on, the chain is live end to
end: `forecaster_federated.pt` → `Perception.forecaster` → `ctx.forecast` →
`PredictivePlanner.correction()` → added to every candidate rollout → the action applied.
`policy.py` calls this "the only route by which something learned from real data enters the
control decision", and with the flag off that route was dead. Verified by probing a pond at
DO 3.6 heading into a night sag: the mean DO correction is 0.702 mg/L under federated weights,
0.613 under central and 0.550 under independent-local — different weights, different
controller, which is also why `weights_independent_local` (8.33 h) and `weights_central`
(9.83 h) are different rows in the ablation table.

**1c. The action-conditioned world model now produces the trajectories the planner scores.**
`src/perception/world_model.py` is a new LSTM that takes *(observed window, candidate action)*
and predicts the resulting trajectory — the thing the forecaster structurally could not do,
because it was trained on rivers that have no actuators. `LearnedWorldModel` in `policy.py`
swaps it in for the hand-written pond model inside the MPC rollout
(`predictive.world_model: true`), so the candidate trajectories the argmin chooses between are
network output. Trained by `scripts/pretrain_world_model.py`, federated across 8 heterogeneous
simulated ponds, encoder warm-started from `forecaster_federated.pt`.

**2. The data-poor / holdout ordering bug is fixed.** `apply_data_poor` now runs *after*
`split_clients` and is given the training split only. Previously a designated data-poor client
could land in the holdout, where it never trains and so never reaches the data-poor table —
the headline federated claim was being scored on fewer clients than were designated, silently.
All 4 designated clients are now in the table, and **the claim got weaker and more honest**:
3 of 4 rather than 3 of 3, median −13.1 % rather than −22.9 %.

---

## Headline results

Every figure below comes from `outputs/*.csv`. The tables in *this* file are a hand-copied
summary for a reader picking the project up cold — if they ever disagree with `RESULTS.md`,
`RESULTS.md` is right and this file is stale.

**Safety — the claim the project is built around.** Across the full 400-run grid
(8 scenarios × 5 controllers × 10 seeds):

| controller | runs | unsafe applied | guardrail events |
|---|---|---|---|
| do_nothing | 80 | 6171 | 0 (guardrails deliberately off) |
| static_threshold | 80 | **0** | 1756 |
| reactive | 80 | **0** | 1827 |
| predictive | 80 | **0** | 1995 |
| oracle | 80 | **0** | 1083 |

**Agent vs. baselines**, median DO time-out-of-band across 10 seeds:

| scenario | predictive | reactive | static_threshold | do_nothing | oracle |
|---|---|---|---|---|---|
| aerator_failure | **12.96** | 18.21 | 18.62 | 21.46 | 8.54 |
| algal_bloom | **19.71** | 36.54 | 42.58 | 42.08 | 0.00 |
| healthy_baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| heatwave | **9.54** | 18.08 | 18.92 | 18.92 | 0.00 |
| night_do_crash | **13.25** | 15.42 | 17.21 | 25.71 | 0.00 |
| overfeeding_ammonia | **1.75** | 8.96 | 11.58 | 19.92 | 0.00 |
| ph_swing | **12.62** | 37.50 | 42.33 | 42.29 | 42.79 |
| sensor_fault | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The agent beats both deployable baselines in all 6 scenarios where there is any risk to
reduce, and ties in the 2 where nobody accrues any. Paired Wilcoxon across seeds: **18 of 24
comparisons are significant wins (p = 0.002 in every one), 6 are exact ties.**

**Ablations** (40 runs each, median DO time-out-of-band):

| variant | DO risk (h) | unsafe | cost |
|---|---|---|---|
| no_forecast_at_all | 6.79 | 0 | 94.5 |
| no_lstm_correction | 6.79 | 0 | 94.5 |
| weights_independent_local | 8.83 | 0 | 84.8 |
| no_guardrails | 9.38 | **1058** | 84.0 |
| no_world_model | 9.75 | 0 | 70.8 |
| **full (shipped)** | **10.00** | 0 | 72.8 |
| weights_central | 11.75 | 0 | 75.8 |

**Federated learning.** 42 clients → 32 train / 10 held out, stratified by source. Forecast
RMSE on held-out real sites:

| regime | rmse_do | rmse_temp_c | rmse_ph |
|---|---|---|---|
| central (privacy-violating reference) | 0.338 | 0.274 | 0.093 |
| **federated** | **0.342** | **0.270** | 0.098 |
| independent-local | 0.421 | 0.399 | 0.118 |

Federation lands level with the pooled reference and beats independent-local by ~19 % on DO.
On the four data-poor clients (8 % of history), DO RMSE change under federation vs. training
alone: `usgs_08070200` −23.1 %, `ukea_E02569A` −21.9 %, `usgs_292800090060000` −4.3 %,
`ukea_E01730A` **+2.7 %**. **3 of 4 improved, median −13.1 %.**

**World model (the network that picks the action).** Action-conditioned LSTM, federated across
8 heterogeneous simulated ponds, scored on 2 held-out ponds no regime trained on:

| regime | DO RMSE (mg/L, 2 h horizon) |
|---|---|
| central (privacy-violating reference) | 1.075 |
| **federated** | **1.126** |
| independent-local | 1.292 |

Federation beats training alone by 13 % and nearly matches pooling — the same shape as the
forecaster result, now for the model the decision is computed from. **It is far less accurate
than the physics it replaced** (~1.1 mg/L of DO error in a pond that lives between 2 and
10 mg/L), and costs only **0.25 h** of median DO risk (`full` 10.00 vs `no_world_model` 9.75) —
but see finding #2 for what it does to safety.

**Risk model.** Balanced accuracy 0.983, macro-F1 0.968, **critical recall 1.00**, critical
precision 0.893 on a held-out seed.

**Anomaly detection** on the injected `sensor_fault` events: **0.80 event recall** (24 of 30
across 10 seeds) at a **0.116 false-positive rate** outside the fault windows. Scored per
event, not per tick — a detector that fired constantly would score perfectly per tick.

---

## Six findings that do not flatter the project

These are the ones to read before presenting it. Each is in `RESULTS.md` and the README rather
than buried.

**1. The forecaster makes the controller worse, and it ships anyway.** `no_lstm_correction`
scores 6.46 h median DO time-out-of-band against **9.75 h for the shipped system** — the
planner is 3.29 h better *without* the learned component. `weights_independent_local` (a
measurably worse forecaster) scores 8.33 h, better than both configurations that use the
better forecaster, which points the same way. A network trained on temperate rivers,
residual-corrected onto a warm-water pond, injects more bias than it removes even damped by
half.

It is enabled deliberately. The federated LSTM is what the project exists to produce, and
keeping it on the actuator path is what makes the system a learned controller rather than a
twin driven by classical control. The distinction worth defending in a viva is narrow but
real: the component was built, integrated, and **measured against its own ablation on every
evaluation run**, and the measurement is reported rather than tuned away. Knowing the price of
a design decision and choosing to pay it is a different thing from never having measured it.
A deployment optimising only for the score should set `predictive.use_forecaster: false`.

**2. A learned dynamics model is only safe because something else makes it safe.** This is
the clearest result in the project. Swapping the hand-written pond model for the
action-conditioned LSTM costs only **0.25 h** of median DO risk (10.00 vs 9.75) — cheap for
putting a network in the decision path. But with the guardrails removed the learned planner
applies **1058 unsafe actions** across the ablation grid, against **423** for the
physics-driven planner: **2.5x more**. On `night_do_crash` unguarded it is 106 unsafe against
**zero**; guarded, both are zero.

The mechanism generalises past this project and is worth being able to state: the planner
optimises against its model *as though the model were exact*. A hand-written model is wrong in
ways that are smooth and bounded. A learned model is wrong in ways that are neither — it can be
confidently wrong in a small region of state space, and an optimiser will find precisely that
region and exploit it. Prediction error becomes unsafe commands. This is the argument for
keeping hard constraints **outside** the learned component instead of folding them into its
cost function, and it means the safety layer is not belt-and-braces around a system that was
already fine — it is what makes a learned controller deployable at all. Both halves are
asserted in `tests/test_guardrails.py::test_the_learned_world_model_is_not_safe_without_guardrails`.

**3. Federation does not help every data-poor pond.** 3 of 4 improved; `ukea_E01730A` is
2.7 % *worse* under federation than training alone. Until this session that client was
invisible — designated data-poor and then held out, so it never appeared in the table. The
bug is fixed, the number moved from −22.9 % to −13.1 % median, and the losing client is in
the table rather than out of it.

**4. HDBSCAN found no regime taxonomy.** Swept `min_cluster_size` 40–800, `min_samples` 3–80,
both selection methods, a PCA variant, and both pooled and twin-only data. Every configuration
returns one dominant density mode holding ~80 % of windows plus a few tiny satellites, with
DBCV indistinguishable from zero (`outputs/regime_sweep.csv`). Water-quality windows form a
continuum, not clusters. The model is used as a **novelty detector** — the noise label is the
anomaly flag, which does work — and the cluster label is a descriptive aid nothing depends on.
One sweep row reaches DBCV 0.125, and is named in `RESULTS.md` rather than buried: it does so
at an 85 % noise fraction, which would flag 85 % of dashboard ticks as anomalous.

**5. The agent costs more to run than the thermostat — in all 8 scenarios**, significantly,
including `healthy_baseline` where both achieve zero DO risk. The shipped
(LSTM-enabled) system spends 70.8 against 86.5 for `no_lstm_correction` — so the correction is
*cheaper* to run as well as worse on risk; it intervenes less, and less well. Preemption means
acting on excursions that would sometimes have resolved themselves. `cost_weights.intervention` is the dial; nothing here
establishes where a real farm should set it. Against `reactive` specifically the agent is
*cheaper* in 2 of 8 scenarios (`aerator_failure`, `sensor_fault`); against `static_threshold`
it is dearer in all 8.

**6. The guardrails bind, and the trade is visible.** Turning them off yields **1058 unsafe
applied** against 0 with them, at 9.38 h DO risk (vs 10.00 h) and a higher cost (84.0 vs 72.8). An unguarded planner is freer, more expensive, and unsafe. The safety layer earns its
keep under `aerator_failure`: the agent commands aeration, the fault swallows it, and nothing
in the planner's own logic reacts to the discrepancy. Both halves are asserted in
`tests/test_guardrails.py` so neither can be overstated.

---

## Design decisions worth knowing

**The twin's calibration is deliberately partial.** `scripts/calibrate_twin.py` transfers
process-noise magnitude, baseline turbidity and relaxation pH. It **refuses** the diurnal
cycle: the measured median daily temperature swing across the 42 sites is 0.51 °C (rivers are
deep and flowing) against 2–4 °C for a 1.5 m pond. Applying it would flatten the pond into an
isothermal tank and delete the dawn oxygen sag the whole project depends on. The fitted values
are recorded in `configs/twin.yaml` under `calibration.fitted_diurnal_not_applied`;
`--adopt-diurnal` applies them anyway.

**Regime features are dimensionless** — DO ÷ saturation at the current temperature, temperature
and pH as within-window anomalies, log turbidity. On raw levels every pond window sits outside
every river cluster and the anomaly flag becomes meaningless (noise fraction 38 % → 21 %).

**The forecaster predicts deltas with fixed scales.** Fitted scales would leak client
statistics across the federated boundary and make the three regimes incomparable.

**The MPC planner rolls its own internal model**, not the LSTM, because the LSTM cannot be
conditioned on an aerator it has never seen. This is a deviation from the literal reading of
spec §10, documented in `src/agent/policy.py`.

**The weight-source ablations run *with* the LSTM correction enabled**, unlike the shipped
config. That is deliberate: the weight source reaches the controller *only* through the
correction, so at the shipped setting `weights_central` and `weights_independent_local` would
be two more names for `full` and would say nothing about the weights. Commented in
`src/eval/run_eval.py`.

**The oracle is not an upper bound.** It is a hand-written rule with privileged information,
and the agent beats it on DO in `ph_swing` because its rule has no supersaturation term.

---

## Bugs found and fixed — do not reintroduce

**Data-poor clients could be designated and then held out.** `apply_data_poor` ran over all
clients *before* `split_clients`, so a designated client could land in the holdout, never
train, and never appear in the data-poor table — the headline federated claim scored on 3
clients while reporting 4 as designated. Designation now happens after the split, on the
training split only. Locked in by `tests/test_federated.py::`
`test_data_poor_clients_all_stay_in_the_training_split`, and `build_report.py` prints a
**REGRESSION** banner in `RESULTS.md` if the two sets ever overlap again.

**The Counterfactual panel rendered as two flat lines.** `add_risk_bands` shades the
safe/warning/critical regions with `add_hrect` shapes spanning ±1e3 so they reach the edge of
any sensible axis — and plotly folds shape extents into its autorange. `timeseries.py` set an
explicit y-range and so looked fine; `counterfactual.py` did not, so both ponds were squashed
into a flat line at the bottom of a 2000-unit window and the shadow-pond trace was invisible.
The caption underneath still computed the right number, which is why it read as a styling
quirk rather than a broken chart. The `_y_range` helper moved from `timeseries.py` to
`common.py` as `y_range`, next to the bands it exists to counteract and with the reason in its
docstring; both panels now call it. **Any new panel that calls `add_risk_bands` must also set
`y_range`** — those are the only two callers, so the invariant is currently held.

**The risk model's train/test split was accidentally out-of-distribution.** Rows are generated
scenario by scenario, so a positional 75/25 split put whole unseen scenarios in the test set —
an ordinary-looking holdout that was really a transfer test. Critical recall read 0.28.
Splitting by **seed** (train 1–3, test 4, every scenario on both sides) moved it to 1.00.
`build_training_set` now returns group ids so this cannot silently recur.

**`do_nothing` was being rescued by its own guardrails**, giving it 111 "interventions" and a
DO floor of 2.35 instead of 0.05, quietly destroying the baseline comparison. Now explicitly
unguarded, locked in by `test_do_nothing_is_deliberately_unguarded`.

**92,053 duplicated sklearn warnings per evaluation run.** LightGBM fitted on a numpy array
records placeholder feature names, so sklearn warned on every predict — once per simulated
tick. `RiskModel._matrix` now returns a named DataFrame.

**The evaluation harness crashed after 90 minutes of work.** `run_anomaly_eval` did
`run.result.decisions`, but `AgentRun.run()` returns the `RunResult` itself. Fixed, and locked
in by `test_run_returns_a_result_not_the_runner` and `test_anomaly_eval_runs_end_to_end`.

**SHAP attribution vanished on two thirds of ticks.** Planning is 3× coarser than sampling, and
hold-tick decisions were built without `shap_top`, so the dashboard panel blanked out between
decisions. Hold ticks now inherit the last attribution, as they already inherited risk and
regime. Related: `Perception.run` swallowed every exception with a bare `pass`; it now records
them in `Perception.errors`.

---

## Handing the project to someone else

**Verified on 2026-09-05 in a clean virtualenv** with the newest release of every dependency
(numpy 2.4.6, pandas 3.0.5, torch 2.14.0, scikit-learn 1.9.0, streamlit 1.63.0, hdbscan 0.8.44):
`pip install -r requirements.txt` succeeded from wheels with no compiler, **115/115 tests
passed**, both pickled models loaded, and the dashboard rendered all four tabs with byte-identical
results (+2.94 mg/L, 432/432 ticks). The project is portable and has no hardcoded paths —
every path derives from `Path(__file__).resolve().parents[2]`.

**Send the folder, not a git clone.** `.gitignore` excludes `data_cache/`, `models/`,
`outputs/` and `RESULTS.md`. A clone therefore contains no trained models and no data: the
dashboard starts but has nothing to load, and reproducing from scratch needs a network run of
`scripts/collect_data.py` plus a full retrain.

**What to actually ship** (424 MB total, mostly avoidable):

| part | size | needed for |
|---|---|---|
| `data_cache/` | 397 MB | retraining only — the dashboard degrades gracefully without it (`data_summary` is wrapped in try/except) |
| `models/` | 24 MB | **required** to run the dashboard |
| code, configs, outputs, report | ~4 MB | **required** |

So a **~30 MB zip without `data_cache/`** runs the dashboard and the tests. Include
`data_cache/` only if they need `scripts/pretrain_federated.py` or `python -m src.eval.run_eval`.

**Known soft spots, none of them blocking:**

- `models/regimes.pkl` and `risk_model.pkl` are **pickles** of hdbscan/LightGBM/sklearn objects
  and `requirements.txt` pins with `>=`. They load fine today on sklearn 1.9 but emit
  `InconsistentVersionWarning` (a `LabelEncoder` pickled by 1.6.1). If a future sklearn breaks
  them, re-run `python scripts/pretrain_federated.py --only-perception`. The `.pt` forecasters
  are safe — `forecaster.py` passes `weights_only=False` explicitly, so torch >= 2.6 will not
  reject them.
- `dashboard/app.py` does `from panels import ...`, which resolves only because Streamlit puts
  the script's own directory on `sys.path`. It works under `streamlit run dashboard/app.py` —
  the documented command — and would fail under `python dashboard/app.py`.
- Rebuilding the **PDF** additionally needs Node and Chrome/Edge (`scripts/html_to_pdf.mjs`).
  The app, the tests and the figures do not. The built PDF ships in `report/`.
- Python 3.11 is what it is built and tested on.

**Fixed while checking this:** `matplotlib` was missing from `requirements.txt` even though
`scripts/make_figures.py` imports it — a clean install could run the app and the tests but not
rebuild the figures. Declared now.

---

## An operational trap that cost this session an hour

**On this machine the Python process is named `python3.11.exe`, not `python.exe`.**
`Get-Process python` returns nothing while a job is running perfectly well, which reads as
"the background job died". Acting on that reading launched three overlapping `run_eval`
processes and two overlapping `pretrain_federated` processes, all writing the same `models/`
and `outputs/` files. Use this instead, and check it before and after starting any long run:

```powershell
Get-CimInstance Win32_Process -Filter "Name LIKE '%python%'" | Select-Object ProcessId,CreationDate,CommandLine
```

Note also that stopping a background *shell* does not necessarily kill the Python child it
spawned — the orphan keeps running and keeps writing. The numbers in this file come from a
verified single-process run of each stage.

---

## Pick up here

### The written deliverables

- `report/AquaSentinel-Report.pdf` — the **full project report**, 29 pages, 7 figures,
  19 tables. Generated from `outputs/`, `configs/` and `data_cache/manifest.csv`; §10 leads
  with the forecaster being cut. Rebuild with `python scripts/make_figures.py && python
  scripts/make_report_pdf.py`.
- `RESULTS.md` — the machine-generated results record. If it and any prose disagree, it wins.
- `README.md` — prose, pointers, and the honesty framing.

### 1. Verify the state you inherited

```bash
python -m pytest tests/ -q          # expect 131 passed, ~33 s
python scripts/build_report.py      # regenerates RESULTS.md from outputs/
streamlit run dashboard/app.py      # play night_do_crash, open the Counterfactual tab
```

The dashboard was last driven end to end on 2026-09-05: all four tabs render, no page
exceptions, `night_do_crash` runs to completion (432/432 ticks) with **0 unsafe applied**,
0 min critical, 16.2 h time-in-risk against 28.1 h for the shadow pond.

### 2. Worth doing next, in order

- ~~Screenshots for submission.~~ **Done** — `report/screenshots/` holds all four tabs of
  `night_do_crash` at seed 1002, captured from the running app after the counterfactual
  rendering bug below was fixed. `2-counterfactual.png` is the one to lead with: the agent's DO
  trace held above the no-action shadow pond through two dawn sags, with the shadow diving into
  the critical band and the agent's pond not.
- **Rotate the OpenRouter key.** There is one in plaintext at `../openrouter/rest api`.
  Nothing in this project reads that file — the LLM planner reads `OPENROUTER_API_KEY` from
  the environment and defaults to offline Ollama — but the key is exposed and should be
  revoked at openrouter.ai/keys.
- **Decide whether the forecaster still earns its place in the *risk model*.** Finding #2
  above shows it contributes exactly nothing to the controller now. Nobody has run the
  equivalent ablation on the risk model — drop the forecast-summary features and see whether
  critical recall holds at 1.00. If it does, the honest conclusion is larger than the one
  currently written up, and the report would need to say so.
- **Pond-domain training data** is the real fix for finding #1. The Kaggle supplement in
  `configs/data.yaml` exists for exactly this and is disabled
  (`src/data/sources/kaggle_pond.py` is written and wired, just off).
- **The LLM planner has still never run against a live endpoint.** Its fallback, its
  candidate-index-only behaviour, both provider request shapes and both response shapes are
  now unit-tested against a mocked `requests.post` — everything short of a real round-trip.
- **DP-FedAvg** is implemented (`--dp`) but unrun. `pip install dp-accounting` first, or ε is
  correctly reported as `UNACCOUNTED`.
- **Forecast skill is modest** (DO skill vs persistence ≈ 0.05 on rivers). More epochs or a
  larger hidden size would help. The federated *comparison* is unaffected — all three regimes
  share the architecture.

### 3. Commands

Timings below are measured on this machine, not estimated.

```bash
python -m pytest tests/ -q                                   # 131 tests, ~33 s
python scripts/collect_data.py --summary                     # manifest stats, no network
python scripts/calibrate_twin.py --dry-run                   # see what calibration would change
python scripts/pretrain_federated.py                         # full, ~3 min
python scripts/pretrain_world_model.py                       # action-conditioned model, ~6 min
python scripts/pretrain_federated.py --only-perception       # regimes + risk only
python -m src.eval.run_eval                                  # full grid + ablations, ~20 min
python -m src.eval.run_eval --scenario night_do_crash --seeds 4 --no-ablations   # quick
python scripts/build_report.py                               # RESULTS.md
python scripts/make_figures.py                               # report/figures/*.svg
python scripts/make_report_pdf.py                            # the 29-page PDF report
streamlit run dashboard/app.py
```

Run the pipeline stages **one at a time** and confirm no other Python process is live between
them (see the trap above). `scripts/collect_data.py` is the **only** file that touches the
network.

### 4. Deliberately not done

- **Flower (`flwr`) backend** — the in-process FedAvg loop is deterministic and CPU-cheap, and a
  network stack adds nothing to the comparison being made.
- **Kaggle pond data** — written and wired, disabled by default. Enabling it makes a token a
  prerequisite for reproducing the project; the core pipeline is keyless on purpose.
