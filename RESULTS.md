# AquaSentinel — results

_Generated 2026-11-24T16:25:27 by `scripts/build_report.py`. Do not edit._

## 1. Real data actually collected

- **7 real fish ponds**, one per federated client
- **83,825 harmonised rows** at 15-minute cadence
- window 2021-06-18 to 2021-12-30
- 1 distinct regions; sources: kaggle_pond (7 sites)

Channel coverage across sites (this is the non-IID structure, not a simulation of it):

| channel | sites reporting it |
|---|---|
| do | 7 |
| ph | 7 |
| tan | 7 |
| temp_c | 7 |
| turbidity | 7 |

Licences: CC BY 4.0 - ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets (Kaggle)

> Real catfish aquaponics ponds in Nsukka, Nigeria (Udanor et al., *Data in Brief* 2022), logged by low-cost IoT probes. Ponds whose timelines are mostly sensor outage were dropped by the QC thresholds in `configs/data.yaml`; nitrate is excluded because the raw column is an ADC count, not a concentration.

## 2. Federated vs. independent-local vs. central

Forecast RMSE on **held-out real ponds** — ponds no regime trained on. Lower is better.

| regime | rmse_do | rmse_temp_c | rmse_ph |
|---|---|---|---|
| central | 1.789 | 0.520 | 0.553 |
| federated | 1.767 | 0.521 | 0.552 |
| independent_local | 1.915 | 0.530 | 0.553 |

Scored on 2 held-out pond(s). `independent_local` is the mean over every pond-specific model, because a pond that trained alone has no single global model to test — reporting only its best member would be cherry-picking.

### The data-poor ponds — the claim federation is actually about

Each of these clients kept only a fraction of its history and is scored on **its own** held-out tail, under the model it trained alone versus the federated global model it helped produce without sharing a single row.

| regime | site | data_fraction | n_train | rmse_do | rmse_temp_c | rmse_ph |
|---|---|---|---|---|---|---|
| independent_local | kaggle_pond_pond01 | 0.080 | 73 | 1.205 | 0.604 | 0.617 |
| federated | kaggle_pond_pond01 | 0.080 | 73 | 1.248 | 0.590 | 0.614 |
| central | kaggle_pond_pond01 | 0.080 | 73 | 1.218 | 0.594 | 0.615 |
| independent_local | kaggle_pond_pond03 | 0.080 | 70 | 1.728 | 0.810 | 0.493 |
| federated | kaggle_pond_pond03 | 0.080 | 70 | 0.729 | 0.829 | 0.489 |
| central | kaggle_pond_pond03 | 0.080 | 70 | 0.709 | 0.802 | 0.490 |

**Federation improved DO forecast RMSE on 1 of 2 data-poor sites** (median change -27.1%). Scoped to these sites, this window and this architecture — not a general claim.

FedAvg ran 12 rounds; mean local loss 59.13018 → 47.50981.

### Forecast error by horizon (held-out real ponds)

| horizon_minutes | central | federated | independent_local |
|---|---|---|---|
| 15 | 2.661 | 2.636 | 2.667 |
| 30 | 2.779 | 2.756 | 2.739 |
| 45 | 2.972 | 2.942 | 2.918 |
| 60 | 2.656 | 2.634 | 2.643 |
| 75 | 2.903 | 2.876 | 2.855 |
| 90 | 3.305 | 3.301 | 3.300 |
| 105 | 3.312 | 3.319 | 3.341 |
| 120 | 3.756 | 3.734 | 3.734 |
| 135 | 3.756 | 3.730 | 3.713 |
| 150 | 3.720 | 3.703 | 3.690 |
| 165 | 3.864 | 3.852 | 3.849 |
| 180 | 3.936 | 3.936 | 3.888 |
| 195 | 3.898 | 3.878 | 3.861 |
| 210 | 4.019 | 3.973 | 4.002 |
| 225 | 4.148 | 4.110 | 4.051 |
| 240 | 3.986 | 3.958 | 3.896 |
| 255 | 4.100 | 4.070 | 4.036 |
| 270 | 4.204 | 4.198 | 4.134 |
| 285 | 4.331 | 4.320 | 4.232 |
| 300 | 4.432 | 4.407 | 4.334 |
| 315 | 4.603 | 4.602 | 4.474 |
| 330 | 4.473 | 4.473 | 4.344 |
| 345 | 4.734 | 4.702 | 4.595 |
| 360 | 4.575 | 4.546 | 4.468 |

Dissolved oxygen, mg/L. Error growing with horizon is the expected and honest shape.

## 2b. The action-conditioned world model — the network that picks the action

The forecaster in §2 predicts what the water does *on its own*; it was trained on real pond sensor data that contains no log of actuator commands, so it cannot be asked what an aerator would do. The **world model** can: it takes the observed window *and a candidate action* and predicts the resulting trajectory. The MPC planner scores those learned trajectories and takes the argmin, so the action the agent applies is chosen from predictions made by a neural network rather than by a hand-written differential equation.

**Where its training data comes from, stated plainly.** The real pond dataset records water quality but not when the aerator ran, how much was fed or when lime went in - so it has no `(action → response)` pairs. The action-conditioned part is therefore learned from twin rollouts under randomised action sequences. What *is* carried over from real data is the **encoder**, warm-started from `forecaster_federated.pt` — the model FedAvg produced across the real ponds.

**Federation here is across simulated ponds**, each with its own volume, depth, aeration efficiency, algal productivity and temperature regime. That is a different claim from §2's federation over real ponds. Both use the same `fedavg`; only §2's clients are real.

Prediction error on held-out ponds no regime trained on. `rmse` is in scaled-delta units; `rmse_do_mgl` is dissolved oxygen in mg/L over the 2-hour horizon.

| regime | rmse | rmse_do_mgl |
|---|---|---|
| central | 0.204 | 0.734 |
| federated | 0.294 | 1.147 |
| independent_local | 0.309 | 1.198 |

Federated DO error 1.147 mg/L against 1.198 for independent-local (federation beats training alone) and 0.734 for the pooled reference.

**It is markedly less accurate than the physics it replaced**, and that is the honest headline: roughly 1.1 mg/L of DO error over the horizon, in a pond that operates between 2 and 10 mg/L. The consequences are measured in the ablation table (`no_world_model`) and in the safety section, not asserted here.

FedAvg ran 12 rounds; held-out DO RMSE 1.430 → 1.147 mg/L.

## 3. Perception

### Regimes (HDBSCAN, unsupervised — no labels consumed anywhere)

| n_clusters | noise_fraction | dbcv | silhouette | n_points |
|---|---|---|---|---|
| 6 | 0.344 | 0.040 | 0.149 | 4667 |

DBCV is HDBSCAN's density-based validity index, the appropriate one for non-globular clusters; silhouette is included for comparability with the usual literature. `noise_fraction` is the share of windows assigned to no cluster — those are the anomaly flags.

**A DBCV close to zero means the clusters are not well separated**, whatever the silhouette says. The intended deliverable was a taxonomy of operating regimes, and the data does not support one.

#### The sweep behind that claim

| min_cluster_size | min_samples | selection | n_clusters | noise_fraction | dbcv | silhouette | n_points |
|---|---|---|---|---|---|---|---|
| 40 | 10 | eom | 6 | 0.344 | 0.040 | 0.149 | 4667 |
| 40 | 5 | eom | 11 | 0.313 | 0.060 | -0.020 | 4667 |
| 80 | 5 | eom | 4 | 0.299 | 0.084 | 0.117 | 4667 |
| 80 | 10 | eom | 4 | 0.364 | 0.021 | 0.155 | 4667 |
| 150 | 25 | eom | 3 | 0.479 | 0.000 | 0.207 | 4667 |
| 400 | 50 | eom | 0 | 1.000 | 0.000 | - | 4667 |
| 150 | 25 | leaf | 3 | 0.479 | 0.000 | 0.207 | 4667 |
| 400 | 50 | leaf | 0 | 1.000 | 0.000 | - | 4667 |
| 800 | 80 | leaf | 0 | 1.000 | 0.000 | - | 4667 |

Across every configuration DBCV stays at or below 0.084: no setting finds well-separated clusters.

Continuous water-quality windows form a connected continuum — a pond slides around a manifold rather than jumping between discrete states. The model is therefore used as a **novelty detector** (the noise label is the anomaly flag; how well that works is measured in the next section), and the cluster label is a descriptive reading aid that nothing downstream depends on.

### Risk model (LightGBM over current channels + forecast summaries)

| accuracy | balanced_accuracy | macro_f1 | critical_recall | critical_precision | n | held_out_seed |
|---|---|---|---|---|---|---|
| 0.984 | 0.983 | 0.983 | 0.983 | 0.979 | 2136 | 4 |

The target is **rule-derived**: the risk band that `configs/twin.yaml` assigns to the worst point of the true next horizon. This model has not learned water chemistry; it has learned to anticipate a known rule from an imperfect forecast.

### Anomaly detection on injected sensor faults

Event recall **1.00** (30/30 injected fault events flagged across 10 seeds), false-positive rate **0.820** outside the fault windows.

At a false-positive rate of 82% the flag fires on most ordinary windows, so it is not usable as an alarm on its own: the model was fitted mostly on noisy real-pond windows, and the simulated pond's normal behaviour falls outside them.

Scored per event, not per tick: a detector that fires constantly would score perfectly on a per-tick metric. The noise label cannot distinguish a broken probe from a genuine novel excursion, and the false-positive rate is the price of that.

## 4. Agent vs. baselines, in the twin

### Safety — the headline claim

| controller | runs_n | unsafe_applied_total | unsafe_applied_max | guardrail_events_total | scenarios |
|---|---|---|---|---|---|
| do_nothing | 80 | 6169 | 227 | 0 | 8 |
| oracle | 80 | 0 | 0 | 1105 | 8 |
| predictive | 80 | 0 | 0 | 2065 | 8 |
| reactive | 80 | 0 | 0 | 1906 | 8 |
| static_threshold | 80 | 0 | 0 | 1728 | 8 |

`do_nothing` is deliberately unguarded: it is a reference trajectory, not a controller, and letting the safety layer rescue it would destroy the comparison it exists to provide. Every guardrailed controller must show **0** unsafe-applied.

### Time out of band (median across seeds, with IQR)

| scenario | controller | risk_h_do_median | risk_h_do_iqr | time_critical_h_median | interventions_median | intervention_cost_median | unsafe_applied_median |
|---|---|---|---|---|---|---|---|
| aerator_failure | do_nothing | 21.541 | 0.291 | 7.458 | 0.000 | 0.000 | 151.500 |
| aerator_failure | oracle | 8.459 | 0.229 | 0.000 | 432.000 | 432.000 | 0.000 |
| aerator_failure | predictive | 11.667 | 0.438 | 0.000 | 154.500 | 99.750 | 0.000 |
| aerator_failure | reactive | 18.291 | 0.459 | 0.875 | 163.500 | 160.500 | 0.000 |
| aerator_failure | static_threshold | 18.625 | 0.209 | 2.333 | 141.000 | 133.500 | 0.000 |
| algal_bloom | do_nothing | 42.125 | 0.333 | 8.541 | 0.000 | 0.000 | 136.500 |
| algal_bloom | oracle | 0.000 | 0.000 | 6.875 | 465.000 | 478.500 | 0.000 |
| algal_bloom | predictive | 15.459 | 8.459 | 7.917 | 228.500 | 103.500 | 0.000 |
| algal_bloom | reactive | 37.375 | 1.396 | 6.458 | 163.500 | 42.000 | 0.000 |
| algal_bloom | static_threshold | 42.791 | 0.459 | 8.167 | 135.000 | 12.000 | 0.000 |
| healthy_baseline | do_nothing | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| healthy_baseline | oracle | 0.000 | 0.000 | 0.000 | 3.000 | 3.000 | 0.000 |
| healthy_baseline | predictive | 0.000 | 0.000 | 0.000 | 13.500 | 17.250 | 0.000 |
| healthy_baseline | reactive | 0.000 | 0.000 | 0.000 | 3.000 | 4.500 | 0.000 |
| healthy_baseline | static_threshold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heatwave | do_nothing | 18.708 | 0.521 | 0.000 | 0.000 | 0.000 | 0.000 |
| heatwave | oracle | 0.000 | 0.000 | 0.000 | 711.000 | 711.000 | 0.000 |
| heatwave | predictive | 9.791 | 1.249 | 0.000 | 69.000 | 69.000 | 0.000 |
| heatwave | reactive | 18.541 | 0.875 | 0.000 | 4.500 | 9.750 | 0.000 |
| heatwave | static_threshold | 18.708 | 0.521 | 0.000 | 0.000 | 0.000 | 0.000 |
| night_do_crash | do_nothing | 25.667 | 0.167 | 10.167 | 0.000 | 0.000 | 226.500 |
| night_do_crash | oracle | 0.000 | 0.000 | 0.000 | 432.000 | 432.000 | 0.000 |
| night_do_crash | predictive | 14.541 | 3.208 | 0.000 | 160.500 | 52.500 | 0.000 |
| night_do_crash | reactive | 15.375 | 1.167 | 0.000 | 81.000 | 54.000 | 0.000 |
| night_do_crash | static_threshold | 17.250 | 0.562 | 0.000 | 45.000 | 33.000 | 0.000 |
| overfeeding_ammonia | do_nothing | 19.833 | 0.188 | 6.333 | 0.000 | 0.000 | 105.000 |
| overfeeding_ammonia | oracle | 0.000 | 0.000 | 2.583 | 219.000 | 142.500 | 0.000 |
| overfeeding_ammonia | predictive | 1.167 | 2.812 | 1.708 | 125.500 | 93.000 | 0.000 |
| overfeeding_ammonia | reactive | 8.833 | 1.896 | 1.000 | 144.000 | 33.000 | 0.000 |
| overfeeding_ammonia | static_threshold | 11.708 | 2.271 | 5.125 | 104.000 | 12.000 | 0.000 |
| ph_swing | do_nothing | 42.209 | 0.312 | 2.250 | 0.000 | 0.000 | 0.000 |
| ph_swing | oracle | 42.667 | 0.209 | 0.250 | 30.000 | 10.500 | 0.000 |
| ph_swing | predictive | 11.709 | 2.604 | 0.875 | 149.500 | 113.250 | 0.000 |
| ph_swing | reactive | 37.459 | 0.312 | 0.000 | 90.000 | 30.000 | 0.000 |
| ph_swing | static_threshold | 45.375 | 2.667 | 0.000 | 27.000 | 0.000 | 0.000 |
| sensor_fault | do_nothing | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sensor_fault | oracle | 0.000 | 0.000 | 0.000 | 8.500 | 1.500 | 0.000 |
| sensor_fault | predictive | 0.000 | 0.000 | 0.000 | 22.500 | 17.250 | 0.000 |
| sensor_fault | reactive | 0.000 | 0.000 | 0.000 | 11.000 | 9.250 | 0.000 |
| sensor_fault | static_threshold | 0.000 | 0.000 | 0.000 | 8.000 | 5.000 | 0.000 |

Medians, not means — a single unlucky seed must not carry a claim.

### Paired Wilcoxon signed-rank, agent vs. each baseline

Paired because the same seed gives every controller the identical perturbation schedule and noise realisation. Negative `median_diff` means the agent is better. `p` and `n` are reported for every comparison; a comparison with too few seeds to test says so instead of quietly printing a mean.

| scenario | controller_b | median_a | median_b | median_diff | p | n | effect_size_rb | note |
|---|---|---|---|---|---|---|---|---|
| aerator_failure | static_threshold | 11.667 | 18.625 | -6.791 | 0.002 | 10 | 1.000 | nan |
| aerator_failure | do_nothing | 11.667 | 21.541 | -9.875 | 0.002 | 10 | 1.000 | nan |
| aerator_failure | reactive | 11.667 | 18.291 | -6.375 | 0.002 | 10 | 1.000 | nan |
| algal_bloom | static_threshold | 15.459 | 42.791 | -27.250 | 0.002 | 10 | 1.000 | nan |
| algal_bloom | do_nothing | 15.459 | 42.125 | -26.541 | 0.002 | 10 | 1.000 | nan |
| algal_bloom | reactive | 15.459 | 37.375 | -21.834 | 0.002 | 10 | 1.000 | nan |
| healthy_baseline | static_threshold | 0.000 | 0.000 | 0.000 | 1.000 | 10 | - | all differences are exactly zero |
| healthy_baseline | do_nothing | 0.000 | 0.000 | 0.000 | 1.000 | 10 | - | all differences are exactly zero |
| healthy_baseline | reactive | 0.000 | 0.000 | 0.000 | 1.000 | 10 | - | all differences are exactly zero |
| heatwave | static_threshold | 9.791 | 18.708 | -8.584 | 0.002 | 10 | 1.000 | nan |
| heatwave | do_nothing | 9.791 | 18.708 | -8.584 | 0.002 | 10 | 1.000 | nan |
| heatwave | reactive | 9.791 | 18.541 | -8.584 | 0.002 | 10 | 1.000 | nan |
| night_do_crash | static_threshold | 14.541 | 17.250 | -2.709 | 0.011 | 10 | 0.956 | nan |
| night_do_crash | do_nothing | 14.541 | 25.667 | -11.375 | 0.002 | 10 | 1.000 | nan |
| night_do_crash | reactive | 14.541 | 15.375 | -0.916 | 0.232 | 10 | 0.455 | nan |
| overfeeding_ammonia | static_threshold | 1.167 | 11.708 | -10.542 | 0.002 | 10 | 1.000 | nan |
| overfeeding_ammonia | do_nothing | 1.167 | 19.833 | -18.458 | 0.002 | 10 | 1.000 | nan |
| overfeeding_ammonia | reactive | 1.167 | 8.833 | -7.292 | 0.002 | 10 | 1.000 | nan |
| ph_swing | static_threshold | 11.709 | 45.375 | -33.376 | 0.002 | 10 | 1.000 | nan |
| ph_swing | do_nothing | 11.709 | 42.209 | -30.500 | 0.002 | 10 | 1.000 | nan |
| ph_swing | reactive | 11.709 | 37.459 | -25.583 | 0.002 | 10 | 1.000 | nan |
| sensor_fault | static_threshold | 0.000 | 0.000 | 0.000 | 1.000 | 10 | - | all differences are exactly zero |
| sensor_fault | do_nothing | 0.000 | 0.000 | 0.000 | 1.000 | 10 | - | all differences are exactly zero |
| sensor_fault | reactive | 0.000 | 0.000 | 0.000 | 1.000 | 10 | - | all differences are exactly zero |

The agent is significantly better (p < 0.05) in **17 of 24** scenario × baseline comparisons on this metric.
The 6 exact ties are all in `healthy_baseline`, `sensor_fault` — scenarios in which no controller accrues any DO time-out-of-band, so there is nothing to win.

### What that costs

Risk reduction is not free, and reporting only the win would be a half-truth. Agent versus static threshold on cumulative intervention cost — **positive `median_diff` means the agent spends more**:

| scenario | median_a | median_b | median_diff | p | n |
|---|---|---|---|---|---|
| aerator_failure | 99.750 | 133.500 | -33.500 | 0.002 | 10 |
| algal_bloom | 103.500 | 12.000 | 92.500 | 0.002 | 10 |
| healthy_baseline | 17.250 | 0.000 | 17.250 | 0.008 | 10 |
| heatwave | 69.000 | 0.000 | 69.000 | 0.002 | 10 |
| night_do_crash | 52.500 | 33.000 | 21.500 | 0.002 | 10 |
| overfeeding_ammonia | 93.000 | 12.000 | 81.000 | 0.002 | 10 |
| ph_swing | 113.250 | 0.000 | 113.250 | 0.002 | 10 |
| sensor_fault | 17.250 | 5.000 | 10.500 | 0.002 | 10 |

The agent is more expensive to run in **7 of 8** scenarios, including ones where it achieves the same zero DO risk the thermostat already achieved. It buys preemption — acting before a threshold is crossed — and preemption means acting on excursions that would sometimes have resolved on their own. `cost_weights.intervention` in `configs/agent.yaml` is the dial that trades one against the other; nothing here establishes where a real farm should set it.

## 5. Ablations

| ablation | median_do_risk_h | median_critical_h | median_cost | unsafe_total | runs |
|---|---|---|---|---|---|
| no_forecast_at_all | 8.791 | 0.000 | 96.750 | 0 | 40 |
| no_lstm_correction | 8.791 | 0.000 | 96.750 | 0 | 40 |
| no_guardrails | 9.416 | 0.000 | 75.750 | 1057 | 40 |
| no_world_model | 10.041 | 0.000 | 68.250 | 0 | 40 |
| weights_central | 10.209 | 0.000 | 86.250 | 0 | 40 |
| full | 10.833 | 0.000 | 75.500 | 0 | 40 |
| weights_independent_local | 11.375 | 0.000 | 73.250 | 0 | 40 |

**The LSTM residual correction makes control worse, and it ships anyway.** The system scores 10.83 h of DO time-out-of-band with the correction against 8.79 h without it — so `no_lstm_correction` is the better-performing configuration, by 2.04 h. A forecaster trained on noisy low-cost pond probes, used as a correction on the simulated pond, injects more bias than it removes. The correction is damped by 0.5 in `policy.PredictivePlanner.correction` and it is still net negative.

It is enabled in the shipped configuration as a deliberate choice rather than an oversight. The federated LSTM is the component this project exists to produce, and keeping it on the actuator path is what makes the end-to-end system a *learned* controller rather than a digital twin rolled forward by classical control. The cost of that choice is stated here, in the table above, and in the README — quantified rather than hidden. A deployment that cared only about the score should set `predictive.use_forecaster: false`.

- `full` — the shipped configuration: federated weights, forecast overlay, LSTM correction applied to the planner, guardrails on.
- `no_lstm_correction` — the MPC planner rolls its internal pond model forward without the LSTM residual correction. This isolates what the forecaster contributes to control, and it is what justifies calling the correction's cost measured rather than assumed.
- `no_forecast_at_all` — no forecaster loaded, so the risk model and the guardrails lose their forward-looking features too.
- `no_guardrails` — the ablation that is expected to be *worse on safety*. If `unsafe_total` here is not greater than zero, the guardrails are not doing anything and the safety claim is empty.
- `weights_*` — the same agent driven by each federated regime's weights, at the shipped setting. The LSTM correction is the path by which the weight source reaches the controller at all.
- `no_world_model` — the planner rolls the **hand-written** pond model forward instead of the action-conditioned LSTM. This is what isolates the learned dynamics model: `full` minus this row is the price of replacing physics with a network inside the decision.

## 6. Provenance

- models trained 2026-10-25T16:15:47
- 7 clients; 2 held out: `kaggle_pond_pond04, kaggle_pond_pond09`
- data-poor clients: `kaggle_pond_pond03, kaggle_pond_pond01`
- FedAvg: 12 rounds x 2 local epochs, client fraction 1.0, sample-weighted: True
- differential privacy: disabled

- twin: UNCALIBRATED - hand-set warm-water pond defaults (FAO / Boyd).
  - a fit to the 7 real ponds was computed and **deliberately not applied** (DO noise 4.2278 mg/L/√h, baseline turbidity 93.3647 NTU - probe faults, not pond physics). See `calibration.note` in `configs/twin.yaml`.

Every table above was generated by `scripts/build_report.py` reading `outputs/` and `data_cache/manifest.csv`. No figure in this file was typed by hand.
