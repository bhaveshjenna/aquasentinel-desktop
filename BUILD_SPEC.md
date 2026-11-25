# AquaSentinel — Build Spec

An aquaculture water-quality **monitoring and action-suggestion system** with a live-stream
Streamlit dashboard. A physics-lite **pond digital twin** generates interactive water chemistry;
a **perception stack** (LSTM forecasting + HDBSCAN regime discovery + SHAP explanation) reads it;
an **agent** proposes and applies actions through safety guardrails; the twin responds so the
dashboard shows the *effect* of each action. The perception models are **pretrained with
federated learning across many independent ponds/sites, using ample real water-quality data
collected online** — each site trains its own local model, coupled only through weight averaging.
Everything runs locally and offline *after* a one-time online data-collection step.

**Purpose of this document:** an execution plan for Claude Code. Read it fully before writing
code. Build in the phase order of Section 13 — get a runnable dashboard early, then layer ML.

**Scope & honesty note (put this in the README too):** the interactive pond is a *simulation*.
The real online data is continuous water-quality monitoring from rivers/streams/lakes, used as an
ample, realistic training source for the forecasting and regime models — it is a **proxy**, and
public monitoring sites are **not** fish ponds (aeration, feeding, stocking density differ), so
the stream→pond domain gap is stated, not hidden. We use only the **continuous sensor channels**
and **unsupervised** structure from real data; we do **not** consume any dataset's provided
quality *labels* (those are often scripted thresholds). Every performance claim is a *relative*
comparison — agent vs. baselines inside the twin, federated vs. independent-local vs. central on
held-out real sites — never a real-world guarantee. Twin coefficients and risk bands are
calibratable defaults (calibrate against FAO pond aquaculture, Boyd, APHA), not measured facts.
The agent's contribution is a safety-guardrailed control loop, evaluated for zero unsafe actions
and time-in-risk against baselines — not a claim of learned water chemistry.

---

## 1. What the system does, end to end

1. **Offline, once:** collect ample real water-quality time series from public online APIs — many
   monitoring sites, each treated as an independent **pond/client** (Section 8).
2. **Offline, once:** pretrain the LSTM forecaster + risk model via **federated learning** across
   those independent per-pond clients; compare against independent-local and central baselines;
   fit the HDBSCAN regime model. Cache weights to `models/` (Section 9).
3. **Live:** a **scenario** is chosen on the dashboard (e.g. "nighttime DO crash").
4. The **digital twin** steps forward in simulated time, producing true water-quality state; the
   **sensor simulator** samples it (noise, dropouts) → a "live" stream.
5. **Perception** runs each tick: LSTM **forecasts** the next horizon; HDBSCAN assigns the current
   **regime** and flags anomalies; the **risk model** scores risk and **SHAP** explains the drivers.
6. The **agent** perceives state+forecast+risk+regime, proposes an action, checks **guardrails**,
   and applies the allowed action to the twin.
7. The twin responds; the dashboard shows the resulting trajectory **against a no-action
   counterfactual**, plus a running scorecard vs. baselines, and lets you switch the driving
   weights between **federated / independent-local / central**.

---

## 2. Component roles (each must earn its place — no bolted-on buzzwords)

| Component | Concrete job | Honest framing |
|-----------|--------------|----------------|
| **Data acquisition** | Pull ample real multivariate water-quality time series from public APIs; one site = one pond/client | Real, keyless, cached; proxy for pond dynamics |
| **Digital twin** | Interactive ODE-ish simulator of DO/temp/pH/ammonia/nitrite/turbidity; responds to actions | The live environment; ground truth for the agent |
| **LSTM** | Multi-step, multivariate forecast of sensor channels | Forecasts continuous values — legitimately learned |
| **HDBSCAN** | Unsupervised discovery of operating regimes; noise points → anomalies | No labels used; regimes named post-hoc |
| **Risk model** | Maps current+forecast → risk from domain bands | Rule-informed risk, *not* "learned water quality" |
| **SHAP** | Attributes risk to input channels | Feeds the agent's rationale |
| **Federated learning** | Independent per-pond models, aggregated via FedAvg, no raw-data sharing | Measured vs. independent-local & central on real held-out sites |
| **Agent** | Perceive → plan → guardrail-check → act → observe | Contribution is the safety loop + preemptive action |

---

## 3. Repository layout

```
aquasentinel/
  configs/
    data.yaml            # online sources, site lists, channels, cache paths (Section 8)
    twin.yaml            # twin coefficients + risk bands (calibratable)
    scenarios.yaml       # scenario definitions (Section 6)
    sensors.yaml         # sampling rate, noise, dropout
    models.yaml          # LSTM + risk model hyperparams
    federated.yaml       # clients=sites, non-IID handling, FedAvg rounds (Section 9)
    agent.yaml           # action space, guardrails, planner mode, cost weights
  src/
    data/
      sources/
        usgs.py          # USGS NWIS Instantaneous Values client (keyless)
        ukea.py          # UK Environment Agency Hydrology API client (keyless)
        kaggle_pond.py   # OPTIONAL aquaculture pond IoT datasets (needs kaggle token)
      collect.py         # orchestrates collection -> cache -> manifest (Section 8)
      harmonize.py       # map source params -> common channel schema; resample; QC
      registry.py        # loads data.yaml; yields per-site client datasets
    twin/
      pond.py            # PondTwin: state, step(action) -> state (Section 5)
      dynamics.py        # DO/temp/pH/ammonia/nitrite/turbidity terms
      scenarios.py       # loads scenarios.yaml, applies perturbation schedules
      calibrate.py       # fit twin defaults to real-data statistics (Section 5)
    sensors/
      stream.py          # SensorStream: samples twin, adds noise/dropouts, yields ticks
    perception/
      forecaster.py      # LSTMForecaster: fit/predict horizon (PyTorch)
      regimes.py         # RegimeModel: HDBSCAN fit + assign + anomaly flag
      risk.py            # RiskModel (LightGBM) + risk bands
      explain.py         # SHAP wrapper over the risk model
    federated/
      client.py          # PondClient: owns one site's data + its own local model
      simulate.py        # FedAvg over independent clients; independent-local + central baselines
    agent/
      policy.py          # planners: reactive / predictive(MPC-lite) / llm (optional)
      guardrails.py      # hard constraints, rate limits, clamps, dosing caps
      actuators.py       # action space -> twin action mapping
      loop.py            # perceive -> plan -> check -> act -> observe
    eval/
      baselines.py       # do-nothing, static-threshold, oracle
      metrics.py         # forecast skill, DBCV, time-in-risk, interventions, unsafe
      run_eval.py        # scenario x seed x controller + federated comparison -> outputs/
    common/
      runtime.py         # seeds, thread caps, device=cpu, determinism
      schema.py          # dataclasses: State, Observation, Forecast, Action, Decision
  data_cache/            # downloaded raw + harmonized parquet per site (git-ignored)
  models/                # pretrained federated weights + regime model
  outputs/               # eval CSVs — source of truth for every reported number
  dashboard/
    app.py               # Streamlit entry (Section 12)
    panels/              # timeseries, regime, risk_shap, agent, counterfactual, scorecard
  tests/
  scripts/
    collect_data.py      # Phase 1 online collection (one-time, cached)
    pretrain_federated.py# Phase 6 federated pretraining -> models/
  README.md
```

---

## 4. Core data contracts (`src/common/schema.py`)

Dataclasses used everywhere (twin, real data, dashboard, eval share these types):

- `State`: timestamp + channels — `do, temp_c, ph, tan` (total ammonia N), `nh3` (un-ionized,
  derived), `no2, turbidity, conductivity`, plus latent twin vars (`algae, biomass, alkalinity`)
  not exposed to sensors.
- `Observation`: sensor- or API-sampled `State`; missing channels = `NaN` (real sites vary in
  which channels they report — carry a per-channel availability mask).
- `Forecast`: per-channel arrays over horizon `H`, optional prediction intervals.
- `Action`: `aerator ∈ [0,1]`, `exchange ∈ {0,small,large}`, `feed ∈ {skip,reduced,normal}`,
  `lime ∈ {0,dose}`, `salt ∈ {0,dose}`; `no_op` = defaults.
- `Decision`: proposed + applied `Action`, `risk_score`, `regime`, `anomaly`, `shap_top`,
  `guardrail_events`, `rationale`.

Real data and twin data both land as sequences of `Observation`, so the forecaster/regime/risk
code is identical whether the input is a real site or the live twin.

---

## 5. The pond digital twin (`src/twin/`)

The interactive environment — actions must produce visible, physically-plausible responses.
Discrete-time Euler steps (dt configurable, e.g. 5 min); all coefficients in `twin.yaml`, marked
**calibratable defaults**. Dynamics (`dynamics.py`), warm-water-pond defaults to tune:

- **Temperature**: diurnal sinusoid (config amplitude/phase); heatwave offset; exchange nudges
  toward inflow temp.
- **DO**: `dDO/dt = k_a·(DO_sat(T)−DO) + P_max·light(t)·algae − R_20·θ^(T−20)·(biomass+algae)
  − BOD + k_aer·aerator`. Photosynthesis is zero at night → the dawn DO sag.
- **Ammonia (TAN)**: `feed_excretion − nitrification − uptake`; exchange dilutes toward inflow;
  derive un-ionized `NH3` from TAN via pH+temperature (toxic form the risk model watches).
- **Nitrite (NO2)**: nitrification intermediate; salt/chloride dosing mitigates its toxicity.
- **pH**: rises daytime (photosynthesis), falls at night (respiration), buffered by alkalinity;
  lime dosing raises it (rate-limited). Bloom amplifies the swing.
- **Turbidity/algae**: bloom scenarios grow algae → pH swing + a die-off DO crash.

`PondTwin.step(action) -> State`. `scenarios.py` injects scheduled perturbations (Section 6).

**Calibration (`calibrate.py`):** fit the twin's baseline means, diurnal amplitudes, and noise to
statistics computed from the collected real data (Section 8) so the simulation is realistic. Store
the calibrated values back into `twin.yaml`. This is the bridge between real data and the twin.

**Risk bands (defaults in `twin.yaml`, calibratable):** DO critical `<2`, warning `<4`, healthy
`5–12`, supersaturation `>15`; pH healthy `6.5–9`, critical `<5.5`/`>9.5`; un-ionized NH3 warning
`>0.02`, critical `>0.05` mg/L; temp window default `26–32`°C. Cite sources in config comments.

---

## 6. Scenarios (`configs/scenarios.yaml`)

Each = initial state + perturbation schedule + fixed seed; the dashboard's selector lists them:
`healthy_baseline`, `night_do_crash`, `overfeeding_ammonia`, `algal_bloom`, `heatwave`,
`aerator_failure`, `ph_swing`, `sensor_fault`. Each individually runnable and reproducible.

---

## 7. Perception stack (`src/perception/`)

Trained on the collected real data (Section 8), run live on the twin stream.

- **LSTMForecaster** (PyTorch, CPU): last `W` steps of the (imputed, masked) multivariate stream →
  next `H` steps per channel. Metrics: per-channel, per-horizon RMSE/MAE. Save/load weights = the
  federated checkpoint.
- **RegimeModel** (`hdbscan`): fit on windowed feature vectors from real + twin data;
  `approximate_predict` at run time; noise label → **anomaly**. Report **DBCV** + silhouette; name
  regimes post-hoc from centroids.
- **RiskModel** (LightGBM): features = current channels + LSTM forecast summaries (min forecast DO
  over H, max forecast NH3, …); target = risk class from twin bands on the *forecast* horizon.
  Explicitly rule-derived.
- **explain.py** (SHAP `TreeExplainer`): top risk drivers per tick → dashboard + agent rationale.

---

## 8. Online data acquisition (`src/data/`, `scripts/collect_data.py`) — "ample data online"

Collect real, high-frequency, multi-site water-quality time series so that **each monitoring site
is an independent pond/client** with its own real distribution (naturally non-IID across geography
and season). One-time online step; everything downstream reads the **cache**. All primary sources
are **keyless, no registration**.

**Primary sources (implement `usgs.py`, `ukea.py`):**

- **USGS NWIS Instantaneous Values** — `https://waterservices.usgs.gov/nwis/iv/?format=json`.
  Keyless. ~15-min readings, thousands of US sites. Parameter codes: `00010` water temp (°C),
  `00300` dissolved oxygen (mg/L), `00400` pH, `00095` specific conductance, `63680` turbidity.
  Query `sites=<no>&parameterCd=00010,00300,00400,00095,63680&startDT=&endDT=` (history back to
  2007 for many params; some provisional params limited to ~120 days). Enumerate qualifying sites
  via the **site service** (`/nwis/site/?...&parameterCd=00300&hasDataTypeCd=iv`) or the
  WaterQualityWatch site list — pick sites that actually report DO+pH+turbidity.
- **UK Environment Agency Hydrology API** — `https://environment.data.gov.uk/hydrology`. Keyless,
  ~8000 stations, 15-min. **Richer for aquaculture**: DO (mg/L), **Ammonium (mg/L)**,
  **Nitrate (mg/L as N)**, Temperature, pH, Turbidity, Conductivity, Chlorophyll, Blue-Green Algae.
  Use its stations → measures → readings endpoints. This is the source that supplies the nutrient
  channels USGS lacks. Respect its fair-use guidance (request only the resolution/window needed).

**Optional supplement (`kaggle_pond.py`, needs a Kaggle token):** public aquaculture pond IoT
datasets for pond-specific realism. Use only their continuous channels, never their labels.
Document license; keep optional so the core stays keyless.

**Collection pipeline (`collect.py` + `harmonize.py`):**

1. Read `data.yaml`: source list, per-source site IDs (target **20–50 sites total** = many ponds),
   channels, date window (aim for **several months at 15-min** → thousands+ rows/site = ample).
2. Fetch per site with retries + backoff; save raw JSON and a harmonized **parquet** per site to
   `data_cache/`. Never re-download if cached (idempotent).
3. `harmonize.py`: map each source's parameters onto the common channel schema (Section 4),
   resample to a common cadence, forward-fill short gaps with a mask, QC (drop impossible values,
   flag flatlines/detection-limit zeros), and record a per-site **channel-availability mask**
   (real sites differ in which channels exist — this drives realistic missingness + non-IID).
4. Write `data_cache/manifest.csv`: site, source, lat/lon, channels present, row count, date range,
   license. Every later "number of sites / rows" is read from this file, not transcribed.

**Honesty:** rivers/streams ≠ ponds. State the domain gap. The real data buys realistic
multivariate dynamics, ample volume, and genuine cross-site heterogeneity for the federated
study; the twin supplies the pond-specific interactive control the agent acts on.

---

## 9. Federated learning (`src/federated/`, `scripts/pretrain_federated.py`) — independent models per pond

The core requirement: **each pond learns its own model independently; federation is the only thing
that couples them.**

- **`PondClient`** (`client.py`): owns exactly one site's cached data (Section 8) and its **own
  independent model instance** (LSTM forecaster + optional risk model). It exposes `local_train()`,
  `get_weights()`, `set_weights()`, `evaluate()`. A client never sees another client's raw data.
- **`simulate.py`** runs three regimes over the same set of clients so any difference is a property
  of the data/coupling, not the code:
  - **independent-local** — each client trains only on its own site; models never share anything.
    This is the "independent models learning from different ponds" baseline.
  - **federated (FedAvg)** — each round: every client does `E` local epochs on its own data; the
    server averages weights; broadcast back. Custom, in-process, deterministic, CPU-cheap (no
    network). Handle non-IID naturally — real sites already differ; also support weighting the
    average by client sample count.
  - **central** — pool all sites' data and train one model (upper-reference; violates the privacy
    premise, for comparison only).
- **Evaluation** (into `outputs/federated_comparison.csv`): per-channel forecast RMSE/MAE on
  **held-out real sites** (leave-some-sites-out) for independent-local vs. federated vs. central.
  Target the honest claim: a **data-poor pond** benefits from federation vs. training alone,
  without sharing raw data — quantified, scoped to these real sites.
- `pretrain_federated.py` writes the chosen (federated) weights + regime model to `models/` for the
  dashboard; the dashboard's model-source switch loads independent-local / federated / central to
  show the difference live.
- **Optional stretch:** DP-FedAvg (clip + noise) with a proper RDP/accountant-based ε; a Flower
  (`flwr`) implementation as an alternative to the custom loop.

---

## 10. The agent (`src/agent/`)

Loop (`loop.py`): **perceive → plan → guardrail-check → act → observe**, one decision per tick.

- **Action space** (`actuators.py`): the `Action` in Section 4, mapped to twin effects.
- **Planners** (`policy.py`), selectable in `agent.yaml`:
  - `reactive` — utility rules on current + forecast (transparent baseline-plus).
  - `predictive` (default) — **MPC-lite**: for each candidate action, roll the forecaster forward
    `H` steps under that action, score `cost = w_risk·time_in_risk + w_cost·intervention +
    w_stab·volatility`, pick argmin. Genuinely agentic (plans, uses tools, closes the loop),
    explainable.
  - `llm` (optional, offline via Ollama or an API key) — receives structured state + forecast +
    SHAP + candidate evaluations, selects + justifies in language; **still passes through
    guardrails**; never emits raw actuator commands unchecked.
- **Guardrails** (`guardrails.py`), hard/always-on: reject/clamp any action predicted to push a
  channel into **critical**; dosing caps + per-action rate limits; no large exchange unless inflow
  known-better; emergency DO-critical → aerator max. Log every proposed-vs-applied delta;
  **unsafe-applied must stay 0**.

---

## 11. Evaluation (`src/eval/`) — what makes it semester-worthy

Write every number to `outputs/`:

- **Forecasting on real data**: per-channel/horizon RMSE/MAE; **independent-local vs. federated vs.
  central** on held-out sites (Section 9) — the federated headline.
- **Regimes**: DBCV, silhouette; anomaly-detection recall on injected `sensor_fault`/failure events.
- **Agent** (in the twin): controllers `do_nothing`, `static_threshold`, `agent(predictive)`,
  `oracle`; per scenario report cumulative **time-in-risk**, **intervention count + cost**,
  **unsafe-applied (target 0)**; agent vs. static-threshold vs. do-nothing with **Wilcoxon
  signed-rank** across seeds (report `p`, `n` — never a bare mean).
- **Ablations**: agent with/without LSTM forecast; risk with/without forecast features; twin driven
  by federated vs. independent-local weights.

`run_eval.py` regenerates the tables; README numbers are read from `outputs/` at build time.

---

## 12. Dashboard (`dashboard/app.py`) — the deliverable they watch

Streamlit, live-updating via `st.session_state` + `st.fragment(run_every=...)` (or a play/pause
loop). Plotly.

**Sidebar:** scenario selector; play/pause/step + speed; agent on/off; planner mode
(reactive/predictive/llm); **model source (federated / independent-local / central)**; seed.

**Main panels (`dashboard/panels/`):**
- **Live time series** — DO, temp, pH, NH3, NO2, turbidity, each with **LSTM forecast overlay** and
  safe/warning/critical bands shaded.
- **Regime + anomaly** — current HDBSCAN regime + anomaly indicator.
- **Risk + SHAP** — risk gauge + top SHAP drivers this tick.
- **Agent** — proposed action, rationale (forecast + SHAP + regime), guardrail status, applied
  action, scrolling intervention log.
- **Counterfactual** — a **forked shadow twin** seeded identically with the agent disabled;
  agent-trajectory vs. no-action side by side. This is the money shot — make it prominent.
- **Scorecard** — running time-in-risk, interventions, unsafe-applied, vs. static-threshold.

---

## 13. Build order (MVP-first; each phase ends runnable)

### Phase 0 — skeleton
- [ ] Repo layout, `common/runtime.py`, `schema.py`, configs stubbed.

### Phase 1 — online data collection (one-time, cached)  ← do early so real data is ready
- [ ] `usgs.py` + `ukea.py` clients; `collect.py` + `harmonize.py`; `scripts/collect_data.py`.
- [ ] Populate `data.yaml` with 20–50 qualifying sites. **Milestone: `data_cache/manifest.csv` with ample rows across many sites.**

### Phase 2 — twin + stream + dashboard MVP
- [ ] `PondTwin` (DO/temp/pH/ammonia) + `calibrate.py` to real-data stats; `healthy_baseline` + `night_do_crash`.
- [ ] `SensorStream`; Streamlit app: scenario selector, play/pause, live time-series with risk bands. **Milestone: watch a DO crash live.**

### Phase 3 — reactive agent + counterfactual
- [ ] `actuators`, `guardrails`, `policy.reactive`, agent loop, shadow-twin counterfactual, scorecard. **Milestone: agent visibly rescues the DO crash vs. no-action.**

### Phase 4 — perception ML (trained on real data)
- [ ] `LSTMForecaster` + forecast overlay; `RegimeModel` (HDBSCAN) + regime/anomaly; `RiskModel` + SHAP. **Milestone: forecast, regime, explained risk on the dashboard.**

### Phase 5 — predictive agent
- [ ] `policy.predictive` (MPC-lite). **Milestone: agent acts preemptively before DO hits critical.**

### Phase 6 — federated learning (independent ponds)
- [ ] `PondClient`; `simulate.py` with independent-local / federated / central; `pretrain_federated.py` → `models/`; dashboard model-source switch. **Milestone: `outputs/federated_comparison.csv` showing federated vs. independent-local on held-out sites.**

### Phase 7 — full scenarios + evaluation
- [ ] Remaining scenarios; `run_eval.py` grid + Wilcoxon; ablations. **Milestone: complete `outputs/`.**

### Phase 8 — polish + optional stretch
- [ ] LLM planner (optional), DP-FedAvg / Flower, tests, README, screenshots. **Milestone: submission-ready.**

---

## 14. Tech stack & conventions

- Python 3.11; `requests`/`httpx` (data collection), PyTorch (CPU), `hdbscan`, `shap`, `lightgbm`,
  `streamlit`, `plotly`, `pandas`, `pyarrow`, `numpy`, `pyyaml`, `scipy`. Optional: `flwr`,
  `opacus`/`dp-accounting`, `ollama`, `kaggle`.
- **Online once, offline after.** Data collection and any model download are **separate, logged,
  cached** steps (`scripts/collect_data.py`). Training, inference, eval, and the dashboard run
  fully offline against `data_cache/` and `models/`. Never fetch from the network at dashboard runtime.
- **Independent-by-default federation.** A `PondClient` never reads another client's raw data; only
  weights cross the boundary. Keep independent-local, federated, and central as three switchable
  regimes over the same clients.
- **Determinism**: one global `SEED` (numpy/torch/lightgbm); twin noise seeded per scenario;
  reproducible on CPU. No GPU required; keep memory modest.
- **Config over code**: sources/sites, twin coefficients, model/agent/federated settings live in
  `configs/*.yaml`.
- **File-driven numbers**: every value in the README/tables is read from `outputs/` or
  `data_cache/manifest.csv` at build time, never transcribed.
- **Honesty of claims** (README "Supported / Not claimed"): supported — the safety loop (zero
  unsafe actions), relative agent-vs-baseline results in the twin, federated-vs-independent-local
  forecast comparison on real held-out sites; not claimed — real-pond validity, that risk reflects
  learned chemistry, that stream data equals pond data, average-case guarantees outside the tested
  scenarios.
- **Tests**: twin sanity (aeration raises DO, exchange lowers TAN), guardrail units (unsafe action
  always clamped), collector harmonization (schema + mask correctness), forecaster shape/finiteness,
  eval reproducibility.
