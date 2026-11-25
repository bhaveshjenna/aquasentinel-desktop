# AquaSentinel

**Safe Predictive Control and Federated Forecasting for Aquaculture Water Quality Simulation** — a water-quality
monitoring and action-suggestion system for fish ponds, with a native desktop operator console.

### Project & Team Attribution

- **Institution:** Department of Computer Science and Engineering, Birla Institute of Technology (BIT), Mesra, Ranchi
- **Project Guide:** Dr. Akriti Nigam
- **Team Members:**
  - Bhavesh Jenna (`BTECH/10796/23`)
  - Mihika Lakra (`BTECH/10580/23`)
  - Rishi Talluri (`BTECH/10829/23`)

---

- A physics-lite **pond digital twin** simulates oxygen, temperature, pH, ammonia, nitrite and
  turbidity, and responds to actions.
- A **perception stack** reads it: an LSTM forecaster, an HDBSCAN novelty flag, a LightGBM risk
  model and SHAP explanations.
- An **agent** proposes actions with model-predictive planning through an action-conditioned
  world model, and every proposal passes **hard safety guardrails** before it reaches the pond.
- A **shadow pond** runs alongside with the agent switched off, so every action's effect is shown
  against what would have happened without it.
- The forecaster is trained by **federated learning across real fish ponds**: each pond trains on
  its own data and only model weights are shared.

Everything runs locally and offline after a one-time data download.

---

## Read this before anything else

**The real data is real fish ponds, from one region.** Catfish aquaponics ponds on local farms in
Nsukka, Enugu State, Nigeria — the *Sensor Based Aquaponics Fish Pond Datasets* (Kaggle,
CC BY 4.0), published with Udanor et al., *Data in Brief* (2022), doi:10.1016/j.dib.2022.108400.
Eleven pond files were logged by low-cost ESP32 probes every ~20 s, June–October 2021. **7 ponds
pass quality control** (83,825 rows at 15-minute cadence), each with dissolved oxygen,
temperature, pH, total ammonia nitrogen and turbidity.

**The probes are low-cost and the data is rough.** Every pond logger is offline for 37–55 % of its
timeline; disconnected-probe codes (−127 °C), pH below 3 and ammonia at 10¹¹ are removed by QC;
nitrate is unusable (the column is a raw 0–4095 ADC count) and is not used. Ponds 8, 10, 11 and 12
are excluded (mostly empty, or unreadable timestamps). See `configs/data.yaml` and
`src/data/sources/kaggle_pond.py`.

**The pond the agent controls is a simulation.** Actions cannot be tried on the real ponds, and the
dataset records no actuator commands. `src/twin/` is a calibratable model of a warm-water earthen
pond with hand-set defaults (FAO, Boyd). A fit of its noise parameters to the real ponds was
computed and **deliberately not applied**: the fitted magnitudes reflect probe faults, not pond
physics (recorded in `configs/twin.yaml` under `calibration`).

**No dataset quality labels are used**, and **every performance claim is relative** — agent versus
baselines inside the twin, federated versus independent-local versus central on held-out real
ponds. Nothing here is a real-world guarantee.

**History.** An earlier version trained on 42 US/UK river monitoring sites and had an optional LLM
planner. Both were removed; that version is kept in a separate copy of the project folder.
`STATUS.md` and `EXPLAINER.md` describe that earlier version and are marked as such.

---

## Results

`RESULTS.md` is **generated** from `outputs/*.csv` by `scripts/build_report.py`; every table in it
is read from a file. The figures below are a hand-copied summary — if they disagree with
`RESULTS.md`, `RESULTS.md` is right.

- **Safety: 0 unsafe actions** applied by any guardrailed controller across 8 scenarios ×
  5 controllers × 10 seeds. With the guardrails removed, the learned planner applies **1,057**.
- **Agent vs baselines:** significantly lower dissolved-oxygen time-out-of-band in **17 of 24**
  paired Wilcoxon comparisons; the 6 ties are in the two scenarios where no controller has any risk.
  It is **more expensive to operate** than a simple aerator thermostat in 7 of 8 scenarios.
- **Federated learning** (5 ponds train, 2 held out): held-out DO forecast error **1.77 mg/L**
  federated, against 1.92 for each pond training alone and 1.79 for pooling all data centrally.
  Of the two data-poor ponds (8 % of their history), one improves sharply under federation
  (−58 %) and one gets slightly worse (+3.5 %). A small study, presented as such.
- **Results that do not flatter the system**, all reported in `RESULTS.md`:
  - The LSTM residual correction makes control *worse* (10.83 h vs 8.79 h of DO time-out-of-band
    without it). It ships enabled deliberately; `predictive.use_forecaster: false` turns it off.
  - The HDBSCAN anomaly flag catches every injected probe fault but also fires on **82 %** of
    normal readings — not usable as an alarm on its own.
  - HDBSCAN finds no regime taxonomy (DBCV ≈ 0): water-quality windows form a continuum.
  - Absolute forecast error is large because the probes are noisy; the claims are comparisons.

## Documents

| File | What it is |
|---|---|
| `report/AquaSentinel-Report.pdf` | Comprehensive system architecture & research report |
| `report/AquaSentinel-Code-Walkthrough.pdf` | Architectural walkthrough detailing key algorithmic components |
| `RESULTS.md` | Benchmark evaluation tables and ablation results, generated from `outputs/` |

All benchmark metrics in `RESULTS.md` are computed directly from `outputs/` at build time, so re-running the evaluation grid updates the documentation automatically.

---

## Quick start

### Environment & Dependencies

Install all core ML, native GUI (`PySide6`, `pyqtgraph`), reporting (`python-docx`, `python-pptx`), and test (`pytest-qt`) dependencies in one command:

```bash
pip install -r requirements.txt
```

### Pipelines & Pre-training

```bash
python scripts/collect_data.py          # ONE-TIME, online: downloads the pond dataset (Kaggle token)
python scripts/pretrain_federated.py    # federated forecaster + anomaly + risk models   (~1 min)
python scripts/pretrain_world_model.py --rounds 12 --episodes 30   # world model   (~1 min)
python -m src.eval.run_eval             # full evaluation grid + ablations   (~20 min)
python scripts/build_report.py          # RESULTS.md
```

### Launching the Frontends

Two one-click Windows launchers are provided:
- **`run_desktop.bat`** — launches the primary native PySide6 + pyqtgraph desktop operator console (`python -u -m desktop`).
- **`run_dashboard.bat`** — launches the Streamlit web dashboard backup at `http://localhost:8501` (useful on Windows machines where AMD/integrated display drivers block Qt window compositing).

Both `.bat` launchers automatically discover the active Python virtual environment (local, parent, or `%VIRTUAL_ENV%`) and set `NUMBA_DISABLE_JIT=1` and `KMP_DUPLICATE_LIB_OK=TRUE` to avoid LLVM/OpenMP hangs in OneDrive-synced directories.

Alternatively, launch either directly from terminal:
```bash
python -u -m desktop                    # native Qt desktop console
streamlit run dashboard/app.py          # Streamlit web dashboard
```

Rebuilding the documentation reports:

```bash
python scripts/make_figures.py && python scripts/make_report_pdf.py   # project report PDF
python scripts/make_code_walkthrough_pdf.py                          # code walkthrough PDF
```

PDF printing needs Node and Chrome or Edge (`scripts/html_to_pdf.mjs`). The desktop app runs
without `models/` (it falls back to the rule-based risk bands and says so).

**Desktop console:** `Space` play/pause · `S` step one decision · `R` reset. Views: Live dashboard,
Counterfactual, Agent, Scorecard. A Streamlit version of the same dashboard is kept in
`dashboard/` (`streamlit run dashboard/app.py`).

### Verification & Testing

Run the full automated test suite (137 tests covering digital twin physics, guardrail enforcement, sensor streams, ML perception models, federated learning, and both GUI frontends):

```bash
pytest tests/ -p no:cacheprovider -v
```

---

## How it fits together

```
 OFFLINE, ONCE                                LIVE, EVERY 5 SIMULATED MINUTES
 Kaggle pond CSVs (Nsukka)                    PondTwin ─► SensorStream ─► noisy reading
   └► kaggle_pond loader ─► harmonize/QC                    │
        └► data_cache/*.parquet (7 ponds)                   ▼
             └► PondClient × 7                   Perception: LSTM forecast · HDBSCAN flag
                  └► FedAvg ─► models/                       · LightGBM risk · SHAP
                       └► world model (encoder              │
                          warm-started, trained             ▼
                          on twin rollouts)        Planner (MPC over 14 actions,
                                                            world-model rollouts)
                                                            ▼
                                                 GUARDRAILS ─► applied action ─► PondTwin
                                                 Shadow pond: same seed and noise, no actions
```

| Component | Job | Honest framing |
|---|---|---|
| Data | 7 real ponds, one pond = one federated client | Real pond data from low-cost probes; ~half of each timeline is outage |
| Digital twin | Interactive DO / temp / pH / TAN / NO₂ / turbidity simulator | The agent's environment — not a validated model of a real farm |
| LSTM forecaster | Multi-step, multi-channel forecast | Trained on real ponds, on **changes** with fixed scales |
| World model | Predicts the pond's response to a candidate action | Real data has no actuator log, so actions are learned in the twin; encoder from the federated forecaster |
| HDBSCAN | Novelty flag (noise label) | Found no regime taxonomy; high false-alarm rate on this data |
| Risk model | Current readings + forecast → safe / warning / critical | **Rule-informed**: its target comes from `twin.yaml` |
| SHAP | Per-decision attribution of the risk score | Same explanation for operator and agent |
| Federated learning | FedAvg across ponds, weights only | Measured against independent-local and central on held-out ponds |
| Agent | perceive → plan → guardrail-check → act | The contribution is the safety loop and preemption |

## The agent

One plan every 15 simulated minutes; the safety layer runs on **every** 5-minute reading.

- **Cadence & Impulse Actuation:** The policy re-plans on a 15-minute cadence. Between decision points, continuous actuators (`aerator`, `exchange`, `feed`) remain held under 5-minute `safety_only()` supervision. Discrete chemical impulse treatments (`lime`, `salt`) execute once on the initial decision tick and are immediately reset to `none` on intermediate ticks so single-shot doses never repeat across the cadence window.
- **Planners** (`configs/agent.yaml`): `reactive` (transparent rules over readings and forecast)
  and `predictive` (default: scores 14 curated actions through the world model; cost = time at
  risk + 3 × time critical + operating cost + volatility + lost-growth penalty).
- **Guardrails** (`src/agent/guardrails.py`), in order: dosing caps and minimum intervals → rate
  limits → contextual blocks (no lime into alkaline water, no feeding at low DO, no large exchange
  into worse or thermally shocking water) → reject actions predicted to cause a critical state →
  emergency override (DO critical now or forecast → aerator to max, feeding stopped). Thresholds
  are checked against the noisy probe shifted ~3σ in the unsafe direction.
- **Baselines:** `do_nothing` (deliberately unguarded reference), `static_threshold` (aerator
  thermostat — the bar to clear), `reactive`, `oracle` (reads the twin's hidden state; a reference
  heuristic, not an upper bound).

## Federated learning

`configs/federated.yaml`. One pond = one `PondClient`, which exposes only `local_train`,
`get_weights`, `set_weights` and `evaluate`; the server receives weights and a sample count.
Three regimes over the same ponds: **independent-local** (each pond alone), **federated** (FedAvg,
12 rounds × 2 local epochs, sample-weighted) and **central** (pooled; breaks the privacy premise,
reference only). Matched total training work, a shared initialisation, and 2 ponds held out that
no regime trains on. DP-FedAvg is implemented (`--dp`) but not run.

## Scenarios

`configs/scenarios.yaml`: `healthy_baseline` · `night_do_crash` · `overfeeding_ammonia` ·
`algal_bloom` · `heatwave` · `aerator_failure` · `ph_swing` · `sensor_fault`. A test asserts
each one produces the stress its name claims. `sensor_fault` corrupts the reading, not the water.

## Layout

```
configs/           data, twin, scenarios, sensors, models, federated, agent
src/
  common/          runtime (seeds, config), schema (State, Observation, Action, Decision)
  data/            sources/kaggle_pond (usgs, ukea kept but unused), collect, harmonize, registry
  twin/            pond, dynamics, scenarios, calibrate, risk_bands
  sensors/         stream
  perception/      forecaster, world_model, regimes, risk, explain
  federated/       client, simulate
  agent/           policy, guardrails, actuators, loop
  eval/            baselines, metrics, run_eval
desktop/           the PySide6 operator console (main UI)
dashboard/         the Streamlit version
scripts/           data, training, evaluation and document builders
report/            generated documents, architecture diagrams, and figures
tests/             137 tests across 11 modules (twin, sensors, data/QC, perception, federation, guardrails, agent, desktop, dashboard)
run_desktop.bat    one-click Windows launcher for native Qt desktop console
run_dashboard.bat  one-click Windows launcher for Streamlit web dashboard
```

## Conventions

- Python 3.11, CPU only. One global seed; the twin's noise is seeded per scenario.
- Online once (`scripts/collect_data.py`), offline after.
- All behaviour lives in `configs/*.yaml`.
- Every reported number comes from `outputs/`; documents read it at build time.
- Every controller comparison is a paired Wilcoxon test across seeds with `p` and `n`.

## Licences and attribution

- Pond data: *Sensor Based Aquaponics Fish Pond Datasets*, Kaggle
  (ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets), CC BY 4.0; Udanor et al.,
  *Data in Brief* 43 (2022) 108400.
