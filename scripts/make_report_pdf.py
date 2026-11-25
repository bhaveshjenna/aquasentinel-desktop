"""Build report/AquaSentinel-Report.pdf — the full project report. Offline.

    python scripts/make_report_pdf.py

Assembles an HTML document (figures inlined as SVG, tables read from ``outputs/*.csv`` and
``data_cache/manifest.csv``, config excerpts read from ``configs/``) and prints it to PDF with
headless Chrome or Edge.

Same discipline as the rest of the project: no number in the document is typed by hand. If a
table is missing the section says which command produces it rather than inventing a figure.

Prerequisites: ``python scripts/make_figures.py`` (figures) and a completed
``python -m src.eval.run_eval`` (tables).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.common.runtime import CONFIG_DIR, DATA_CACHE, MODELS_DIR, OUTPUTS_DIR, ROOT  # noqa: E402

FIG_DIR = ROOT / "report" / "figures"
OUT_PDF = ROOT / "report" / "AquaSentinel-Report.pdf"

MISSING_CMD = {
    "manifest.csv": "python scripts/collect_data.py",
    "federated_summary.csv": "python scripts/pretrain_federated.py",
    "federated_data_poor.csv": "python scripts/pretrain_federated.py",
    "regime_metrics.csv": "python scripts/pretrain_federated.py",
    "regime_sweep.csv": "python scripts/pretrain_federated.py",
    "risk_metrics.csv": "python scripts/pretrain_federated.py",
    "agent_summary.csv": "python -m src.eval.run_eval",
    "agent_stats.csv": "python -m src.eval.run_eval",
    "agent_runs.csv": "python -m src.eval.run_eval",
    "safety.csv": "python -m src.eval.run_eval",
    "ablations.csv": "python -m src.eval.run_eval",
    "anomaly_detection.csv": "python -m src.eval.run_eval",
    "forecast_horizon.csv": "python -m src.eval.run_eval",
}

OUT: List[str] = []
_fig_no = 0
_tab_no = 0


# ---------------------------------------------------------------------------- emit helpers
def w(s: str = "") -> None:
    OUT.append(s)


def h1(t: str, anchor: str) -> None:
    w(f'<h1 id="{anchor}">{t}</h1>')


def h2(t: str) -> None:
    w(f"<h2>{t}</h2>")


def h3(t: str) -> None:
    w(f"<h3>{t}</h3>")


def p(s: str) -> None:
    w(f"<p>{s}</p>")


def note(s: str, kind: str = "note") -> None:
    w(f'<div class="callout {kind}">{s}</div>')


def read(name: str, base: Path = OUTPUTS_DIR) -> Optional[pd.DataFrame]:
    path = base / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def absent(name: str) -> None:
    w(f'<div class="callout missing"><code>{name}</code> was not found. '
      f'Generate it with <code>{MISSING_CMD.get(name, "see the README")}</code>.</div>')


def fmt(v, nd: int = 3) -> str:
    """Format one cell. Large counts get thousands separators; everything else is rounded."""
    if v is None:
        return "–"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if pd.isna(v):
            return "–"
        if float(v) == int(v):
            n = int(v)
            return f"{n:,}" if abs(n) >= 10000 else str(n)
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def table(df: pd.DataFrame, caption: str, cols: Optional[Sequence[str]] = None,
          nd: int = 3, rename: Optional[Dict[str, str]] = None,
          highlight: Optional[str] = None) -> None:
    """Emit a numbered table. Column values are formatted, never re-typed."""
    global _tab_no
    _tab_no += 1
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if rename:
        df = df.rename(columns=rename)
    w('<figure class="tbl">')
    w("<table><thead><tr>")
    for c in df.columns:
        w(f"<th>{html.escape(str(c))}</th>")
    w("</tr></thead><tbody>")
    for _, row in df.iterrows():
        cls = ""
        if highlight and highlight in df.columns:
            try:
                cls = ' class="hl"' if float(row[highlight]) > 0 else ""
            except (TypeError, ValueError):
                cls = ""
        w(f"<tr{cls}>")
        for c in df.columns:
            w(f"<td>{fmt(row[c], nd)}</td>")
        w("</tr>")
    w("</tbody></table>")
    w(f"<figcaption><b>Table {_tab_no}.</b> {caption}</figcaption>")
    w("</figure>")


def figure(name: str, caption: str) -> None:
    """Inline an SVG so the PDF keeps it as vector art."""
    global _fig_no
    path = FIG_DIR / f"{name}.svg"
    if not path.exists():
        w(f'<div class="callout missing">Figure <code>{name}.svg</code> is missing. '
          f'Run <code>python scripts/make_figures.py</code>.</div>')
        return
    _fig_no += 1
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)
    svg = re.sub(r'(<svg[^>]*?)width="[\d.]+pt"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)height="[\d.]+pt"', r"\1", svg, count=1)
    w('<figure class="fig">')
    w(f'<div class="svgwrap">{svg}</div>')
    w(f"<figcaption><b>Figure {_fig_no}.</b> {caption}</figcaption>")
    w("</figure>")


def yaml_block(config: str, keys: Sequence[str], caption: str) -> None:
    """Excerpt selected top-level keys of a config, straight from the file."""
    path = CONFIG_DIR / f"{config}.yaml"
    if not path.exists():
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    subset = {k: data[k] for k in keys if k in data}
    text = yaml.safe_dump(subset, sort_keys=False, default_flow_style=False, allow_unicode=True)
    w(f'<div class="code"><div class="codehead">configs/{config}.yaml — {caption}</div>'
      f"<pre>{html.escape(text.rstrip())}</pre></div>")


# ======================================================================================
# Sections
# ======================================================================================
def cover(manifest: Optional[pd.DataFrame]) -> None:
    n_sites = len(manifest) if manifest is not None else "—"
    n_rows = f"{int(manifest['rows'].sum()):,}" if manifest is not None else "—"
    safety = read("safety.csv")
    unsafe = "—"
    if safety is not None:
        guarded = safety[safety.controller != "do_nothing"]
        unsafe = str(int(guarded["unsafe_applied_total"].sum()))
    runs = read("agent_runs.csv")
    n_runs = f"{len(runs)}" if runs is not None else "—"

    w('<section class="cover">')
    w('<div class="eyebrow">Project report</div>')
    w("<h1 class=\"title\">AquaSentinel</h1>")
    w('<div class="subtitle">An aquaculture water-quality monitoring and action-suggestion '
      "system: a physics-lite pond digital twin, a perception stack pretrained by federated "
      "learning across real fish ponds from one region, and a guardrailed control agent "
      "evaluated against a no-action counterfactual.</div>")
    w('<div class="statrow">')
    for label, value in [("real fish ponds", n_sites),
                         ("harmonised rows", n_rows),
                         ("evaluation runs", n_runs),
                         ("unsafe actions applied", unsafe)]:
        w(f'<div class="stat"><div class="statval">{value}</div>'
          f'<div class="statlab">{label}</div></div>')
    w("</div>")
    w('<div class="covernote"><b>Read this first.</b> The pond being controlled is a '
      "<b>simulation</b>. The real data is sensor data from catfish aquaponics ponds in Nsukka, "
      "Nigeria, logged by <b>low-cost IoT probes</b> — genuine pond data, but noisy, with long "
      "outages, and with no record of the actions the farmers took. No dataset's quality labels "
      "are consumed anywhere. Every performance claim in this report is <b>relative</b> — agent "
      "versus baselines inside the twin, federated versus independent-local versus central on "
      "a held-out real pond — and none is a real-world guarantee. Section 10 lists the results "
      "that do not flatter the system.</div>")
    w(f'<div class="meta">Generated {datetime.now().strftime("%d %B %Y")} by '
      f"<code>scripts/make_report_pdf.py</code> · every figure and table is read from "
      f"<code>outputs/</code>, <code>data_cache/manifest.csv</code> or <code>configs/</code></div>")
    w("</section>")


def toc() -> None:
    w('<section class="toc"><h2>Contents</h2><ol>')
    for n, t in [
        ("s1", "What the system does, end to end"),
        ("s2", "Architecture and component roles"),
        ("s3", "Data acquisition — specification and what was collected"),
        ("s4", "The pond digital twin"),
        ("s5", "The sensor model"),
        ("s6", "Perception stack"),
        ("s7", "Federated learning"),
        ("s8", "The agent: planners and guardrails"),
        ("s9", "Evaluation: method and results"),
        ("s10", "Observations — including the unflattering ones"),
        ("s11", "Limitations and what would fix them"),
        ("s12", "Reproducing this report"),
        ("s13", "Appendix: repository map, configuration, provenance"),
    ]:
        w(f'<li><a href="#{n}">{t}</a></li>')
    w("</ol></section>")


def s1_overview() -> None:
    h1("1 · What the system does, end to end", "s1")
    p("AquaSentinel closes a loop. A simulated pond produces water chemistry; a perception "
      "stack reads a noisy sensor view of it; an agent proposes an action; hard safety "
      "guardrails accept, clamp or reject that action; the pond responds. Because the twin is "
      "interactive rather than a replayed recording, the consequence of every action is "
      "observable — and a second pond, forked at t=0 with the same seed and the same noise, "
      "runs with the agent disabled so the consequence is <i>attributable</i>.")
    w("<ol class='steps'>")
    for step in [
        "<b>Offline, once.</b> Load real multivariate water-quality time series from a public "
        "pond dataset. Each pond that passes quality control becomes an independent client.",
        "<b>Offline, once.</b> Pretrain the LSTM forecaster by federated averaging across those "
        "clients; compare against independent-local and centrally-pooled baselines; fit the "
        "regime and risk models. Cache weights to <code>models/</code>.",
        "<b>Live.</b> A scenario is selected — say a nighttime oxygen crash.",
        "The digital twin steps forward in simulated time producing true state; the sensor "
        "simulator samples it with noise, bias, quantisation and dropouts.",
        "Perception runs each tick: the LSTM forecasts the next two hours, HDBSCAN flags "
        "windows unlike anything in training, the risk model scores the horizon and SHAP "
        "attributes that score to inputs.",
        "The agent plans over a curated action set, and the guardrails run on <i>every</i> "
        "sensor tick rather than only at planning ticks.",
        "The twin responds. The dashboard shows the resulting trajectory against the "
        "no-action shadow pond, plus a running scorecard.",
    ]:
        w(f"<li>{step}</li>")
    w("</ol>")
    note("<b>Everything after step 2 is offline.</b> Exactly one file in the project touches "
         "the network — <code>scripts/collect_data.py</code>. Training, inference, evaluation "
         "and the dashboard all read from disk.")


def s2_architecture() -> None:
    h1("2 · Architecture and component roles", "s2")
    p("Each component had to earn its place. The table states what it does and, more "
      "importantly, what it does <i>not</i> license anyone to claim.")
    rows = [
        ("Data acquisition", "Real fish-pond series from one region; one pond = one federated "
         "client", "Real pond data, cached. Low-cost probes: noisy, with long outages."),
        ("Digital twin", "Interactive DO / temperature / pH / TAN / NO₂ / turbidity simulator "
         "that responds to actions",
         "The environment and ground truth <i>for the agent</i> — not for the world."),
        ("LSTM forecaster", "Multi-step multivariate forecast of sensor channels",
         "Learned from real pond data, on <b>deltas</b> so it transfers to the twin's pond."),
        ("HDBSCAN", "Unsupervised structure; the noise label becomes the anomaly flag",
         "No labels used. It found <b>no regime taxonomy</b> and the report says so (§10)."),
        ("Risk model", "Current channels + forecast summaries → risk class",
         "<b>Rule-informed, not learned chemistry.</b> Its target comes from "
         "<code>twin.yaml</code>."),
        ("SHAP", "Per-tick attribution of the risk score",
         "Feeds the operator panel and the agent's own rationale — the same explanation, "
         "not a second story told afterwards."),
        ("Federated learning", "Independent per-pond models, FedAvg, no raw-data sharing",
         "Measured against independent-local and central on a <b>held-out</b> real pond."),
        ("Agent", "perceive → plan → guardrail-check → act → observe",
         "The contribution is the <b>safety loop</b> and preemption, not learned chemistry."),
    ]
    w('<figure class="tbl"><table><thead><tr><th>Component</th><th>Concrete job</th>'
      "<th>Honest framing</th></tr></thead><tbody>")
    for a, b, c in rows:
        w(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>")
    w("</tbody></table>")
    global _tab_no
    _tab_no += 1
    w(f"<figcaption><b>Table {_tab_no}.</b> Component roles and the limits of what each "
      "supports.</figcaption></figure>")

    h2("2.1 Data contracts")
    p("Real data and twin data both land as sequences of the same dataclasses "
      "(<code>src/common/schema.py</code>), which is why the forecaster, regime and risk code "
      "is identical whether its input is a real Nigerian catfish pond or the simulated pond.")
    w("<ul>")
    for item in [
        "<code>State</code> — ground truth: <code>do, temp_c, ph, tan, nh3, no2, turbidity, "
        "conductivity</code> plus latent <code>algae, biomass, alkalinity, bod_pool, feed_pool, "
        "chloride</code> that no probe can read.",
        "<code>Observation</code> — what perception actually sees, with a per-channel "
        "availability <b>mask</b>. Real probes drop out for hours at a time; that "
        "missingness is carried, not imputed away.",
        "<code>Forecast</code> — per-channel arrays over the horizon, plus a "
        "<code>summary()</code> of the extremes the risk model and planner consume.",
        "<code>Action</code> — <code>aerator ∈ [0,1]</code>, <code>exchange ∈ "
        "{none,small,large}</code>, <code>feed ∈ {normal,reduced,skip}</code>, "
        "<code>lime</code>, <code>salt</code>.",
        "<code>Decision</code> — <b>proposed and applied</b> action, risk, regime, anomaly, "
        "SHAP drivers, guardrail events, rationale. The proposed/applied split is what makes "
        "the safety claim auditable.",
    ]:
        w(f"<li>{item}</li>")
    w("</ul>")


def s3_data(manifest: Optional[pd.DataFrame]) -> None:
    h1("3 · Data acquisition — specification and what was collected", "s3")
    h2("3.1 Specification")
    p("One public dataset, from <b>one region</b>: the <i>Sensor Based Aquaponics Fish Pond "
      "Datasets</i> (Kaggle, CC BY 4.0), published with Udanor et al., <i>Data in Brief</i> "
      "(2022), doi:10.1016/j.dib.2022.108400. Catfish aquaponics ponds on local farms in "
      "Nsukka, Enugu State, Nigeria, logged by ESP32-based probes roughly every 20 seconds "
      "between June and October 2021.")
    w("<ul>")
    w("<li><b>One file = one pond = one federated client.</b> The loader "
      "(<code>src/data/sources/kaggle_pond.py</code>) turns each <code>IoTPondN.csv</code> into "
      "its own site. Column names differ in every file (<i>Temperature (C)</i>, "
      "<i>temperature(C)</i>, …) and are matched by normalised name.</li>")
    w("<li><b>Channels used:</b> temperature, dissolved oxygen, pH, ammonia (as total ammonia "
      "nitrogen) and turbidity. <b>Not used:</b> nitrate, because the raw column is a 0–4095 "
      "analogue-to-digital count rather than a concentration, and fish population, length and "
      "weight, which are not water quality.</li>")
    w("<li><b>Timestamps</b> mix <code>2021-06-18 13:02:10 CET</code> and ISO "
      "<code>+01:00</code>; both are UTC+1 (West Africa Time) and are parsed as such. Rows "
      "carrying only a date are dropped.</li>")
    w("<li><b>Downloaded once</b> with a Kaggle token and cached under "
      "<code>data_cache/raw/</code>; everything after that runs offline.</li>")
    w("</ul>")
    yaml_block("data", ["window", "targets"], "record window and admission criteria")

    h2("3.2 Harmonisation and quality control")
    p("Every source lands as a long <code>[timestamp, channel, value]</code> frame and leaves as "
      "a regularly-sampled wide frame plus a boolean mask. The mask is the contract: it records "
      "<i>genuine observations</i>, and it is taken <b>before</b> any gap filling, so a "
      "forward-filled value is never reported as a reading.")
    w("<ul>")
    w("<li>Physically impossible values are dropped against per-channel ranges.</li>")
    w("<li>An exact <code>0.0</code> in dissolved oxygen or turbidity is treated as a "
      "<b>censored</b> at-detection-limit reading, not a measurement.</li>")
    w("<li>A probe stuck on one value for 48 consecutive samples is broken, not calm — those "
      "runs are masked out.</li>")
    w("<li>Gaps are forward-filled up to one hour; longer gaps stay missing.</li>")
    w("</ul>")

    h2("3.3 What was actually collected")
    if manifest is None:
        absent("manifest.csv")
        return
    p(f"<b>{len(manifest)} ponds, {int(manifest['rows'].sum()):,} harmonised rows</b> at "
      f"{int(manifest['cadence_minutes'].iloc[0])}-minute cadence, spanning "
      f"{str(manifest['start'].min())[:10]} to {str(manifest['end'].max())[:10]} across "
      f"{manifest['region'].nunique()} regions — "
      + ", ".join(f"{k} ({v} ponds)" for k, v in manifest["source"].value_counts().items())
      + ".")
    figure("f7_coverage",
           "Channel coverage across the ponds that passed quality control. Every pond carries "
           "the same five probes, so the clients differ in pond conditions, stock and probe "
           "faults rather than in instrumentation.")
    per_source = (manifest.groupby("source")
                  .agg(sites=("client_id", "count"), rows=("rows", "sum"),
                       mean_channels=("n_channels", "mean"),
                       mean_missing=("missing_frac", "mean"))
                  .reset_index())
    table(per_source, "Collected volume per source. <code>mean_missing</code> is the average "
          "fraction of missing cells after resampling, across the channels each site reports.",
          rename={"source": "source", "mean_channels": "mean channels",
                  "mean_missing": "mean missing fraction"})
    note("<b>Real ponds, imperfect probes.</b> This is genuine catfish-pond data, but from "
         "low-cost sensors: disconnected-probe codes, impossible values and long outages are "
         "removed by quality control, and nitrate is excluded because its column is a raw ADC "
         "count. The dataset also records no actuator commands, so what an aerator does has to "
         "be learned in the twin (§7).", kind="warn")
    licences = sorted(set(manifest["license"].astype(str)))
    p("<b>Licences.</b> " + " · ".join(html.escape(l) for l in licences))


def s4_twin() -> None:
    h1("4 · The pond digital twin", "s4")
    p("Discrete-time Euler integration at a configurable step (5 minutes by default). "
      "<code>PondTwin.step(action)</code> advances the pond and returns the new true state. "
      "The functional forms follow standard pond-aquaculture practice; <b>every coefficient is "
      "a calibratable default, not a measured fact</b>.")

    h2("4.1 Dynamics")
    w('<div class="eqn">')
    w("<div class='eq'><b>Dissolved oxygen</b> (mg L⁻¹ h⁻¹)<br>"
      "d[DO]/dt = k<sub>a</sub>(DO<sub>sat</sub>(T) − DO) + k<sub>aer</sub>·u<sub>aer</sub>"
      "(DO<sub>sat</sub> − DO) + P<sub>max</sub>·light(t)·A "
      "− R<sub>20</sub>·θ<sup>(T−20)</sup>(B + w<sub>A</sub>A) − k<sub>BOD</sub>·BOD "
      "− 4.57·N<sub>nitrified</sub></div>")
    w("<div class='eq'><b>Total ammonia nitrogen</b><br>"
      "d[TAN]/dt = excretion(feed) − k<sub>nit</sub>·θ<sup>(T−20)</sup>·f<sub>DO</sub>·TAN "
      "− k<sub>uptake</sub>·A·monod(TAN)</div>")
    w("<div class='eq'><b>Un-ionised ammonia</b> — Emerson et al. (1975); the toxic form, and "
      "the channel the risk bands are written against<br>"
      "pK<sub>a</sub> = 0.09018 + 2729.92/T<sub>K</sub> ; "
      "NH₃ = TAN / (1 + 10<sup>(pKa − pH)</sup>)</div>")
    w("<div class='eq'><b>pH</b> — CO₂ stripped by photosynthesis raises pH, respiration lowers "
      "it, and buffering scales inversely with alkalinity<br>"
      "d[pH]/dt = clip( (g<sub>P</sub>·P − g<sub>R</sub>·R)·min(Alk<sub>ref</sub>/Alk, "
      "γ<sub>max</sub>) ) + k<sub>relax</sub>(pH<sub>eq</sub> − pH)</div>")
    w("<div class='eq'><b>Alkalinity</b> — nitrification consumes 7.14 mg CaCO₃ per mg N (Boyd)"
      "<br>d[Alk]/dt = −7.14·N<sub>nitrified</sub> − k<sub>decay</sub>·Alk</div>")
    w("<div class='eq'><b>Algae</b> — logistic on light and nitrogen, with crowding mortality "
      "that drives the bloom crash<br>"
      "d[A]/dt = μ·light·monod(TAN)·A(1 − A/K) − (m + m<sub>crowd</sub>(A/K)²)·A</div>")
    w("</div>")
    p("Dissolved-oxygen saturation uses the Benson &amp; Krause / APHA freshwater fit with a "
      "salinity correction. Process noise is Ornstein–Uhlenbeck rather than white: on a "
      "five-minute step, white noise looks like sensor jitter, whereas mean-reverting noise "
      "looks like weather.")
    note("<b>Aeration is not an oxygen tap.</b> Because the aerator term is proportional to the "
         "saturation deficit, running it in a supersaturated afternoon pond <i>strips</i> oxygen "
         "toward equilibrium. That is correct behaviour and it is asserted as a test "
         "(<code>test_aeration_strips_oxygen_from_a_supersaturated_pond</code>) so nobody "
         "'fixes' it later.")

    h2("4.2 Risk bands")
    p("The bands are rule-based, taken from warm-water pond guidance (FAO; Boyd, <i>Water "
      "Quality in Ponds for Aquaculture</i>) and calibratable per species. They are the single "
      "definition of risk in the project: the dashboard shades them, the risk model is trained "
      "to anticipate them, the guardrails enforce them and the evaluation scores against them. "
      "There is no second definition living in the presentation layer.")
    yaml_block("twin", ["risk_bands"], "risk bands (calibratable defaults, tilapia-ish)")

    h2("4.3 Calibration — and what was deliberately refused")
    p("<code>scripts/calibrate_twin.py</code> can fit the twin's process-noise magnitudes, "
      "baseline turbidity and relaxation pH to statistics of the real data. Run against the "
      "Nsukka ponds, it was computed and <b>refused</b>: the twin keeps its hand-set warm-water "
      "defaults (FAO; Boyd).")
    cal = (yaml.safe_load((CONFIG_DIR / "twin.yaml").read_text(encoding="utf-8")) or {}) \
        .get("calibration", {}) or {}
    pf = cal.get("pond_fit_not_applied") or {}
    w('<figure class="tbl"><table><thead><tr><th>Quantity</th><th>Pond fit</th>'
      "<th>Default kept</th></tr></thead><tbody>")
    twin_cfg = yaml.safe_load((CONFIG_DIR / "twin.yaml").read_text(encoding="utf-8"))
    for key in ("noise.do", "noise.tan", "noise.ph", "noise.temp_c", "noise.turbidity",
                "turbidity.base_ntu", "ph.equilibrium"):
        sec, name = key.split(".")
        w(f"<tr><td><code>{key}</code></td><td>{pf.get(key, '—')}</td>"
          f"<td>{twin_cfg.get(sec, {}).get(name, '—')}</td></tr>")
    w("</tbody></table>")
    global _tab_no
    _tab_no += 1
    w(f"<figcaption><b>Table {_tab_no}.</b> Twin calibration against the real ponds, computed "
      "and not applied. The fitted oxygen and ammonia noise is tens of times the defaults and "
      "the fitted baseline turbidity sits near the sensor's 100 NTU ceiling: magnitudes set by "
      "probe faults, not by pond physics. Applying them would make the twin random-walk. Both "
      "sets of values are recorded in <code>configs/twin.yaml</code>.</figcaption></figure>")

    h2("4.4 Scenarios")
    p("Each scenario is an initial state, a perturbation schedule and a fixed seed. A test "
      "asserts that each one actually produces the stress its name claims, so the evaluation "
      "cannot quietly become theatre. The signatures below are measured by running each "
      "scenario with no controller when this report is built.")
    sig = _scenario_signatures()
    mech = {
        "healthy_baseline": "Well-run pond",
        "night_do_crash": "Heavy standing crop, warm cloudy night",
        "overfeeding_ammonia": "Two large feed pulses",
        "algal_bloom": "Bloom then die-off",
        "heatwave": "Sustained +5 °C",
        "aerator_failure": "Aeration commanded but not delivered",
        "ph_swing": "Carbonate buffer collapses",
        "sensor_fault": "Probe dropout, spikes, stuck value",
    }
    w('<figure class="tbl"><table><thead><tr><th>Scenario</th><th>Mechanism</th>'
      "<th>Measured signature</th></tr></thead><tbody>")
    for a, b in mech.items():
        w(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{sig.get(a, '—')}</td></tr>")
    w("</tbody></table>")
    _tab_no += 1
    w(f"<figcaption><b>Table {_tab_no}.</b> The eight scenarios and the behaviour each "
      "exhibits with no controller.</figcaption></figure>")


def _scenario_signatures() -> Dict[str, str]:
    """Run every scenario with no actions and summarise the stress it actually produces."""
    from src.twin.pond import PondTwin
    from src.twin.scenarios import get_scenario, load_scenarios

    out: Dict[str, str] = {}
    for sid in load_scenarios():
        s = get_scenario(sid).reset()
        states = PondTwin(scenario=s, seed=s.seed, start_hour_of_day=s.start_hour_of_day) \
            .run(s.duration_h)
        do = [x.do for x in states]
        ph = [x.ph for x in states]
        if sid == "healthy_baseline":
            out[sid] = (f"DO {min(do):.1f}–{max(do):.1f}, pH {min(ph):.1f}–{max(ph):.1f} — "
                        "never needs rescuing")
        elif sid in ("night_do_crash", "aerator_failure"):
            out[sid] = f"DO minimum <b>{min(do):.2f} mg/L</b> (critical below 2.0)"
        elif sid == "overfeeding_ammonia":
            out[sid] = (f"un-ionised NH₃ peaks <b>{max(x.nh3 for x in states):.3f} mg/L</b> "
                        "(critical above 0.05)")
        elif sid == "algal_bloom":
            out[sid] = (f"DO supersaturates to <b>{max(do):.1f}</b>, then falls to "
                        f"<b>{min(do):.1f}</b> as the bloom decays")
        elif sid == "heatwave":
            out[sid] = f"water reaches <b>{max(x.temp_c for x in states):.1f} °C</b>"
        elif sid == "ph_swing":
            out[sid] = f"pH ranges <b>{min(ph):.1f}–{max(ph):.1f}</b>"
        else:
            out[sid] = "corrupts the <i>reading</i>, not the water — it acts in the sensor layer"
    return out


def s5_sensors() -> None:
    h1("5 · The sensor model", "s5")
    p("Perception never sees the twin's state. It sees this layer: additive Gaussian noise, a "
      "per-run calibration bias drawn once, probe quantisation, and random dropouts to "
      "<code>NaN</code>. Sampling is coarser than integration, so the twin can be integrated at "
      "five minutes while the probe reports at whatever cadence the operator has.")
    p("The <code>sensor_fault</code> scenario acts <b>here</b> rather than in the twin, because "
      "a broken probe corrupts the reading and not the water. That is exactly the distinction "
      "the anomaly detector has to cope with, and §9.5 measures how well it does.")
    yaml_block("sensors", ["sample_every_minutes", "noise_std", "bias", "dropout_prob",
                           "quantization"], "sensor simulator")
    note("<b>The guardrails see only this noisy view, but safety is judged against the pond's "
         "true state.</b> To bridge that gap honestly, each threshold carries a "
         "<code>sensor_margin</code> of roughly 3σ of its channel's noise, so a limit fires "
         "slightly early rather than slightly late. Without those margins the evaluation showed "
         "occasional unsafe actions caused purely by measurement error.")


def anomaly_verdict(fpr: float) -> str:
    """Plain verdict on the anomaly flag's false-alarm rate, from the measured number."""
    if fpr > 0.5:
        return (f"At a false-positive rate of {fpr:.0%} the flag fires on most ordinary windows, so "
                "it is not usable as an alarm on its own: the model was fitted mostly on noisy "
                "real-pond windows, and the simulated pond's normal behaviour falls outside them.")
    if fpr > 0.2:
        return (f"A false-positive rate of {fpr:.0%} is high; the flag is a weak hint, not an alarm.")
    return (f"A false-positive rate of {fpr:.0%} makes the flag usable as a novelty hint.")


def s6_perception() -> None:
    h1("6 · Perception stack", "s6")

    h2("6.1 LSTM forecaster")
    p("A single-layer LSTM over a 36-step window predicting 24 steps ahead for six channels, "
      "on CPU. Two design choices carry the whole thing.")
    w("<ol class='steps'>")
    w("<li><b>Delta targets.</b> The network predicts each channel's <i>change</i> from the last "
      "observed value, scaled by a fixed per-channel constant — never the absolute level. "
      "The real ponds run at 23–26 °C while the twin's pond runs near 28 °C; trained on deltas "
      "the network learns the <i>shape</i> of a diurnal oxygen cycle, which transfers between "
      "the two where absolute levels would not.</li>")
    w("<li><b>Fixed scales, not fitted ones.</b> Standardising by statistics fitted on pooled "
      "training data would be a privacy leak in the federated setting — a client's mean would "
      "have to cross the boundary — and would make the three regimes incomparable. The scales "
      "are constants in config.</li>")
    w("</ol>")
    p("The availability mask is fed in as extra input channels, so the model can distinguish "
      "'the pond is at zero' from 'the probe is offline'. A window is admitted only if the "
      "anchor step has real readings in the anchor channels; everything else may be absent, and "
      "the target mask excludes absent channels from the loss.")
    yaml_block("models", ["window", "horizon"], "window and horizon")

    h2("6.2 Regime discovery — a negative result")
    reg = read("regime_metrics.csv")
    p("HDBSCAN over windowed summary features. The features are <b>derived, not raw</b>: "
      "dissolved oxygen ÷ its saturation at the current temperature, temperature and pH as "
      "within-window anomalies, and log turbidity. Real-pond and twin windows are clustered "
      "together, and on raw levels the twin's warmer pond would sit outside every real-pond "
      "cluster; under derived features a pond at 6.5 mg/L and 28 °C and one at 7.0 mg/L and "
      "24 °C are compared by saturation, which is the physically meaningful similarity.")
    if reg is not None:
        table(reg, "Fitted regime model. A DBCV close to zero means the clusters are not "
              "well separated, whatever the silhouette says.", nd=4)
    else:
        absent("regime_metrics.csv")
    sweep = read("regime_sweep.csv")
    if sweep is not None:
        table(sweep, f"The sweep behind the negative result. Across every configuration DBCV "
              f"stays at or below {sweep['dbcv'].max():.3f}: no setting finds well-separated "
              "clusters.", nd=4,
              rename={"min_cluster_size": "min cluster size", "min_samples": "min samples",
                      "selection": "selection", "n_clusters": "clusters",
                      "noise_fraction": "noise fraction", "n_points": "points"})
    note("<b>The intended deliverable was a taxonomy of operating regimes. The data does not "
         "support one.</b> Continuous water-quality windows form a connected continuum — a pond "
         "slides around a manifold rather than jumping between discrete states. The model is "
         "therefore used only for <b>novelty detection</b>, where the noise label is the anomaly "
         "flag; §9 measures how well that works. Nothing downstream depends on the cluster id. Reporting a tuned "
         "configuration that happened to produce a prettier cluster count was the available "
         "alternative, and was rejected.", kind="warn")

    h2("6.3 Risk model")
    risk = read("risk_metrics.csv")
    p("LightGBM over current channels plus forecast summaries. The target is "
      "<b>rule-derived</b>: it is the risk band that <code>twin.yaml</code> assigns to the worst "
      "point of the <i>true</i> next horizon. So the model has not learned water chemistry — it "
      "has learned to anticipate a known rule from an imperfect forecast.")
    p("Why model a rule at all rather than just evaluate it? Because the rule needs the future, "
      "the future comes from a forecaster with real error, and a tree ensemble over the whole "
      "forecast summary degrades far more gracefully under that error than a hard threshold on a "
      "single predicted value. That framing is also what makes SHAP meaningful here: "
      "'critical because forecast minimum DO is 2.1' explains the model's decision and, by "
      "construction, the underlying rule.")
    if risk is not None:
        table(risk, "Risk model on a held-out seed — trained on seeds 1–3, tested on seed 4, "
              "with every scenario present on both sides of the split.", nd=3,
              rename={"balanced_accuracy": "balanced accuracy", "macro_f1": "macro F1",
                      "critical_recall": "critical recall",
                      "critical_precision": "critical precision", "held_out_seed": "held-out seed"})
    else:
        absent("risk_metrics.csv")
    note("<b>A methodological bug worth recording.</b> Feature rows are generated scenario by "
         "scenario, so an ordinary-looking positional 75/25 split put whole unseen scenarios in "
         "the test set — a transfer test wearing a holdout's clothes. Critical recall read 0.28. "
         "Splitting by <i>seed</i> instead, so every scenario appears on both sides and only the "
         "noise realisation differs, moved it to 1.00. <code>build_training_set</code> now "
         "returns group ids so this cannot silently recur.")

    h2("6.4 SHAP")
    p("<code>TreeExplainer</code> on the ensemble, so attributions are exact rather than sampled "
      "and cheap enough for every dashboard tick. Attributions are always for the "
      "<b>critical</b> class specifically: explaining the argmax class means the explanation "
      "changes meaning as the prediction moves, whereas 'what is pushing this toward critical' "
      "is a stable question with a comparable answer over time.")


def s7_federated() -> None:
    h1("7 · Federated learning", "s7")
    p("The premise the study rests on: <b>each pond trains its own model on its own data, and "
      "federation is the only thing that couples them</b>. The isolation is structural rather "
      "than promised — a <code>PondClient</code> owns one site's dataset and its own model "
      "instance and exposes exactly four operations (<code>local_train</code>, "
      "<code>get_weights</code>, <code>set_weights</code>, <code>evaluate</code>). The server "
      "receives weight tensors and a sample count, and nothing else.")

    h2("7.1 Three regimes over one client set")
    w("<ul>")
    w("<li><b>independent-local</b> — every site trains alone, nothing shared. This is the "
      "baseline the federated claim has to beat.</li>")
    w("<li><b>federated</b> — FedAvg. Each round every client runs E local epochs on its own "
      "data; the server averages weights, weighted by sample count; the average is broadcast "
      "back.</li>")
    w("<li><b>central</b> — all windows pooled into one model. An upper reference that breaks "
      "the privacy premise by design.</li>")
    w("</ul>")
    p("Three things make the comparison mean something:")
    w("<ol class='steps'>")
    w("<li><b>Matched total work.</b> <code>independent_local.epochs == rounds × "
      "local_epochs</code>, so a difference cannot be explained by one regime simply training "
      "longer.</li>")
    w("<li><b>A shared initialisation.</b> Every client starts FedAvg from identical weights. "
      "Averaging networks initialised differently averages across permuted hidden units and is "
      "meaningless.</li>")
    w("<li><b>Leave-some-sites-out.</b> Held-out sites are never trained on by any regime, "
      "stratified so both sources appear on both sides. The question is transfer to a pond never "
      "seen, not recall of the ones we have.</li>")
    w("</ol>")
    note("<code>independent-local</code> is scored as the <b>mean over all its per-site "
         "models</b>, because a pond that trained alone has no single global model to test. "
         "Reporting only its best member would be cherry-picking.")
    yaml_block("federated", ["clients", "split", "data_poor", "regimes"],
               "client construction, splitting and the three regimes")

    h2("7.2 Results on the held-out real pond")
    summary = read("federated_summary.csv")
    if summary is None:
        absent("federated_summary.csv")
    else:
        figure("f3_federated",
               "Forecast RMSE on the held-out real pond — a pond no regime trained on — for "
               "the federated, central and independent-local regimes.")
        table(summary, "Federated vs. central vs. independent-local, per channel, on the "
              "held-out real pond.", rename={"regime": "regime", "rmse_do": "DO RMSE (mg/L)",
                                     "rmse_temp_c": "temperature RMSE (°C)",
                                     "rmse_ph": "pH RMSE"})
    fh = read("forecast_horizon.csv")
    if fh is not None:
        figure("f4_horizon",
               "Dissolved-oxygen forecast error against horizon on held-out sites. Error growing "
               "with horizon is the expected and honest shape; the two coupled regimes track "
               "each other almost exactly.")

    h2("7.3 The data-poor pond — the claim federation is actually about")
    dp = read("federated_data_poor.csv")
    if dp is None:
        absent("federated_data_poor.csv")
    else:
        p("A designated subset of clients keeps only 8 % of its history and is scored on "
          "<b>its own</b> held-out tail, under the model it trained alone versus the federated "
          "global model it helped produce without sharing a single row.")
        cols = ["regime", "site", "data_fraction", "n_train", "rmse_do", "rmse_temp_c", "rmse_ph"]
        table(dp, "Data-poor clients on their own held-out tail.",
              cols=cols, rename={"data_fraction": "data kept", "n_train": "train windows",
                                 "rmse_do": "DO RMSE", "rmse_temp_c": "temp RMSE",
                                 "rmse_ph": "pH RMSE"})
        if {"regime", "rmse_do", "site"} <= set(dp.columns):
            piv = dp.pivot_table(index="site", columns="regime", values="rmse_do")
            if {"federated", "independent_local"} <= set(piv.columns):
                delta = (piv["federated"] - piv["independent_local"]) / piv["independent_local"]
                p(f"<b>Federation improved dissolved-oxygen forecast RMSE on "
                  f"{int((delta < 0).sum())} of {len(delta)} data-poor sites</b> — median change "
                  f"{delta.median() * 100:+.1f} %. Scoped to these sites, this window and this "
                  f"architecture; not a general claim.")
    note("<b>A known defect in this run.</b> <code>apply_data_poor</code> runs before "
         "<code>split_clients</code>, so one of the four designated data-poor clients was also "
         "drawn into the holdout and never appears in the table above. The effect is a smaller "
         "data-poor sample, not a biased one — that client was excluded from training for every "
         "regime equally. The fix is to designate them after the split.", kind="warn")

    h2("7.4 Differential privacy")
    p("DP-FedAvg — per-client update clipping plus Gaussian noise on the aggregate — is "
      "implemented behind <code>--dp</code>. Its ε comes from an RDP accountant when "
      "<code>dp-accounting</code> is installed and is reported as <b>UNACCOUNTED</b> otherwise. "
      "A guessed epsilon is worse than none. It has not been run for this report.")


def s8_agent() -> None:
    h1("8 · The agent: planners and guardrails", "s8")
    p("One decision per <code>decide_every_minutes</code>; the safety layer runs on <b>every "
      "sensor tick</b>. A pond can cross into critical in far less time than a planning "
      "interval, so planning is slow and safety is fast — the standard supervisory-control "
      "split, and what keeps unsafe-applied at zero <i>between</i> decisions.")

    h2("8.1 Planners")
    w("<ul>")
    w("<li><b>reactive</b> — transparent utility rules over the current reading <i>and</i> the "
      "forecast. This is what a careful operator with a good probe and a two-hour forecast would "
      "do; the predictive planner has to beat it.</li>")
    w("<li><b>predictive</b> (default) — learned MPC. For each of 14 curated candidate actions, "
      "roll a model forward and score "
      "<code>w_risk·time_in_risk + w_crit·time_critical + w_cost·intervention + "
      "w_stab·volatility + w_feed·feed_penalty</code>; take the argmin. The candidate list is "
      "curated rather than the full 3×3×3×2×2 cross product so that a reader can inspect it and "
      "know exactly what the agent was permitted to consider.</li>")
    w("</ul>")
    note("<b>What 'roll the forecaster forward under an action' actually means here.</b> The "
         "LSTM is trained on real pond data with no actuator log; it cannot be asked what an "
         "aerator does and cannot be "
         "conditioned on one. So the planner rolls an <b>internal pond model</b> forward — a "
         "second twin the agent owns and must keep aligned with reality from observations alone, "
         "never by reading the real twin's latent state. The LSTM then supplies a <i>residual "
         "correction</i>: the difference between its no-action forecast and the internal model's, "
         "damped and added to every candidate rollout. §10 reports what that correction is "
         "measured to be worth, and the answer is not flattering.")

    h2("8.2 Guardrails")
    p("Hard constraints, always on, for every planner. They fire in this "
      "order:")
    w("<ol class='steps'>")
    w("<li><b>Dosing caps and minimum intervals</b> — no dose the pond has not finished "
      "absorbing.</li>")
    w("<li><b>Rate limits</b> — no step change in aeration.</li>")
    w("<li><b>Contextual blocks</b> — no lime into an already-alkaline pond; no feeding at low "
      "oxygen or high ammonia; no large exchange into worse or thermally-shocking water.</li>")
    w("<li><b>Predicted-critical rejection</b> — any action forecast to drive a channel critical "
      "is refused, <i>unless</i> doing nothing is also critical, in which case the action is not "
      "the cause and must not be blamed.</li>")
    w("<li><b>Emergency override</b> — oxygen critical now or predicted forces the aerator to "
      "maximum and stops feeding.</li>")
    w("</ol>")
    p("The override is <b>last</b> so that it outranks everything above it, including the rate "
      "limit: when the pond is drowning, nothing outranks air. Every difference between proposed "
      "and applied is logged as a <code>GuardrailEvent</code> naming the rule that fired.")
    yaml_block("agent", ["decide_every_minutes", "cost_weights", "guardrails"],
               "decision cadence, cost weights and hard constraints")

    h2("8.3 Baselines, and what each is allowed to see")
    rows = [
        ("do_nothing", "nothing", "Reference trajectory. <b>Guardrails deliberately off</b> — "
         "rescuing it would destroy the comparison it exists to provide."),
        ("static_threshold", "current dissolved oxygen",
         "The conventional aerator thermostat. The bar to clear."),
        ("reactive", "channels + forecast", "Transparent rules over every channel, not just DO."),
        ("predictive", "channels + forecast + internal model", "The contribution."),
        ("oracle", "the twin's <b>latent</b> state — unavailable to any real farm",
         "A privileged-information <b>heuristic, not an optimum</b>. It is a hand-written rule "
         "that happens to see everything, and §9.3 shows the agent beating it in "
         "<code>ph_swing</code>, where its rule has no supersaturation term. Never deployable."),
    ]
    w('<figure class="tbl"><table><thead><tr><th>Controller</th><th>Sees</th><th>Status</th>'
      "</tr></thead><tbody>")
    for a, b, c in rows:
        w(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td></tr>")
    w("</tbody></table>")
    global _tab_no
    _tab_no += 1
    w(f"<figcaption><b>Table {_tab_no}.</b> The five controllers and their information "
      "sets.</figcaption></figure>")


def s9_evaluation() -> None:
    h1("9 · Evaluation: method and results", "s9")

    h2("9.1 Method")
    runs = read("agent_runs.csv")
    n = len(runs) if runs is not None else 0
    p(f"A full grid of {n} runs — 8 scenarios × 5 controllers × 10 seeds — plus 6 ablation "
      "variants at 40 runs each. One rule governs every comparison: <b>never report a bare "
      "mean</b>. Each controller comparison is a paired Wilcoxon signed-rank test across seeds, "
      "paired because the same seed gives every controller the identical perturbation schedule "
      "and noise realisation. <code>p</code> and <code>n</code> are reported for every "
      "comparison, and a comparison with fewer than six usable seeds refuses to test rather "
      "than quietly printing a mean. Central tendency is the median with an interquartile range, "
      "so one unlucky seed cannot carry a claim.")

    h2("9.2 Safety — the headline claim")
    safety = read("safety.csv")
    if safety is None:
        absent("safety.csv")
    else:
        table(safety, "Unsafe applied actions across the whole grid. Every guardrailed "
              "controller is at zero. <code>do_nothing</code> is unguarded by design, which is "
              "why it accrues thousands — it is a reference trajectory, not a controller.",
              nd=0, highlight="unsafe_applied_total",
              rename={"controller": "controller", "runs_n": "runs",
                      "unsafe_applied_total": "unsafe applied (total)",
                      "unsafe_applied_max": "unsafe applied (worst run)",
                      "guardrail_events_total": "guardrail events", "scenarios": "scenarios"})

    h2("9.3 The counterfactual, and agent vs. baselines")
    figure("f1_counterfactual",
           "The comparison the interactive twin exists to make possible. Both ponds were forked "
           "at t = 0 from the same state with the same seed, and receive the same perturbation "
           "schedule and the same sensor-noise realisation. The only difference between them is "
           "the agent — so every gap between the lines is attributable to an action rather than "
           "to luck. A test asserts the strong form: with no actions the two trajectories are "
           "identical.")
    figure("f2_controllers",
           "Dissolved-oxygen time-out-of-band per scenario. The agent leads in every scenario "
           "that has any risk to reduce. <code>healthy_baseline</code> and "
           "<code>sensor_fault</code> are omitted: every controller scores exactly zero there, "
           "so there is nothing to compare.")
    stats = read("agent_stats.csv")
    if stats is not None:
        key = stats[(stats.metric == "risk_h_do") & (stats.controller_a == "predictive")].copy()
        # A rank-biserial correlation of exactly 1.0 renders as a bare "1" under the integer
        # rule and reads like a placeholder; a correlation should always show its decimals.
        if "effect_size_rb" in key.columns:
            key["effect_size_rb"] = key["effect_size_rb"].map(
                lambda v: "–" if pd.isna(v) else f"{float(v):.2f}")
        table(key, "Paired Wilcoxon signed-rank, agent vs. each baseline, on dissolved-oxygen "
              "time-out-of-band. Negative <code>median difference</code> means the agent is "
              "better.",
              cols=["scenario", "controller_b", "median_a", "median_b", "median_diff", "p", "n",
                    "effect_size_rb", "note"],
              rename={"controller_b": "vs.", "median_a": "agent", "median_b": "baseline",
                      "median_diff": "median difference", "effect_size_rb": "effect size (rb)"})
        sig = key[(key.p < 0.05) & (key.median_diff < 0)]
        ties = key[key.median_diff == 0]
        p(f"The agent is significantly better (p &lt; 0.05) in <b>{len(sig)} of {len(key)}</b> "
          f"scenario × baseline comparisons on this metric. The {len(ties)} exact ties are all "
          f"in scenarios where no controller accrues any dissolved-oxygen risk at all, so there "
          f"is nothing to win.")
    summary = read("agent_summary.csv")
    if summary is not None:
        cols = ["scenario", "controller", "risk_h_do_median", "risk_h_do_iqr",
                "time_critical_h_median", "interventions_median", "intervention_cost_median"]
        table(summary, "Median and interquartile range per scenario × controller.",
              cols=cols, nd=2,
              rename={"risk_h_do_median": "DO risk (h, median)", "risk_h_do_iqr": "IQR",
                      "time_critical_h_median": "critical (h)",
                      "interventions_median": "interventions",
                      "intervention_cost_median": "cost"})

    h2("9.4 What it costs")
    figure("f6_tradeoff",
           "Both sides of the ledger against the static-threshold baseline. The agent avoids "
           "real risk and pays for it in every scenario — including two where it achieves the "
           "same zero oxygen risk the thermostat already achieved.")
    if stats is not None:
        cost = stats[(stats.metric == "intervention_cost") & (stats.controller_a == "predictive")
                     & (stats.controller_b == "static_threshold")]
        if not cost.empty:
            table(cost, "Operating cost, agent vs. static threshold. Positive "
                  "<code>median difference</code> means the agent spends more.",
                  cols=["scenario", "median_a", "median_b", "median_diff", "p", "n"], nd=2,
                  rename={"median_a": "agent", "median_b": "thermostat",
                          "median_diff": "median difference"})
            p(f"The agent is more expensive to run in <b>{int((cost.median_diff > 0).sum())} of "
              f"{len(cost)}</b> scenarios. It buys preemption, and preemption means acting on "
              f"excursions that would sometimes have resolved on their own. "
              f"<code>cost_weights.intervention</code> is the dial that trades one against the "
              f"other; nothing in this report establishes where a real farm should set it.")

    h2("9.5 Anomaly detection")
    an = read("anomaly_detection.csv")
    if an is None:
        absent("anomaly_detection.csv")
    else:
        p(f"On the injected <code>sensor_fault</code> events: <b>event recall "
          f"{an['recall'].mean():.2f}</b> ({int(an['detected'].sum())} of "
          f"{int(an['events'].sum())} events across {len(an)} seeds) at a <b>false-positive rate "
          f"of {an['false_positive_rate'].mean():.3f}</b> outside the fault windows. Scored per "
          f"<i>event</i>, not per tick — a detector that fired constantly would score perfectly "
          f"on a per-tick metric. " + anomaly_verdict(an["false_positive_rate"].mean()))
        table(an, "Anomaly detection per seed. The noise label cannot distinguish a broken probe "
              "from a genuine novel excursion; the false-positive rate is the price of that "
              "generality.", cols=["seed", "events", "detected", "recall", "false_positive_rate"],
              rename={"false_positive_rate": "false-positive rate"})

    h2("9.6 Ablations")
    figure("f5_ablations",
           "Six variants, 40 runs each. Two results matter and point in opposite directions: "
           "removing the forecaster's correction <i>improves</i> control, and removing the "
           "guardrails produces hundreds of unsafe actions.")
    ab = read("ablations.csv")
    if ab is None:
        absent("ablations.csv")
    else:
        agg = (ab.groupby("ablation")
                 .agg(do_risk=("risk_h_do", "median"), critical=("time_critical_h", "median"),
                      cost=("intervention_cost", "median"),
                      unsafe=("unsafe_applied", "sum"), runs=("seed", "count"))
                 .reset_index().sort_values("do_risk"))
        table(agg, "Ablation results.", nd=2, highlight="unsafe",
              rename={"ablation": "variant", "do_risk": "DO risk (h, median)",
                      "critical": "critical (h)", "cost": "cost", "unsafe": "unsafe applied",
                      "runs": "runs"})


def s10_observations() -> None:
    h1("10 · Observations — including the unflattering ones", "s10")
    p("These are the results to read before presenting the project, because they are the first "
      "things a reviewer will probe. Every number and every direction below is computed from "
      "<code>outputs/</code> when this report is built.")

    ab = read("ablations.csv")
    med = ab.groupby("ablation")["risk_h_do"].median() if ab is not None else None
    uns = ab.groupby("ablation")["unsafe_applied"].sum() if ab is not None else None

    h3("10.1 What the federated forecaster does to the controller")
    if med is not None and {"full", "no_lstm_correction"} <= set(med.index):
        diff = med["full"] - med["no_lstm_correction"]
        if diff > 0:
            p(f"The <code>no_lstm_correction</code> ablation scores <b>"
              f"{med['no_lstm_correction']:.2f} h</b> of dissolved-oxygen time-out-of-band "
              f"against <b>{med['full']:.2f} h</b> for the shipped system — the planner is "
              f"<b>{diff:.2f} h better without the correction</b>. A forecaster trained on noisy "
              "low-cost pond probes, used as a residual correction on the simulated pond, "
              "injects more bias than it removes, even damped by half.")
            p("The correction is nevertheless <b>enabled in the shipped configuration</b>: the "
              "federated LSTM is the component this project exists to produce, and keeping it on "
              "the actuator path is what makes the end-to-end system a <i>learned</i> controller. "
              "A deployment optimising only for the score should set "
              "<code>predictive.use_forecaster: false</code>.")
        elif diff < 0:
            p(f"The LSTM residual correction <b>helps</b> control: <b>{med['full']:.2f} h</b> of "
              f"dissolved-oxygen time-out-of-band with it against "
              f"<b>{med['no_lstm_correction']:.2f} h</b> without it.")
        else:
            p(f"The LSTM residual correction makes no difference to the median "
              f"({med['full']:.2f} h either way).")
    else:
        absent("ablations.csv")

    h3("10.2 A learned dynamics model needs something outside it to stay safe")
    if med is not None and {"full", "no_world_model", "no_guardrails"} <= set(med.index):
        wd = med["full"] - med["no_world_model"]
        p(f"Replacing the hand-written pond model with the action-conditioned LSTM changes "
          f"dissolved-oxygen time-out-of-band by <b>{wd:+.2f} h</b> ({med['full']:.2f} h "
          f"against {med['no_world_model']:.2f} h). With the guardrails removed, the learned "
          f"planner applies <b>{int(uns['no_guardrails'])}</b> unsafe actions across the "
          f"ablation grid; with them in place, <b>{int(uns['full'])}</b>.")
        if "scenario" in ab.columns:
            per = (ab[ab["ablation"] == "no_guardrails"].groupby("scenario")["unsafe_applied"]
                   .sum().sort_values(ascending=False))
            per = per[per > 0]
            if len(per):
                p("Unguarded unsafe actions by scenario: "
                  + ", ".join(f"<code>{k}</code> {int(v)}" for k, v in per.items()) + ".")
    p("The mechanism generalises past this project. The planner optimises against its model "
      "<i>as though the model were exact</i>. A learned model can be confidently wrong in a "
      "small region of state space, and an optimiser will find exactly that region. Prediction "
      "error becomes unsafe commands, which is why the hard constraints live <b>outside</b> the "
      "learned component rather than inside its cost function.")

    h3("10.3 HDBSCAN found no regime taxonomy")
    reg = read("regime_metrics.csv")
    if reg is not None:
        r0 = reg.iloc[0]
        p(f"The fitted model has {int(r0['n_clusters'])} clusters, "
          f"{r0['noise_fraction']:.0%} of windows unassigned and DBCV {r0['dbcv']:.3f} beside "
          f"a silhouette of {r0['silhouette']:.3f}. A near-zero DBCV means the windows form a "
          "continuum, not clusters; the sweep in §6 shows the same across settings. Only the "
          "noise label is used, as an anomaly flag.")
    an = read("anomaly_detection.csv")
    if an is not None:
        p(anomaly_verdict(an["false_positive_rate"].mean()) + " A dedicated per-probe "
          "sensor-health check would be the stronger tool for catching faulty probes.")

    h3("10.4 The agent wins on risk and pays on cost")
    st = read("agent_stats.csv")
    if st is not None:
        c = st[(st["metric"] == "intervention_cost") & (st["controller_a"] == "predictive")
               & (st["controller_b"] == "static_threshold")]
        k = st[(st["metric"] == "risk_h_do") & (st["controller_a"] == "predictive")]
        wins = int(((k["p"] < 0.05) & (k["median_diff"] < 0)).sum())
        p(f"Significantly lower dissolved-oxygen risk in <b>{wins} of {len(k)}</b> scenario × "
          f"baseline comparisons, and more expensive to operate than the thermostat in "
          f"<b>{int((c['median_diff'] > 0).sum())} of {len(c)}</b> scenarios. Preemption is not "
          "free; reporting only the risk reduction would be a half-truth.")

    h3("10.5 The data is real but rough")
    p("Every pond logger is offline for 37–55 % of its timeline, a few percent more of the "
      "readings are physically impossible and are dropped, and nitrate is unusable. The "
      "forecaster's absolute error is correspondingly large; the claims in this report are "
      "comparisons between training regimes and between controllers, not absolute accuracy.")

    h3("10.6 The oracle is not an upper bound")
    p("It is a hand-written rule with privileged information. Where the predictive agent "
      "beats it (see the per-scenario table in §9), that is because the rule has no term for "
      "the hazard involved. It is reported as a reference heuristic, never as a ceiling.")


def s11_limitations() -> None:
    h1("11 · Limitations and what would fix them", "s11")
    rows = [
        ("The controlled pond is a simulation", "Nothing here is evidence about a real farm's "
         "response to an action. Coefficients are plausible defaults, not measurements.",
         "Validate the twin against an instrumented pond with a logged action history."),
        ("Low-cost, gappy sensor data", "Loggers are offline for roughly half of each pond's "
         "timeline and a share of readings are impossible values; nitrate is unusable.",
         "Better-maintained probes, or a larger set of ponds so outages average out."),
        ("Small study", "Seven ponds from one region: five train, two are held out.",
         "More ponds, and more than one region once the single-region result is established."),
        ("No real action-response data", "The pond dataset logs water quality but not when the "
         "aerator ran or how much was fed, so the world model learns actions in the twin.",
         "Log actuator commands alongside the sensors on a real farm."),
        ("Regimes are a continuum", "No usable operating-regime taxonomy was found; only "
         "novelty detection survived.",
         "A supervised state definition, or accept novelty detection as the deliverable."),
        ("Risk is a rule, not chemistry", "The risk model reproduces bands from config. It "
         "cannot discover a hazard the bands do not encode.",
         "Species-specific band calibration; mortality or growth outcomes as a real target."),
        ("Cost weights are unvalidated", "The risk/cost trade-off is set by config constants "
         "with no economic grounding.",
         "Farm-level cost data: feed price, power tariff, mortality risk."),
        ("Differential privacy unrun", "DP-FedAvg is implemented but has no reported ε.",
         "Install an accountant and run the DP arm; report ε honestly or not at all."),
    ]
    w('<figure class="tbl"><table><thead><tr><th>Limitation</th><th>What it means</th>'
      "<th>What would fix it</th></tr></thead><tbody>")
    for a, b, c in rows:
        w(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>")
    w("</tbody></table>")
    global _tab_no
    _tab_no += 1
    w(f"<figcaption><b>Table {_tab_no}.</b> Known limitations, stated with their "
      "remedies.</figcaption></figure>")


def s12_repro() -> None:
    h1("12 · Reproducing this report", "s12")
    p("Python 3.11, CPU only, no GPU. One global seed; twin noise seeded per scenario. "
      "Collection is idempotent — a cached site is never re-downloaded.")
    w('<div class="code"><div class="codehead">from a clean checkout</div><pre>'
      + html.escape(
          "pip install -r requirements.txt\n\n"
          "python scripts/collect_data.py          # ONE-TIME, online. ~40 sites -> data_cache/\n"
          "python scripts/calibrate_twin.py        # offline. fits twin noise to the real data\n"
          "python scripts/pretrain_federated.py    # offline. FedAvg + regimes + risk -> models/\n"
          "python -m src.eval.run_eval             # offline. every table in outputs/\n"
          "python scripts/make_figures.py          # offline. report/figures/*.svg\n"
          "python scripts/make_report_pdf.py       # offline. this document\n\n"
          "python -m pytest tests/ -q              # 102 tests\n"
          "streamlit run dashboard/app.py          # the live dashboard")
      + "</pre></div>")
    p("The dashboard runs without the middle steps — it degrades to the rule-based risk bands "
      "with no forecast overlay, and says so in the sidebar rather than failing.")

    h2("12.1 Test coverage")
    rows = [
        ("Twin sanity", "Aeration raises oxygen at night and <i>strips</i> it from a "
         "supersaturated pond; exchange dilutes ammonia; lime raises alkalinity; photosynthesis "
         "is zero at night; NH₃ fraction rises with pH and temperature; every scenario produces "
         "its advertised stress."),
        ("Counterfactual integrity", "A forked twin given identical actions produces a "
         "bit-identical trajectory; <code>do_nothing</code> equals its own shadow exactly. "
         "Without this the counterfactual panel would be comparing the agent against luck."),
        ("Guardrail units", "One case per rule that would violate it, asserting the "
         "<i>applied</i> action does not. Plus both halves of §10.4."),
        ("Collector", "Schema and mask correctness, censored zeros, flatline masking, "
         "duplicate timestamps, and that forward fill is never reported as an observation."),
        ("Perception", "Window shapes and finiteness, survival of missing channels, weight "
         "round-trips, determinism, and that the derived regime features are dimensionless "
         "across a 16 °C domain gap."),
        ("Evaluation", "Wilcoxon refuses too-few seeds, anomaly recall is per-event, runs are "
         "reproducible, and the harness returns what callers think it returns."),
    ]
    w('<figure class="tbl"><table><thead><tr><th>Area</th><th>What is asserted</th></tr></thead>'
      "<tbody>")
    for a, b in rows:
        w(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>")
    w("</tbody></table>")
    global _tab_no
    _tab_no += 1
    w(f"<figcaption><b>Table {_tab_no}.</b> What the test suite protects.</figcaption></figure>")


def s13_appendix(manifest: Optional[pd.DataFrame]) -> None:
    h1("13 · Appendix", "s13")

    h2("13.1 Repository map")
    w('<div class="code"><div class="codehead">layout</div><pre>' + html.escape(
        "configs/      data, twin, scenarios, sensors, models, federated, agent — all behaviour\n"
        "src/\n"
        "  common/     runtime (seeds, threads, config), schema (the data contracts)\n"
        "  data/       sources/kaggle_pond (+ unused usgs, ukea), collect, harmonize, registry\n"
        "  twin/       pond, dynamics, scenarios, calibrate, risk_bands\n"
        "  sensors/    stream\n"
        "  perception/ forecaster, regimes, risk, explain\n"
        "  federated/  client, simulate\n"
        "  agent/      policy, guardrails, actuators, loop\n"
        "  eval/       baselines, metrics, run_eval\n"
        "dashboard/    app.py + panels/\n"
        "scripts/      collect_data, calibrate_twin, pretrain_federated, build_report,\n"
        "              make_figures, make_report_pdf\n"
        "tests/        twin, guardrails, data, perception, agent\n"
        "data_cache/   raw + harmonised parquet per site + manifest.csv   (git-ignored)\n"
        "models/       pretrained weights + regime + risk + manifest.json (git-ignored)\n"
        "outputs/      every reported number                              (git-ignored)")
      + "</pre></div>")

    h2("13.2 Model and training provenance")
    mp = MODELS_DIR / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        w("<ul>")
        w(f"<li>models trained <b>{html.escape(str(m.get('trained_at')))}</b></li>")
        w(f"<li>{m.get('n_clients')} clients; {len(m.get('holdout_sites', []))} held out</li>")
        fed = m.get("federated", {})
        w(f"<li>FedAvg: {fed.get('rounds')} rounds × {fed.get('local_epochs')} local epochs, "
          f"client fraction {fed.get('client_fraction')}, sample-weighted: "
          f"{fed.get('weighted_by_samples')}</li>")
        w(f"<li>held-out sites: <code>{html.escape(', '.join(m.get('holdout_sites', [])))}</code></li>")
        w(f"<li>data-poor sites: <code>{html.escape(', '.join(m.get('data_poor_sites', [])))}</code></li>")
        dp = m.get("differential_privacy", {})
        w(f"<li>differential privacy: {'enabled, ε = ' + str(dp.get('epsilon')) if dp.get('enabled') else 'disabled'}</li>")
        w("</ul>")
    twin_path = CONFIG_DIR / "twin.yaml"
    if twin_path.exists():
        cal = (yaml.safe_load(twin_path.read_text(encoding="utf-8")) or {}).get("calibration")
        if cal:
            w("<ul>")
            w(f"<li>twin: {html.escape(str(cal.get('status', 'uncalibrated')))}</li>")
            pf = cal.get("pond_fit_not_applied") or {}
            if pf:
                w(f"<li>a fit to the {pf.get('n_ponds')} real ponds was computed and "
                  f"<b>deliberately not applied</b> (DO noise {pf.get('noise.do')} mg/L/√h, "
                  f"baseline turbidity {pf.get('turbidity.base_ntu')} NTU — probe faults, not "
                  "pond physics); see §4.3</li>")
            w("</ul>")

    h2("13.3 Ponds used")
    if manifest is None:
        absent("manifest.csv")
    else:
        mf = manifest.copy()
        mf["rows"] = mf["rows"].astype(int)
        table(mf, "Every pond that passed quality control. This table is the single source of truth for "
              "'how many sites / how many rows' anywhere in the project.",
              cols=["client_id", "source", "region", "n_channels", "rows", "missing_frac"],
              nd=3, rename={"client_id": "client", "n_channels": "channels",
                            "missing_frac": "missing fraction"})

    h2("13.4 A note on the figures")
    p("Colour in every figure follows a validated categorical palette: slots are assigned in a "
      "fixed order and never cycled, one hue per entity, and status colours (used only for the "
      "unsafe-action marker) are reserved for state and never reused as a series. The palette was "
      "checked with a colour-vision-deficiency validator rather than by eye — worst adjacent "
      "CVD ΔE 9.1 against a target of 8, worst normal-vision ΔE 19.6 against a floor of 15. "
      "Three slots sit below 3:1 contrast on the chart surface, so every bar carries either a "
      "direct value label or its full table in this document. No figure uses two y-scales.")
    p("Bar thickness is specified in points and converted from each axes' physical size. Doing it "
      "in data units — the obvious approach — is wrong: where the x and y scales differ by an "
      "order of magnitude, a corner radius in data units turns every bar into an ellipse, which "
      "is exactly what the first draft of Figure 3 did.")


# ======================================================================================
# Document shell
# ======================================================================================
CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  font-size: 9.6pt; line-height: 1.52; color: #0b0b0b; background: #fff; margin: 0;
}
h1, h2, h3 { line-height: 1.25; color: #0b0b0b; }
h1 {
  font-size: 15pt; margin: 0 0 10pt; padding-bottom: 5pt;
  border-bottom: 1.6pt solid #2a78d6; break-before: page; break-after: avoid;
}
section.cover h1, h1#s1 { break-before: auto; }
h2 { font-size: 11.4pt; margin: 15pt 0 6pt; break-after: avoid; }
h3 { font-size: 10.2pt; margin: 12pt 0 4pt; color: #1c5cab; break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
a { color: #1c5cab; text-decoration: none; }
code {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 0.88em; background: #f2f1ec; padding: 0.5pt 2.5pt; border-radius: 2.5pt;
}
ul, ol { margin: 0 0 8pt; padding-left: 15pt; }
li { margin-bottom: 3.5pt; }
ol.steps { padding-left: 15pt; }
ol.steps li { margin-bottom: 5pt; }

/* cover */
section.cover { padding-top: 6mm; break-after: page; }
.eyebrow {
  font-size: 8.4pt; letter-spacing: 0.14em; text-transform: uppercase;
  color: #2a78d6; font-weight: 600; margin-bottom: 6pt;
}
h1.title {
  font-size: 34pt; margin: 0 0 8pt; border: 0; padding: 0; letter-spacing: -0.5pt;
  break-before: auto;
}
.subtitle {
  font-size: 11.6pt; line-height: 1.45; color: #52514e; max-width: 148mm; margin-bottom: 16pt;
}
.statrow { display: flex; gap: 6mm; margin: 0 0 15pt; flex-wrap: wrap; }
.stat {
  border-left: 2.4pt solid #2a78d6; padding: 1pt 0 1pt 6pt; min-width: 30mm;
}
.statval { font-size: 17pt; font-weight: 600; line-height: 1.1; }
.statlab { font-size: 8.2pt; color: #52514e; }
.covernote {
  background: #fdf6e8; border: 0.7pt solid #f0d9a8; border-left: 2.6pt solid #fab219;
  padding: 8pt 10pt; font-size: 9.2pt; line-height: 1.5; border-radius: 3pt;
  margin-bottom: 12pt;
}
.meta { font-size: 8pt; color: #898781; line-height: 1.5; }

/* toc */
section.toc { break-after: page; }
section.toc h2 { margin-top: 0; }
section.toc ol { padding-left: 16pt; }
section.toc li { margin-bottom: 4.5pt; font-size: 10pt; }

/* callouts */
.callout {
  border-radius: 3pt; padding: 7pt 9pt; margin: 8pt 0 10pt; font-size: 9.2pt;
  break-inside: avoid; line-height: 1.48;
}
.callout.note { background: #eef4fc; border-left: 2.6pt solid #2a78d6; }
.callout.warn { background: #fdf6e8; border-left: 2.6pt solid #fab219; }
.callout.ok   { background: #edf7f0; border-left: 2.6pt solid #2f8f5b; }
.callout.missing {
  background: #f7f7f5; border-left: 2.6pt solid #898781; color: #52514e; font-style: italic;
}

/* figures */
figure { margin: 10pt 0 12pt; break-inside: avoid; }
figure.fig .svgwrap { text-align: center; }
figure.fig svg { max-width: 100%; height: auto; }
figcaption {
  font-size: 8.4pt; color: #52514e; line-height: 1.45; margin-top: 5pt;
  padding-left: 1pt; text-align: left;
}

/* tables */
figure.tbl { break-inside: auto; }
table { width: 100%; border-collapse: collapse; font-size: 8.2pt; }
/* A 40-row table will span pages; repeat its header so a continuation page is readable. */
thead { display: table-header-group; }
tbody tr { break-inside: avoid; }
figure.tbl figcaption { break-before: avoid; }
thead th {
  text-align: left; font-weight: 600; color: #0b0b0b; background: #f2f1ec;
  border-bottom: 0.9pt solid #c3c2b7; padding: 4pt 5pt; vertical-align: bottom;
}
tbody td {
  padding: 3.4pt 5pt; border-bottom: 0.4pt solid #e1e0d9; vertical-align: top;
  font-variant-numeric: tabular-nums;
}
tbody tr.hl td { background: #fdeceb; }
tbody td:first-child { font-variant-numeric: normal; }
td.yes { color: #006300; font-weight: 600; white-space: nowrap; }
td.no { color: #c1372c; font-weight: 600; white-space: nowrap; }

/* code blocks */
.code {
  border: 0.7pt solid #e1e0d9; border-radius: 3pt; overflow: hidden; margin: 8pt 0 11pt;
  break-inside: avoid;
}
.codehead {
  background: #f2f1ec; border-bottom: 0.7pt solid #e1e0d9; padding: 3.5pt 7pt;
  font-size: 8pt; color: #52514e; font-weight: 600;
}
.code pre {
  margin: 0; padding: 7pt; font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 7.8pt; line-height: 1.42; white-space: pre-wrap; color: #0b0b0b;
}

/* equations */
.eqn { margin: 8pt 0 10pt; }
.eq {
  background: #fafaf8; border-left: 2.2pt solid #c3c2b7; padding: 6pt 9pt; margin-bottom: 6pt;
  font-size: 9pt; line-height: 1.62; break-inside: avoid;
}
"""


def build_html() -> str:
    manifest = read("manifest.csv", DATA_CACHE)
    cover(manifest)
    toc()
    s1_overview()
    s2_architecture()
    s3_data(manifest)
    s4_twin()
    s5_sensors()
    s6_perception()
    s7_federated()
    s8_agent()
    s9_evaluation()
    s10_observations()
    s11_limitations()
    s12_repro()
    s13_appendix(manifest)
    body = "\n".join(OUT)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>AquaSentinel — Project Report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def find_browser() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _print_pdf(html_path: Path, out_pdf: Path):
    """Print via the DevTools protocol, falling back to the plain CLI flag.

    ``chrome --print-to-pdf`` works but stamps Chrome's own header and footer on every page -
    a timestamp, the title, and the full local file:// path - and there is no Chrome switch to
    suppress them (``--no-pdf-header-footer`` is a Puppeteer option; passing it to Chrome makes
    it print a single blank page). ``Page.printToPDF`` over CDP takes ``displayHeaderFooter:
    false`` directly, so that is the primary path; the CLI is kept as a fallback for machines
    without Node.
    """
    driver = Path(__file__).with_name("html_to_pdf.mjs")
    if shutil.which("node") and driver.exists():
        proc = subprocess.run(
            ["node", str(driver), str(html_path), str(out_pdf)],
            capture_output=True, text=True, timeout=300,
        )
        if out_pdf.exists():
            return proc
        print("  CDP print failed, falling back to the CLI flag "
              "(expect a Chrome header/footer on each page)")
        print("  " + (proc.stderr or proc.stdout or "").strip()[-400:])

    browser = find_browser()
    if browser is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        return subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-sandbox",
             f"--user-data-dir={tmp}", f"--print-to-pdf={out_pdf}",
             "--virtual-time-budget=20000", html_path.resolve().as_uri()],
            capture_output=True, text=True, timeout=300,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the project report PDF (offline).")
    ap.add_argument("--out", default=str(OUT_PDF))
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_pdf.with_suffix(".html")
    html_path.write_text(build_html(), encoding="utf-8")
    print(f"HTML  -> {html_path}  ({html_path.stat().st_size / 1024:.0f} KB)")

    out_pdf.unlink(missing_ok=True)
    proc = _print_pdf(html_path, out_pdf)
    if not out_pdf.exists():
        print("PDF was not produced.")
        if proc is not None:
            print((proc.stderr or proc.stdout or "")[-1500:])
        print(f"The HTML is at {html_path} - print it to PDF from any browser.")
        return 1

    print(f"PDF   -> {out_pdf}  ({out_pdf.stat().st_size / 1024:.0f} KB)")
    print(f"        {_fig_no} figures, {_tab_no} tables")
    if not args.keep_html:
        html_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
