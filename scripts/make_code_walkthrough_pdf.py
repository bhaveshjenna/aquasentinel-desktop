"""Build report/AquaSentinel-Code-Walkthrough.pdf - where every key step lives, and how it works.

    python scripts/make_code_walkthrough_pdf.py [--keep-html]

For each step of the pipeline the document shows the file, the function and its line range,
the code itself, numbered markers on the lines that matter, and an explanation of each marker.

Nothing is pasted in. Every snippet is cut out of the source by parsing it with ``ast`` at build
time, the line numbers in the gutter are the real ones, and every marker is located by searching
for a fragment of the line it annotates. If a function is renamed or a marked line changes, the
build fails loudly instead of printing a stale explanation. Long docstrings are folded to their
first line (the gutter shows the gap) so the page is spent on code, not prose already explained.
"""

from __future__ import annotations

import argparse
import ast
import html
import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "report" / "AquaSentinel-Code-Walkthrough.pdf"


# ======================================================================================
# Extraction
# ======================================================================================
@dataclass
class Snippet:
    path: str                      # repo-relative
    qualname: str                  # e.g. "PondTwin.step"
    start: int = 0
    end: int = 0
    lines: List[Tuple[Optional[int], str]] = field(default_factory=list)  # (lineno, text)


def _find(tree: ast.AST, qualname: str) -> ast.AST:
    node: ast.AST = tree
    for part in qualname.split("."):
        found = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and child.name == part:
                found = child
                break
        if found is None:
            raise KeyError(f"{qualname}: '{part}' not found")
        node = found
    return node


def extract(path: str, qualname: str, fold_docstrings: bool = True,
            only_head: Optional[int] = None) -> Snippet:
    """Cut one function or class out of a file, with real line numbers.

    ``only_head`` keeps just the first N lines of the body (used for long classes, where the
    constructor is the point), and marks the cut.
    """
    src = (ROOT / path).read_text(encoding="utf-8")
    all_lines = src.splitlines()
    node = _find(ast.parse(src), qualname)
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    end = node.end_lineno

    folded: Dict[int, int] = {}      # first line of a folded docstring -> its last line
    if fold_docstrings:
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
                first = n.body[0]
                if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None),
                                                               ast.Constant) \
                        and isinstance(first.value.value, str) \
                        and first.end_lineno > first.lineno:
                    folded[first.lineno] = first.end_lineno

    out: List[Tuple[Optional[int], str]] = []
    i = start
    while i <= end:
        if i in folded:
            text = all_lines[i - 1]
            indent = text[: len(text) - len(text.lstrip())]
            head = text.strip().strip('"').strip("'").strip()
            out.append((i, f'{indent}"""{head} …"""'))
            out.append((None, f"{indent}# … docstring lines {i + 1}-{folded[i]} folded"))
            i = folded[i] + 1
            continue
        out.append((i, all_lines[i - 1]))
        i += 1
    if only_head is not None and len(out) > only_head:
        cut_at = out[only_head - 1][0] or start
        out = out[:only_head] + [(None, f"    # … lines {cut_at + 1}-{end} not shown")]
    return Snippet(path, qualname, start, end, out)


# ======================================================================================
# Content model
# ======================================================================================
@dataclass
class Item:
    path: str
    qualname: str
    what: str                                  # one paragraph: what this code is for
    marks: List[Tuple[str, str]]               # (fragment of the line, explanation)
    called_from: str = ""
    only_head: Optional[int] = None


@dataclass
class Step:
    num: str
    title: str
    intro: str
    items: List[Item]


# ======================================================================================
# The walkthrough itself
# ======================================================================================
def steps() -> List[Step]:
    return [
        # ------------------------------------------------------------------ 1
        Step("1", "Configuration and reproducibility",
             "Every run starts here. All tunable numbers live in <code>configs/*.yaml</code>, "
             "and one seed controls every random number generator, so any run can be repeated "
             "exactly.",
             [
                 Item("src/common/runtime.py", "set_seed",
                      "Seeds Python, NumPy and PyTorch together and pins PyTorch to one thread, so "
                      "results do not change with the machine's load.",
                      [("random.seed(seed)", "Python's own generator."),
                       ("np.random.seed(seed)", "NumPy, used by the twin's noise and the splits."),
                       ("torch.manual_seed(seed)", "PyTorch: identical initial network weights."),
                       ("torch.set_num_threads(1)", "One thread: reproducible and kind to a laptop.")],
                      "every script and every AgentRun"),
                 Item("src/common/runtime.py", "load_config",
                      "Reads one YAML file from <code>configs/</code>. Cached, so a config is parsed "
                      "once per process.",
                      [("@lru_cache", "Cache: repeated calls return the same parsed dict."),
                       ("path = CONFIG_DIR", "Configs are always found relative to the project root.")],
                      "almost every module"),
             ]),
        # ------------------------------------------------------------------ 2
        Step("2", "Loading the real pond data",
             "The only data source is the Kaggle aquaponics dataset from Nsukka, Nigeria: one CSV per "
             "pond. Each file is turned into a long table of (timestamp, channel, value) rows.",
             [
                 Item("src/data/sources/kaggle_pond.py", "_map_columns",
                      "Every pond file spells its column names differently (\"Temperature (C)\", "
                      "\"temperature(C)\" …). This maps them onto the project's channel names by "
                      "comparing normalised names.",
                      [("exact: Dict[str, str]", "Exact names from config are tried first."),
                       ("n = _norm(col)", "Lower-case, letters and digits only: spelling differences vanish."),
                       ("(n == k) if len(k) <= 3 else (k in n)", "Short keys like \"ph\" must match the whole "
                        "name, so \"phosphate\" is never read as pH."),
                       ("if ch not in seen:", "One column per channel; a duplicate is dropped, not averaged.")],
                      "fetch_site"),
                 Item("src/data/sources/kaggle_pond.py", "_parse_time",
                      "The files mix \"2021-06-18 13:02:10 CET\" with ISO \"+01:00\" timestamps. Both are "
                      "UTC+1 (West Africa Time).",
                      [("str.replace(r\"\\s+[A-Z]{2,4}$\"", "A trailing zone abbreviation becomes the configured offset."),
                       ("has_time = ", "Rows with a date but no clock time cannot be placed in a 15-minute slot."),
                       ("utc=True", "Everything is converted to UTC.")],
                      "fetch_site"),
                 Item("src/data/sources/kaggle_pond.py", "fetch_site",
                      "Reads one pond's CSV and emits the long (timestamp, channel, value) frame the rest "
                      "of the pipeline expects.",
                      [("if cfg.get(\"use_labels\", False):", "Quality labels are refused outright: training on "
                        "them would be circular."),
                       ("csvs = _site_files(", "Finds the file for this pond, e.g. IoTPond7.csv for pond07."),
                       ("tcol = next(", "Uses whichever timestamp column this particular file has."),
                       ("_map_columns(list(df.columns), cfg)", "Only mapped channels are kept; nitrate is never mapped."),
                       ("float(\"inf\")", "Infinite ammonia readings in the raw files are discarded.")],
                      "src/data/collect.py → collect"),
             ]),
        # ------------------------------------------------------------------ 3
        Step("3", "Cleaning, quality control and the availability mask",
             "Raw probe data is resampled to a 15-minute grid, impossible values are removed, stuck "
             "probes are masked, and a mask records which values are genuine readings.",
             [
                 Item("src/data/harmonize.py", "harmonize_site",
                      "Long frame in, wide 15-minute frame plus a True/False mask out. The mask is the "
                      "contract the neural network relies on.",
                      [("unstack(\"channel\")", "One column per channel."),
                       ("wide.resample(rule).mean()", "Onto a regular 15-minute grid."),
                       ("_drop_impossible(", "Drops -127 °C, pH below 3, and other physically impossible values."),
                       ("_censor_detection_limit(", "An exact 0.0 DO is a censored reading, not a measurement."),
                       ("_flag_flatlines(", "A probe stuck on one value for 12 h is broken, not calm."),
                       ("mask = wide.notna()", "The mask is taken BEFORE gap filling, so a filled value is "
                        "never mistaken for a reading."),
                       ("wide.ffill(limit=", "Short gaps (up to 1 h) are filled; longer ones stay empty.")],
                      "collect"),
                 Item("src/data/collect.py", "_qualifies",
                      "Decides whether a pond is good enough to keep. With the current settings 7 of the 11 "
                      "pond files pass.",
                      [("must_have.issubset", "DO is required: it is the channel the project is about."),
                       ("n_useful < int(targets[\"min_channels\"])", "At least 3 useful channels."),
                       ("len(values) < int(targets[\"min_rows\"])", "At least 2,000 15-minute rows."),
                       ("miss > float(targets[\"max_missing_frac\"])", "At most 65 % empty slots - mostly "
                        "logger outage.")],
                      "collect"),
                 Item("src/data/registry.py", "load_site",
                      "Reads one cleaned pond back from parquet into a SiteDataset. This is the only way "
                      "the rest of the project reaches real data.",
                      [("mask_cols = [c for c in df.columns if c.startswith(\"mask_\")]",
                        "The mask travels in the same file as the values."),
                       ("return SiteDataset(", "One pond = one dataset = one federated client.")],
                      "build_clients, calibrate, pretraining"),
             ]),
        # ------------------------------------------------------------------ 4
        Step("4", "The simulated pond (digital twin)",
             "The pond the agent controls is a simulation, because actions cannot be tried on a real "
             "pond. It advances in 5-minute steps of water chemistry and responds to every action.",
             [
                 Item("src/twin/dynamics.py", "do_rate",
                      "The dissolved-oxygen budget: how fast oxygen is gained and lost right now, split into "
                      "named terms so the cause of a crash can be read off.",
                      [("reaeration = ", "Oxygen diffusing in across the surface."),
                       ("aeration = ", "The aerator: pushes DO toward saturation."),
                       ("photosynthesis = ", "Algae produce oxygen in proportion to light - zero at night."),
                       ("respiration = ", "Fish and algae consume oxygen, faster when warm (Arrhenius θ^(T-20))."),
                       ("nitrifier_demand = ", "Bacteria use 4.57 g of oxygen per g of ammonia they convert.")],
                      "PondTwin.step"),
                 Item("src/twin/dynamics.py", "nh3_fraction",
                      "Only un-ionised ammonia (NH3) is toxic, and its share of total ammonia depends on pH "
                      "and temperature (Emerson et al., 1975).",
                      [("pka = ", "Dissociation constant from water temperature."),
                       ("return float(1.0 / (1.0 + 10.0 ** (pka - float(ph))))", "Fraction that is toxic NH3.")],
                      "PondTwin.state, the risk bands"),
                 Item("src/twin/pond.py", "PondTwin.step",
                      "One 5-minute step of the whole pond: forcing, the action, every rate, then Euler "
                      "integration (value += rate × dt).",
                      [("pert = self.scenario.active(self.t_hours)", "Which scenario events are active now."),
                       ("light = D.light_fraction(", "Sunlight: a half-sine by day, zero at night."),
                       ("self._apply_discrete(action, pert)", "Feeding, liming, salt and water exchange."),
                       ("self.delivered_aerator = 0.0 if pert.get(\"aerator_fault\")",
                        "An aerator fault swallows the command: asked for, not delivered."),
                       ("self.s[\"do\"] += ox[\"total\"] * dt", "Euler step for oxygen; the other channels follow."),
                       ("self._add_noise(dt)", "Mean-reverting (Ornstein-Uhlenbeck) noise: looks like weather."),
                       ("self._clamp()", "Keeps every quantity physical (no negative oxygen).")],
                      "SensorStream.tick"),
                 Item("src/twin/pond.py", "PondTwin.copy",
                      "Forks the pond. The counterfactual and the planner both rely on this.",
                      [("copy.deepcopy(v)", "A fully independent copy of every state variable."),
                       ("twin.rng.bit_generator.state = self.rng.bit_generator.state",
                        "The random generator is copied by state, so the fork sees the SAME future noise.")],
                      "InternalModel.rollout, world-model data generation"),
             ]),
        # ------------------------------------------------------------------ 5
        Step("5", "The sensors",
             "The rest of the system never sees the simulated pond's true state. It sees what a probe "
             "would report: biased, noisy, rounded, sometimes missing, sometimes faulty.",
             [
                 Item("src/sensors/stream.py", "SensorStream.sample",
                      "Turns one true state into one observation.",
                      [("value = truth + noise + self.bias[c]", "Random noise plus a fixed per-run calibration offset."),
                       ("value, dropped = self._apply_fault(c, value, fault)", "Scenario-driven probe faults."),
                       ("if self.rng.random() < float(self.cfg[\"dropout_prob\"])", "Random dropouts become NaN."),
                       ("value = round(value / step) * step", "Probe resolution (quantisation)."),
                       ("mask[c] = True", "Every reading carries its availability flag.")],
                      "SensorStream.tick"),
                 Item("src/sensors/stream.py", "SensorStream._apply_fault",
                      "The sensor_fault scenario corrupts the reading, never the water.",
                      [("if mode == \"dropout\":", "The probe goes silent."),
                       ("if mode == \"spike\":", "Random large jumps."),
                       ("if mode == \"stuck\":", "Freezes on the first value seen.")],
                      "SensorStream.sample"),
             ]),
        # ------------------------------------------------------------------ 6
        Step("6", "The LSTM forecaster",
             "Reads 36 steps of history (9 h on the 15-minute pond data) and predicts the next 24 steps "
             "for six channels.",
             [
                 Item("src/perception/forecaster.py", "LSTMForecaster.make_windows",
                      "Cuts a long series into training examples: input window, target window and target mask.",
                      [("if not np.isfinite(last[self.anchor_idx]).all():", "A window is used only if DO and "
                        "temperature are real at its last step - the origin of the prediction."),
                       ("(win - last) / self.scales", "Inputs are changes from the last value divided by FIXED "
                        "scales - never absolute levels."),
                       ("win_mask.astype(np.float32)", "The availability mask is fed in as extra channels."),
                       ("ys.append(np.nan_to_num((tgt - last) / self.scales", "Targets are scaled changes too.")],
                      "PondClient.__init__"),
                 Item("src/perception/forecaster.py", "_LSTMNet",
                      "The network: one LSTM layer of 64 units and a linear head.",
                      [("input_size=2 * n_channels", "6 values + 6 mask flags per time step."),
                       ("self.head = nn.Linear(hidden, horizon * n_channels)", "24 steps × 6 channels at once."),
                       ("y = self.head(out[:, -1, :])", "Uses the LSTM's state after the last input step.")],
                      "LSTMForecaster"),
                 Item("src/perception/forecaster.py", "LSTMForecaster.fit",
                      "Training loop with a masked loss: absent channels contribute nothing.",
                      [("loss = (((pred - yt[idx]) ** 2) * w).sum() / w.sum().clamp(min=1.0)",
                        "Masked mean-squared error: only real readings are scored."),
                       ("nn.utils.clip_grad_norm_", "Gradient clipping keeps training stable on noisy data."),
                       ("torch.Generator().manual_seed(self.seed)", "Deterministic shuffling.")],
                      "PondClient.local_train, run_central"),
                 Item("src/perception/forecaster.py", "LSTMForecaster.predict_array",
                      "Live inference: one window in, a forecast in real units out.",
                      [("filled = _forward_fill(window)", "Interior gaps are filled for input only."),
                       ("return last + delta * self.scales", "Predicted changes are turned back into levels.")],
                      "Perception.run"),
             ]),
        # ------------------------------------------------------------------ 7
        Step("7", "Federated learning across the real ponds",
             "Each pond is a client that trains on its own data. Only weights and a sample count ever "
             "reach the server.",
             [
                 Item("src/federated/client.py", "PondClient.__init__",
                      "One pond, one model. The validation tail is the end of the series.",
                      [("values = site.matrix(self.model.channels)", "This pond's data, and only this pond's."),
                       ("n_val = int(n * float(val_fraction))", "Last 15 % held back for this pond's own evaluation."),
                       ("if self.data_fraction < 1.0:", "Data-poor clients keep only the oldest slice.")],
                      "build_clients"),
                 Item("src/federated/simulate.py", "split_clients",
                      "Holds whole ponds out. No regime ever trains on them.",
                      [("n_hold = max(", "With 7 ponds and the current config: 2 held out."),
                       ("holdout_idx.append(int(rng.choice(pool)))", "A seeded, reproducible pick.")],
                      "compare"),
                 Item("src/federated/simulate.py", "fedavg",
                      "The aggregation rule: a sample-weighted average of client weights.",
                      [("w = counts / counts.sum()", "A pond with more data gets proportionally more say."),
                       ("_clip_updates(", "Optional differential privacy: bound each client's influence."),
                       ("avg = np.tensordot(w, stack, axes=(0, 0))", "The weighted average, tensor by tensor."),
                       ("rng.normal(0.0, sigma", "Optional Gaussian noise for differential privacy.")],
                      "run_federated, pretrain_world_model"),
                 Item("src/federated/simulate.py", "run_federated",
                      "The FedAvg loop: broadcast, train locally, average, repeat.",
                      [("global_weights = train[0].get_weights()", "Everyone starts from the SAME weights - "
                        "averaging differently initialised networks is meaningless."),
                       ("c.set_weights(global_weights)", "Broadcast the current global model."),
                       ("info = c.local_train(epochs=local_epochs)", "Each pond trains on its own data only."),
                       ("global_weights = fedavg(", "The server averages; it never sees a reading.")],
                      "compare"),
                 Item("src/federated/simulate.py", "compare",
                      "Runs the three regimes - independent-local, federated, central - over the same ponds "
                      "and scores them on the held-out ponds.",
                      [("train, holdout = split_clients(clients, fed_cfg)", "Split first ..."),
                       ("poor_ids = apply_data_poor(train, fed_cfg, verbose)", "... then designate data-poor "
                        "clients, from the training ponds only."),
                       ("local_weights = run_independent_local(", "Baseline: each pond alone."),
                       ("fed_weights, fed_log = run_federated(", "The method under test."),
                       ("central_weights = run_central(", "Upper reference that breaks privacy."),
                       ("_score_holdout(", "Scored on ponds nobody trained on.")],
                      "scripts/pretrain_federated.py"),
             ]),
        # ------------------------------------------------------------------ 8
        Step("8", "The action-conditioned world model",
             "The forecaster predicts what the water does on its own. The world model predicts what it "
             "does under a candidate action. The real data has no actuator log, so it learns actions "
             "from twin rollouts, starting from the federated forecaster's encoder.",
             [
                 Item("src/perception/world_model.py", "action_features",
                      "Encodes one action as 8 numbers.",
                      [("float(action.aerator)", "Aerator level 0-1."),
                       ("float(np.clip(hold_fraction, 0.0, 1.0))", "How long the action is held: the planner "
                        "holds a candidate, then stops.")],
                      "ActionWorldModel.predict"),
                 Item("src/perception/world_model.py", "_WorldNet",
                      "LSTM encoder for the history, then the action is joined in.",
                      [("nn.Linear(hidden + N_ACTION_FEATURES, hidden)", "The action enters AFTER the "
                        "history is encoded - it has not happened yet during the window."),
                       ("h = torch.cat([out[:, -1, :], a], dim=1)", "Encoded history + action.")],
                      "ActionWorldModel"),
                 Item("src/perception/world_model.py", "ActionWorldModel.warm_start_encoder",
                      "Transfer learning: copies the federated forecaster's LSTM into this encoder.",
                      [("if k.startswith(\"lstm.\")}",
                        "Takes only the LSTM part of the forecaster."),
                       ("self.net.lstm.load_state_dict(enc, strict=True)", "Same shapes, so it loads directly.")],
                      "scripts/pretrain_world_model.py"),
                 Item("scripts/pretrain_world_model.py", "make_dataset",
                      "Generates (history, action, outcome) examples by forking the twin.",
                      [("act = candidates[int(rng.integers(0, len(candidates)))]", "A random but realistic "
                        "action stream warms the pond up."),
                       ("sim = twin.copy()", "Fork the pond ..."),
                       ("sim.scenario = None", "... without knowledge of upcoming scenario events ..."),
                       ("s2 = sim.step(cand if k < hold_steps else Action.no_op())", "... apply the candidate, "
                        "then release it, and record what happens.")],
                      "PondWMClient"),
             ]),
        # ------------------------------------------------------------------ 9
        Step("9", "Anomaly detection, risk and explanation",
             "Three perception models run on every decision: HDBSCAN flags windows unlike anything seen, "
             "LightGBM scores risk, and SHAP explains that score.",
             [
                 Item("src/perception/regimes.py", "RegimeModel.derive",
                      "Turns raw readings into quantities that mean the same thing in any pond.",
                      [("cols.append(do / np.where(sat > 0, sat, np.nan))", "DO as a fraction of saturation."),
                       ("cols.append(temp - np.nanmean(temp))", "Temperature as a within-window anomaly."),
                       ("cols.append(np.log1p(np.clip(tb, 0, None)))", "Log turbidity: heavy-tailed.")],
                      "featurize_window"),
                 Item("src/perception/regimes.py", "RegimeModel.assign_features",
                      "Live assignment of one window; the noise label is the anomaly flag.",
                      [("return -1, \"anomaly (incomplete reading)\", True", "Unusable input is itself flagged."),
                       ("hdbscan.approximate_predict(self.clusterer, z)", "Assign without refitting."),
                       ("return k, self.names.get(k, f\"regime {k}\"), k == -1", "Label -1 = anomaly.")],
                      "Perception.run"),
                 Item("src/perception/risk.py", "build_training_set",
                      "Builds the risk model's training rows. The label is what the pond ACTUALLY did over "
                      "the next horizon, judged by the rule in twin.yaml.",
                      [("fc = forecaster.predict_array(window)", "Features use the forecast ..."),
                       ("future = pd.DataFrame(arr[i : i + H]", "... but the label uses the real future."),
                       ("worst = max(levels", "The worst risk band reached in the horizon.")],
                      "scripts/pretrain_federated.py"),
                 Item("src/perception/risk.py", "RiskModel.predict",
                      "Live risk: a class and a 0-1 severity score.",
                      [("if self.model is None:", "No trained model: fall back to the rule itself."),
                       ("proba = self.model.predict_proba(X)[0]", "Probability of safe / warning / critical."),
                       ("score = float(sum(p * self.weights.get(c, 0.0)", "Probability-weighted severity: "
                        "risk can be seen building before a threshold is crossed.")],
                      "Perception.run"),
                 Item("src/perception/explain.py", "RiskExplainer.top_drivers",
                      "SHAP attribution: which inputs pushed the risk toward critical this tick.",
                      [("contrib = self.shap_row(row)", "Exact TreeSHAP values for the critical class."),
                       ("order = np.argsort(-np.abs(contrib))[: self.top_k]", "Largest contributions first.")],
                      "Perception.run → dashboard and agent rationale"),
             ]),
        # ------------------------------------------------------------------ 10
        Step("10", "Planning: choosing an action",
             "Every 15 minutes the predictive planner rolls 14 candidate actions forward through the "
             "world model and takes the cheapest outcome.",
             [
                 Item("src/agent/actuators.py", "candidates",
                      "The 14 actions the agent may consider - a readable list instead of all 108 combinations.",
                      [("Action(),", "Doing nothing is always an option."),
                       ("Action(aerator=hi, feed=\"skip\"),", "Escalated response to a DO threat."),
                       ("Action(lime=\"dose\"),", "pH correction.")],
                      "PredictivePlanner"),
                 Item("src/agent/policy.py", "LearnedWorldModel.rollout",
                      "Answers \"what happens if I do X?\" with the neural network.",
                      [("if len(self.history) < self.wm.window", "Not enough history yet: use the physics model."),
                       ("pred = self.wm.predict(window, action, hold_fraction)", "The network's prediction."),
                       ("np.clip(v, *_PHYS_BOUNDS[c])", "Clipped to physical limits so a wild prediction "
                        "cannot dominate the cost.")],
                      "PredictivePlanner.evaluate"),
                 Item("src/agent/policy.py", "PredictivePlanner.evaluate",
                      "Scores one candidate's predicted trajectory.",
                      [("traj = ctx.model.rollout(action, steps, hold)", "Predicted future under this action."),
                       ("traj[c] = traj[c] + delta", "Optional LSTM residual correction."),
                       ("levels = RB.series_risk(frame, self.twin_cfg)", "Risk band at every future step."),
                       ("float(self.w[\"risk\"]) * (time_warn + time_crit)", "Cost = time at risk ..."),
                       ("float(self.w[\"critical\"]) * time_crit", "... critical time weighted 3× ..."),
                       ("float(self.w[\"intervention\"]) * ACT.action_cost", "... operating cost ..."),
                       ("float(self.w[\"feed_loss\"]) * ACT.feed_penalty", "... and lost growth, or the agent "
                        "would starve the fish to stay safe.")],
                      "PredictivePlanner.propose"),
                 Item("src/agent/policy.py", "PredictivePlanner.propose",
                      "Scores all candidates and returns the cheapest with a plain-language reason.",
                      [("scored = [self.evaluate(ctx, a, corr) for a in self.candidates]", "All 14 scored."),
                       ("scored.sort(key=lambda r: r[\"cost\"])", "Lowest cost first."),
                       ("why = (", "The rationale shown in the Agent view.")],
                      "AgentRun._propose"),
             ]),
        # ------------------------------------------------------------------ 11
        Step("11", "The safety guardrails",
             "No proposal reaches the pond without passing these fixed rules. They live outside the "
             "learned components, so a wrong model cannot override them.",
             [
                 Item("src/agent/guardrails.py", "Guardrails.check",
                      "The full rule stack, in a fixed order, at every decision.",
                      [("self._apply_dosing_caps(", "1. Daily caps and minimum intervals for lime, salt, exchange."),
                       ("self._apply_rate_limits(", "2. No step changes in aeration."),
                       ("self._apply_context_blocks(", "3. No lime into alkaline water, no feeding at low DO ..."),
                       ("self._reject_predicted_critical(", "4. Reject an action predicted to cause a critical state."),
                       ("self._emergency_override(", "5. Last, so it wins: DO critical → aerator to max.")],
                      "AgentRun._decide"),
                 Item("src/agent/guardrails.py", "Guardrails._emergency_override",
                      "When the pond is starving of oxygen, nothing outranks air.",
                      [("do = self._pessimistic(obs, \"do\", low=True)", "Judged on the probe minus a noise "
                        "margin, so it fires slightly early, never late."),
                       ("elif np.isfinite(fc_do) and fc_do < float(c[\"emergency_forecast_do\"])",
                        "Also fires on a forecast crash, before it happens."),
                       ("return replace(a, aerator=1.0, feed=\"skip\")", "Aerator full on, feeding stopped.")],
                      "check, safety_only"),
                 Item("src/agent/guardrails.py", "Guardrails.safety_only",
                      "The fast safety layer: runs on every 5-minute reading, between planning decisions.",
                      [("out = self._apply_context_blocks(", "Context blocks ..."),
                       ("out = self._emergency_override(", "... and the emergency override, every tick.")],
                      "AgentRun.tick"),
             ]),
        # ------------------------------------------------------------------ 12
        Step("12", "The control loop and the counterfactual",
             "AgentRun ties everything together: pond, sensors, perception, planner, guardrails - and a "
             "shadow pond with the agent switched off.",
             [
                 Item("src/agent/loop.py", "AgentRun.tick",
                      "One 5-minute tick of the whole system.",
                      [("state, obs = self.stream.tick(self.current_action)", "The pond moves; the probe reads it."),
                       ("self.shadow.tick(Action.no_op())", "The shadow pond moves with no action at all."),
                       ("decide = self.tick_index % self.ticks_per_decision == 0", "Plan every 3rd tick (15 min) ..."),
                       ("decision = self._decide(state, obs, observed)", "... perceive, plan, guard."),
                       ("held, events = self.guardrails.safety_only(", "Between decisions, safety still runs."),
                       ("decision.unsafe_applied = self._is_unsafe(state, decision)", "Every applied action is "
                        "checked against the pond's TRUE state.")],
                      "AgentRun.run, the desktop app"),
                 Item("src/agent/loop.py", "AgentRun._decide",
                      "Perceive → propose → guard.",
                      [("percept = self.perception.run(history, observed, self.twin_cfg)", "Forecast, anomaly, risk, SHAP."),
                       ("proposed, rationale = self._propose(observed, percept)", "The planner's proposal."),
                       ("applied, events = self.guardrails.check(", "The guardrails decide what is applied.")],
                      "AgentRun.tick"),
                 Item("src/agent/loop.py", "AgentRun._observed_dict",
                      "What the controller is allowed to see: the probe, never the truth.",
                      [("(h.values[c] for h in reversed(history[:-1])", "A missing reading falls back to the "
                        "last good one - not to the true value."),
                       ("out[\"nh3\"] = float(derive_nh3(", "Toxic ammonia is derived from observed values.")],
                      "AgentRun.tick"),
                 Item("src/agent/loop.py", "AgentRun._is_unsafe",
                      "The definition of an unsafe action - the number that must stay at zero.",
                      [("a.feed == \"normal\" and np.isfinite(state.do)", "Feeding into low-oxygen water."),
                       ("a.lime == \"dose\" and np.isfinite(state.ph)", "Liming an alkaline pond."),
                       ("a.aerator < 1.0", "Not running the aerator flat out in a DO emergency.")],
                      "AgentRun.tick"),
             ]),
        # ------------------------------------------------------------------ 13
        Step("13", "Evaluation and statistics",
             "Every reported number comes from here and is written to outputs/*.csv.",
             [
                 Item("src/eval/run_eval.py", "run_grid",
                      "8 scenarios × 5 controllers × 10 seeds.",
                      [("set_seed(seed)", "Same seed → same perturbations and noise for every controller."),
                       ("run = AgentRun(sid, controller=ctrl, seed=seed, perception=perception).run()",
                        "One complete run."),
                       ("m = run.metrics()", "Time in risk, interventions, cost, unsafe actions.")],
                      "main"),
                 Item("src/eval/run_eval.py", "run_ablations",
                      "Removes one component at a time to measure what it contributes.",
                      [("{\"name\": \"no_lstm_correction\"", "Planner without the forecaster's correction."),
                       ("{\"name\": \"no_guardrails\"", "The safety layer switched off."),
                       ("\"world_model\": False}", "Hand-written physics instead of the world model.")],
                      "main"),
                 Item("src/eval/metrics.py", "wilcoxon",
                      "Paired significance test across seeds; refuses to test with too few seeds.",
                      [("if n < MIN_N_FOR_TEST:", "Fewer than 6 seeds: report the effect, no p-value."),
                       ("if nonzero == 0:", "Identical results: an honest tie, p = 1."),
                       ("stats.wilcoxon(a, b", "The test itself."),
                       ("\"effect_size_rb\": _rank_biserial(diff)", "Effect size, not just significance.")],
                      "compare_controllers"),
             ]),
        # ------------------------------------------------------------------ 14
        Step("14", "The desktop operator console",
             "A PySide6 + pyqtgraph app that drives the same AgentRun and draws it live.",
             [
                 Item("desktop/engine.py", "RunEngine._on_timer",
                      "The playback clock: advances the pond at the requested speed.",
                      [("self._tick_debt += elapsed * self.speed", "Fractional ticks accumulate, so slow speeds "
                        "really are slow."),
                       ("capped = min(n, max(1, int(self.speed)))", "Never jump more than a second of sim time "
                        "per frame."),
                       ("self._advance(capped)", "Run the ticks ..."),
                       ("self.advanced.emit()", "... then tell the visible panel to redraw.")],
                      "QTimer, 30 times a second"),
                 Item("desktop/engine.py", "Columns.append",
                      "Pre-allocated arrays: drawing frame 2000 costs what frame 10 cost.",
                      [("if self.n >= self._cap:", "Grow by doubling when full."),
                       ("self._c[c][i] = np.nan if v is None else v", "One value per channel per tick.")],
                      "RunEngine._capture"),
                 Item("desktop/panels/timeseries.py", "_ChannelChart.update",
                      "One live chart: probe trace, forecast, risk bands and status pill.",
                      [("y_obs = obs_buf.series(ch, lo)", "The operator sees the probe, not the truth."),
                       ("fx, fy = _forecast_series(forecast, ch, now)", "The LSTM forecast ahead of \"now\"."),
                       ("self.level = channel_risk(ch, self.value, self.bands)", "The status pill uses the same "
                        "risk bands the agent is scored on.")],
                      "TimeSeriesPanel.update_view"),
             ]),
    ]


# ======================================================================================
# Rendering
# ======================================================================================
LEXER = PythonLexer()
FMT = HtmlFormatter(nowrap=True)


def hl(text: str) -> str:
    return highlight(text, LEXER, FMT).rstrip("\n") or "&nbsp;"


def render_item(step: Step, k: int, it: Item, index_rows: list) -> str:
    sn = extract(it.path, it.qualname, only_head=it.only_head)
    # locate every marker
    marks_at: Dict[int, List[int]] = {}
    for m_i, (frag, _) in enumerate(it.marks, start=1):
        hit = next((pos for pos, (ln, txt) in enumerate(sn.lines) if ln and frag in txt), None)
        if hit is None:
            raise ValueError(f"marker not found in {it.path}::{it.qualname}: {frag!r}")
        marks_at.setdefault(hit, []).append(m_i)

    rows = []
    for pos, (ln, txt) in enumerate(sn.lines):
        badge = "".join(f'<span class="mk">{n}</span>' for n in marks_at.get(pos, []))
        cls = "fold" if ln is None else ("hit" if pos in marks_at else "")
        rows.append(f'<tr class="{cls}"><td class="ln">{ln or "⋮"}</td><td class="bd">{badge}</td>'
                    f'<td class="cd">{hl(txt) if ln else html.escape(txt.strip())}</td></tr>')

    anchor = f"s{step.num}-{k}"
    index_rows.append((step.num, step.title, it.qualname, it.path, sn.start, sn.end, anchor))
    notes = "".join(f"<li><span class='mk'>{i}</span> {html.escape(note)}</li>"
                    for i, (_, note) in enumerate(it.marks, start=1))
    called = (f"<div class='called'>Called from: <code>{html.escape(it.called_from)}</code></div>"
              if it.called_from else "")
    return (f'<section class="item" id="{anchor}">'
            f'<div class="loc"><span class="fn">{html.escape(it.qualname)}</span>'
            f'<span class="path">{html.escape(it.path)} · lines {sn.start}–{sn.end}</span></div>'
            f'<p class="what">{it.what}</p>{called}'
            f'<table class="code">{"".join(rows)}</table>'
            f'<ol class="notes">{notes}</ol></section>')


PIPELINE = [
    ("Offline · run once", [
        ("2–3", "Real pond data", "kaggle_pond · harmonize"),
        ("6–7", "Federated forecaster", "forecaster · simulate"),
        ("8", "World model", "world_model"),
        ("9", "Anomaly + risk models", "regimes · risk"),
    ]),
    ("Live · every 5 minutes", [
        ("4–5", "Simulated pond + sensors", "pond · stream"),
        ("9", "Perception", "forecast · anomaly · risk · SHAP"),
        ("10", "Planner", "policy"),
        ("11", "Guardrails", "guardrails"),
        ("12", "Loop + shadow pond", "loop"),
    ]),
    ("Results and interface", [
        ("13", "Evaluation", "run_eval · metrics"),
        ("14", "Desktop console", "desktop/"),
    ]),
]


def build_html() -> str:
    import csv

    with (ROOT / "data_cache" / "manifest.csv").open(encoding="utf-8") as fh:
        ponds = list(csv.DictReader(fh))
    all_steps = steps()
    index_rows: list = []
    body = []
    for st in all_steps:
        parts = [f'<section class="step" id="step{st.num}"><h1><span class="sn">{st.num}</span>'
                 f"{html.escape(st.title)}</h1><p class='intro'>{st.intro}</p>"]
        for k, it in enumerate(st.items, start=1):
            parts.append(render_item(st, k, it, index_rows))
        parts.append("</section>")
        body.append("".join(parts))

    def box(n, t, m):
        return (f'<div class="box"><div class="bn">Step {n}</div><div class="bt">{html.escape(t)}</div>'
                f'<div class="bm">{html.escape(m)}</div></div>')

    flow = "".join(
        f'<div class="lane"><div class="lane-l">{html.escape(label)}</div><div class="row">'
        + '<div class="arr">→</div>'.join(box(*b) for b in boxes) + "</div></div>"
        for label, boxes in PIPELINE)

    idx = "".join(
        f"<tr><td>{n}</td><td><a href='#{a}'><code>{html.escape(q)}</code></a></td>"
        f"<td><code>{html.escape(p)}</code></td><td>{s}–{e}</td></tr>"
        for n, _t, q, p, s, e, a in index_rows)

    css = HtmlFormatter(style="friendly").get_style_defs(".cd")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AquaSentinel - Code Walkthrough</title>
<style>
@page {{ size: A4; margin: 15mm 13mm 15mm 13mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; color: #16181d; line-height: 1.45; }}
h1 {{ font-size: 17pt; margin: 0 0 6px; display: flex; align-items: center; gap: 10px; }}
.sn {{ background: #0b0b0b; color: #fff; font-family: Consolas, monospace; font-size: 12pt;
       padding: 3px 9px; border-radius: 2px; }}
.step {{ page-break-before: always; }}
.intro {{ font-size: 10.2pt; color: #333; margin: 2px 0 12px; }}
.item {{ margin: 0 0 18px; break-inside: auto; }}
.loc {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 2px solid #0b0b0b;
        padding-bottom: 3px; margin-bottom: 5px; }}
.fn {{ font-family: Consolas, monospace; font-weight: 700; font-size: 11pt; }}
.path {{ font-family: Consolas, monospace; font-size: 8.4pt; color: #555; }}
.what {{ margin: 2px 0 4px; }}
.called {{ font-size: 8.4pt; color: #555; margin-bottom: 5px; }}
table.code {{ width: 100%; border-collapse: collapse; font-family: Consolas, monospace; font-size: 7.9pt;
              background: #fbfbf8; border: 1px solid #e3e1d8; }}
table.code td {{ padding: 0 5px; vertical-align: top; white-space: pre-wrap; word-break: break-word; }}
td.ln {{ color: #9a988f; text-align: right; width: 30px; user-select: none; border-right: 1px solid #e3e1d8; }}
td.bd {{ width: 22px; }}
tr.hit td.cd {{ background: #fff4c2; }}
tr.fold td {{ color: #9a988f; font-style: italic; }}
.mk {{ display: inline-block; background: #0b0b0b; color: #fff; font-family: Consolas, monospace;
       font-size: 7pt; font-weight: 700; border-radius: 8px; min-width: 14px; text-align: center;
       padding: 0 3px; margin-right: 2px; }}
ol.notes {{ list-style: none; padding: 0; margin: 6px 0 0; }}
ol.notes li {{ margin: 2px 0; }}
.cover h1 {{ font-size: 30pt; display: block; margin-top: 40px; }}
.sub {{ font-size: 12pt; color: #333; }}
.flow {{ margin: 12px 0; }}
.lane {{ margin: 0 0 8px; }}
.lane-l {{ font-family: Consolas, monospace; font-size: 8pt; font-weight: 700; color: #555;
           text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }}
.row {{ display: flex; align-items: stretch; gap: 4px; }}
.box {{ border: 1.5px solid #0b0b0b; padding: 5px 7px; flex: 1 1 0; background: #efede4; }}
.bn {{ font-family: Consolas, monospace; font-size: 7.5pt; color: #555; }}
.bt {{ font-weight: 700; font-size: 9.5pt; }}
.bm {{ font-family: Consolas, monospace; font-size: 7.5pt; color: #444; }}
.arr {{ font-size: 13pt; align-self: center; }}
table.idx {{ width: 100%; border-collapse: collapse; font-size: 8.6pt; }}
table.idx th, table.idx td {{ border-bottom: 1px solid #ddd; padding: 3px 5px; text-align: left; }}
table.idx th {{ background: #0b0b0b; color: #fff; }}
a {{ color: inherit; text-decoration: none; }}
.note {{ border-left: 4px solid #0b0b0b; background: #f4f3ee; padding: 7px 10px; margin: 10px 0; }}
{css}
</style></head><body>
<section class="cover">
<h1>AquaSentinel — Code Walkthrough</h1>
<p class="sub">Where every key step of the project lives in the code, and what each important line does.</p>
<div class="note">Every snippet below is cut directly from the source files when this document is
built, with the real line numbers in the left-hand gutter. Numbered black markers sit on the lines
that matter; the notes under each snippet explain them. Long docstrings are folded to their first
line (marked with ⋮). Data: {len(ponds)} real catfish aquaponics ponds in Nsukka, Nigeria,
{sum(int(p['rows']) for p in ponds):,} rows. Generated {datetime.now().strftime('%d %B %Y')} by
<code>scripts/make_code_walkthrough_pdf.py</code>.</div>
<h2>How the pieces connect</h2>
<p>Each box names the step number below where its code is explained, and the modules it lives in.</p>
<div class="flow">{flow}</div>
<h2>Index — every step and where its code is</h2>
<table class="idx"><tr><th>Step</th><th>Function</th><th>File</th><th>Lines</th></tr>{idx}</table>
</section>
{''.join(body)}
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PDF)
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    doc = build_html()
    spec = importlib.util.spec_from_file_location("_mrp", Path(__file__).with_name("make_report_pdf.py"))
    mrp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrp)

    with tempfile.TemporaryDirectory() as td:
        html_path = Path(td) / "walkthrough.html"
        html_path.write_text(doc, encoding="utf-8")
        if args.keep_html:
            keep = args.out.with_suffix(".html")
            keep.write_text(doc, encoding="utf-8")
            print(f"html -> {keep}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        mrp._print_pdf(html_path, args.out)
    print(f"pdf  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
