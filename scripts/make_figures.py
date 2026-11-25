"""Render the report figures from outputs/ and a live twin run. Offline.

    python scripts/make_figures.py [--out report/figures]

Every figure reads its numbers from ``outputs/*.csv``, ``data_cache/manifest.csv`` or a
freshly-run scenario - none are typed in. SVG for the PDF, plus a PNG alongside so the figure
can actually be eyeballed rather than assumed correct.

Colour follows a validated categorical palette: slots are assigned in fixed order and never
cycled, one hue per entity, status colours are reserved for state and never reused as a series.
Three of the light slots sit below 3:1 against the chart surface, so the "relief rule" applies -
every figure has its underlying table in the report body, and bars carry selective direct
labels rather than a number on every mark.

**Bar geometry note.** Bar thickness is specified in *points*, converted from the axes'
physical size. Data-unit thickness is the obvious approach and it is wrong: when the x and y
scales differ by an order of magnitude - RMSE 0-0.4 against three category rows - a corner
radius in data units turns every bar into an ellipse. Bars are drawn as thick round-capped
lines clipped at x=0, which gives a rounded data end, a flat baseline end, and correct
geometry at any aspect ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.runtime import DATA_CACHE, OUTPUTS_DIR, ROOT, load_config, set_seed  # noqa: E402

# ---------------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, fixed order, never cycled.
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
# Diverging pair for signed deltas (blue <-> red, neutral gray midpoint).
DIV_GOOD, DIV_BAD, DIV_MID = "#2a78d6", "#d03b3b", "#e1e0d9"
# Status - reserved for state, never a series colour.
ST_CRITICAL, ST_WARNING, ST_GOOD = "#d03b3b", "#fab219", "#0ca30c"

CONTROLLER_COLOR = {
    "predictive": S1,
    "reactive": S2,
    "static_threshold": S3,
    "do_nothing": S4,
    "oracle": S5,
}
CONTROLLER_LABEL = {
    "predictive": "agent (predictive)",
    "reactive": "agent (reactive)",
    "static_threshold": "static DO threshold",
    "do_nothing": "do nothing",
    "oracle": "oracle (latent state)",
}
REGIME_COLOR = {"federated": S1, "central": S2, "independent_local": S3}
REGIME_LABEL = {
    "federated": "federated (FedAvg)",
    "central": "central (pooled)",
    "independent_local": "independent-local",
}


def setup() -> None:
    for family in ("Segoe UI", "Arial", "DejaVu Sans"):
        if any(family.lower() == f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.8,
        "legend.fontsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })


# ---------------------------------------------------------------------------- chrome
def strip(ax, value_axis: str = "x") -> None:
    """Recessive chrome: hairline grid on the value axis only, no box, no tick marks."""
    ax.grid(axis=value_axis, color=GRID, linewidth=0.7)
    ax.grid(axis="y" if value_axis == "x" else "x", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, pad=6)


def _pt_per_yunit(fig, ax) -> float:
    """Points of vertical space per y data unit. Exact from the axes' physical size."""
    y0, y1 = ax.get_ylim()
    height_in = ax.get_position().height * fig.get_figheight()
    return height_in * 72.0 / abs(y1 - y0)


def hbars(fig, ax, specs: Iterable[Tuple[float, float, str]], height: float) -> None:
    """Horizontal bars as thick round-capped lines. Call AFTER set_xlim/set_ylim.

    ``specs`` is (y, value, colour); ``height`` is bar thickness in y data units. Clipping at
    the axes edge flattens the baseline end so only the data end reads round.
    """
    lw = max(height * _pt_per_yunit(fig, ax), 1.0)
    for y, value, color in specs:
        if value is None or not np.isfinite(value):
            continue
        if value <= 0:
            # A measured zero must not look like missing data: mark it at the baseline.
            ax.plot([0], [y], marker="|", markersize=max(lw * 0.7, 4.0),
                    markeredgewidth=1.8, color=color, linestyle="none", zorder=3)
            continue
        ax.plot([0, value], [y, y], color=color, lw=lw, solid_capstyle="round",
                clip_on=True, zorder=3)


def swatches(colors_labels: Sequence[Tuple[str, str]]):
    return [plt.Line2D([0], [0], marker="s", ls="", markersize=7, markerfacecolor=c,
                       markeredgecolor="none", label=l) for c, l in colors_labels]


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out / f"{name}.png", format="png", dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  {path.name}")
    return path


def nice(s: str) -> str:
    return (s.replace("_", " ").replace("do crash", "DO crash").replace("ph swing", "pH swing"))


# ======================================================================================
# F1 - the counterfactual
# ======================================================================================
def fig_counterfactual(out: Path) -> None:
    from src.agent.loop import AgentRun, Perception
    from src.eval.run_eval import load_perception

    set_seed(1002)
    try:
        perception = load_perception("federated")
    except Exception:
        perception = Perception()
    result = AgentRun("night_do_crash", controller="predictive", seed=1002,
                      perception=perception).run()
    live, shadow = result.frame(), result.shadow_frame()
    n = min(len(live), len(shadow))
    t = live["t_hours"].to_numpy()[:n]
    y_live = live["do"].to_numpy()[:n]
    y_shadow = shadow["do"].to_numpy()[:n]
    bands = load_config("twin")["risk_bands"]["do"]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.16)
    ax.axhspan(0, bands["critical_below"], color=ST_CRITICAL, alpha=0.10, lw=0, zorder=0)
    ax.axhspan(bands["critical_below"], bands["warning_below"], color=ST_WARNING,
               alpha=0.10, lw=0, zorder=0)
    ax.axhline(bands["critical_below"], color=ST_CRITICAL, lw=0.9, alpha=0.5, zorder=1)

    ax.plot(t, y_shadow, color=S2, lw=2.0, ls=(0, (5, 2.5)),
            label="no action (shadow pond)", zorder=3)
    ax.plot(t, y_live, color=S1, lw=2.0, label="with agent", zorder=4)

    ax.set_xlim(0, float(t[-1]))
    ax.set_ylim(0, max(float(np.nanmax(y_shadow)), float(np.nanmax(y_live))) * 1.24)

    # Two selective labels, placed into empty space rather than on top of a line.
    i_s, i_l = int(np.nanargmin(y_shadow)), int(np.nanargmin(y_live))
    ax.annotate(f"min {y_shadow[i_s]:.2f} mg/L", (t[i_s], y_shadow[i_s]),
                textcoords="offset points", xytext=(0, 15), ha="center",
                color=S2, fontsize=8.5, fontweight="semibold", zorder=6)
    ax.annotate(f"min {y_live[i_l]:.2f} mg/L", (t[i_l], y_live[i_l]),
                textcoords="offset points", xytext=(0, -17), ha="center",
                color=S1, fontsize=8.5, fontweight="semibold", zorder=6)
    ax.text(float(t[-1]) * 0.985, bands["critical_below"] * 0.45, "critical band",
            color=ST_CRITICAL, fontsize=8, ha="right", va="center")

    ax.set_xlabel("simulated hours")
    ax.set_ylabel("dissolved oxygen (mg/L)")
    ax.set_title("Nighttime DO crash — same seed, same perturbations, same sensor noise",
                 loc="left", color=INK, pad=10)
    strip(ax, value_axis="y")
    ax.legend(frameon=False, loc="upper left", ncols=2, handlelength=2.0,
              bbox_to_anchor=(0.0, 1.03))
    save(fig, out, "f1_counterfactual")


# ======================================================================================
# F2 - DO time-out-of-band by scenario x controller
# ======================================================================================
def fig_controllers(out: Path) -> List[str]:
    runs = pd.read_csv(OUTPUTS_DIR / "agent_runs.csv")
    order = ["predictive", "reactive", "static_threshold", "do_nothing", "oracle"]
    med = runs.pivot_table(index="scenario", columns="controller", values="risk_h_do",
                           aggfunc="median")
    med = med.reindex(columns=[c for c in order if c in med.columns])

    # Scenarios in which every controller scores zero carry no comparison; they belong in the
    # caption, not as ten empty rows with "0" printed on each.
    zero = med.index[med.sum(axis=1) == 0].tolist()
    med = med.drop(index=zero)
    med = med.loc[med.mean(axis=1).sort_values(ascending=False).index]

    n_ctrl = len(med.columns)
    band, gap_frac = 0.84, 0.14
    h = band / n_ctrl * (1 - gap_frac)
    pitch = band / n_ctrl
    xmax = float(med.to_numpy().max())

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    fig.subplots_adjust(left=0.23, right=0.97, top=0.91, bottom=0.22)
    ax.set_yticks(range(len(med.index)))
    ax.set_yticklabels([nice(s) for s in med.index])
    ax.set_ylim(-0.6, len(med.index) - 0.4)
    ax.set_xlim(0, xmax * 1.10)

    specs, labels = [], []
    for si, scenario in enumerate(med.index):
        row = med.loc[scenario]
        worst = row.idxmax()
        for ci, ctrl in enumerate(med.columns):
            v = float(row[ctrl])
            y = si + band / 2 - pitch / 2 - ci * pitch
            specs.append((y, v, CONTROLLER_COLOR[ctrl]))
            # Selective labels: the agent (the claim) and the worst controller (the reference).
            if ctrl == "predictive" or ctrl == worst:
                labels.append((y, v, CONTROLLER_COLOR[ctrl]))
    hbars(fig, ax, specs, h)
    for y, v, color in labels:
        ax.text(v + xmax * 0.012, y, f"{v:.1f}", va="center", ha="left",
                fontsize=7.4, color=color, fontweight="semibold")

    ax.set_xlabel("hours of dissolved oxygen outside its safe band\n"
                  "median of 10 seeds · lower is better · labelled: the agent and the worst controller\n"
                  "a tick at the baseline is a measured zero, not missing data")
    ax.set_title("Agent vs. baselines, per scenario", loc="left", color=INK, pad=10)
    strip(ax, value_axis="x")
    ax.legend(handles=swatches([(CONTROLLER_COLOR[c], CONTROLLER_LABEL[c]) for c in med.columns]),
              frameon=False, loc="upper center", bbox_to_anchor=(0.44, -0.22), ncols=3,
              handletextpad=0.4, columnspacing=1.6)
    save(fig, out, "f2_controllers")
    return zero


# ======================================================================================
# F3 - federated vs independent-local vs central
# ======================================================================================
def fig_federated(out: Path) -> None:
    summary = pd.read_csv(OUTPUTS_DIR / "federated_summary.csv")
    channels = [(k, lab) for k, lab in
                [("rmse_do", "dissolved oxygen  (mg/L)"),
                 ("rmse_temp_c", "temperature  (°C)"),
                 ("rmse_ph", "pH")] if k in summary.columns]
    order = ["federated", "central", "independent_local"]
    summary = summary.set_index("regime").reindex([r for r in order if r in set(summary.regime)])

    fig, axes = plt.subplots(1, len(channels), figsize=(7.2, 2.4))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.68, bottom=0.30, wspace=0.34)
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, channels):
        vals = summary[key]
        vmax = float(vals.max())
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([])
        ax.set_ylim(-0.66, len(vals) - 0.34)
        ax.set_xlim(0, vmax * 1.48)
        hbars(fig, ax, [(len(vals) - 1 - i, float(v), REGIME_COLOR[r])
                        for i, (r, v) in enumerate(vals.items())], 0.40)
        for i, (regime, v) in enumerate(vals.items()):
            ax.text(float(v) + vmax * 0.06, len(vals) - 1 - i, f"{v:.3f}", va="center",
                    ha="left", fontsize=7.8, color=REGIME_COLOR[regime], fontweight="semibold")
        ax.set_title(label, loc="left", fontsize=8.8, color=INK2, pad=6)
        strip(ax, value_axis="x")
    axes[0].set_xlabel("forecast RMSE on held-out real ponds — lower is better", loc="left")
    fig.legend(handles=swatches([(REGIME_COLOR[r], REGIME_LABEL[r]) for r in summary.index]),
               frameon=False, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.4)
    fig.text(0.02, 0.97, "Federation matches the pooled reference and beats training alone",
             ha="left", va="top", fontsize=10.5, color=INK)
    save(fig, out, "f3_federated")


# ======================================================================================
# F4 - forecast error by horizon
# ======================================================================================
def fig_horizon(out: Path) -> None:
    fh = pd.read_csv(OUTPUTS_DIR / "forecast_horizon.csv")
    piv = (fh[fh["channel"] == "do"]
           .pivot_table(index="horizon_minutes", columns="source", values="rmse"))
    order = [c for c in ("federated", "central", "independent_local") if c in piv.columns]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.85, bottom=0.18)
    for src in order:
        ax.plot(piv.index, piv[src], color=REGIME_COLOR[src], lw=2.0,
                label=REGIME_LABEL[src], solid_capstyle="round", zorder=3)
    ax.set_xlabel("forecast horizon (minutes ahead)")
    ax.set_ylabel("DO RMSE (mg/L)")
    ax.set_xlim(float(piv.index.min()), float(piv.index.max()))
    ax.set_ylim(0, float(piv.to_numpy().max()) * 1.20)

    # One annotation per finding, rather than three end-labels that collide. That federated
    # and central track each other almost exactly is itself the result worth pointing at.
    if {"federated", "central", "independent_local"} <= set(piv.columns):
        # Wording derived from the data, not asserted: independent-local is the worst of the
        # three at most horizons but not all of them, and "throughout" would be an overclaim.
        best_other = piv[["federated", "central"]].min(axis=1)
        above = int((piv["independent_local"] > best_other).sum())
        ia = int(len(piv) * 0.26)
        ax.annotate(f"independent-local is the worst of the\nthree at {above} of {len(piv)} horizons",
                    (float(piv.index[ia]), float(piv["independent_local"].iloc[ia])),
                    textcoords="offset points", xytext=(6, 34), fontsize=8,
                    color=S3, fontweight="semibold",
                    arrowprops=dict(arrowstyle="-", color=S3, lw=0.9, alpha=0.75))
        ib = int(len(piv) * 0.70)
        ax.annotate("federated and central\nnearly coincide",
                    (float(piv.index[ib]), float(piv["federated"].iloc[ib])),
                    textcoords="offset points", xytext=(-8, -36), ha="right", fontsize=8,
                    color=S1, fontweight="semibold",
                    arrowprops=dict(arrowstyle="-", color=S1, lw=0.9, alpha=0.75))

    ax.set_title("Forecast error grows with horizon, as it should", loc="left", color=INK, pad=10)
    strip(ax, value_axis="y")
    ax.legend(frameon=False, loc="upper left", ncols=3, handlelength=2.0,
              bbox_to_anchor=(0.0, 1.04))
    save(fig, out, "f4_horizon")


# ======================================================================================
# F5 - ablations
# ======================================================================================
def fig_ablations(out: Path) -> None:
    ab = pd.read_csv(OUTPUTS_DIR / "ablations.csv")
    agg = (ab.groupby("ablation")
             .agg(do_risk=("risk_h_do", "median"), unsafe=("unsafe_applied", "sum"))
             .sort_values("do_risk"))
    pretty = {
        "full": "shipped system",
        "no_lstm_correction": "without LSTM correction",
        "no_world_model": "hand-written pond model",
        "no_forecast_at_all": "no forecaster at all",
        "no_guardrails": "no guardrails",
        "weights_independent_local": "independent-local weights",
        "weights_central": "central weights",
    }
    xmax = float(agg["do_risk"].max())

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fig.subplots_adjust(left=0.26, right=0.98, top=0.85, bottom=0.20)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([pretty.get(n, n) for n in reversed(agg.index)])
    ax.set_ylim(-0.64, len(agg) - 0.36)
    ax.set_xlim(0, xmax * 1.74)

    hbars(fig, ax, [(len(agg) - 1 - i, float(r["do_risk"]),
                     ST_CRITICAL if int(r["unsafe"]) > 0 else S1)
                    for i, (_, r) in enumerate(agg.iterrows())], 0.44)

    for i, (name, row) in enumerate(agg.iterrows()):
        y = len(agg) - 1 - i
        v, unsafe = float(row["do_risk"]), int(row["unsafe"])
        ax.text(v + xmax * 0.03, y, f"{v:.2f} h", va="center", ha="left", fontsize=8, color=INK2)
        if unsafe > 0:
            x0 = v + xmax * 0.17
            ax.plot([x0], [y], marker="X", markersize=6.5, color=ST_CRITICAL,
                    linestyle="none", clip_on=False, zorder=5)
            ax.text(x0 + xmax * 0.04, y, f"{unsafe} unsafe actions applied", va="center",
                    ha="left", fontsize=8, color=ST_CRITICAL, fontweight="semibold")

    ax.set_xlabel("median hours of DO outside its safe band  (40 runs per variant)")
    ax.set_title("Ablations — the forecaster correction hurts; the guardrails are load-bearing",
                 loc="left", color=INK, pad=10)
    strip(ax, value_axis="x")
    save(fig, out, "f5_ablations")


# ======================================================================================
# F6 - risk / cost trade-off (one measure per panel; never a dual axis)
# ======================================================================================
def fig_tradeoff(out: Path) -> None:
    stats = pd.read_csv(OUTPUTS_DIR / "agent_stats.csv")
    sub = stats[(stats.controller_a == "predictive") & (stats.controller_b == "static_threshold")]
    risk = sub[sub.metric == "risk_h_do"].set_index("scenario")["median_diff"]
    cost = sub[sub.metric == "intervention_cost"].set_index("scenario")["median_diff"]
    scen = risk.sort_values().index.tolist()
    rmax, cmax = float(risk.abs().max()), float(cost.abs().max())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.17, wspace=0.30)
    a1.set_yticks(range(len(scen)))
    a1.set_yticklabels([nice(s) for s in reversed(scen)])
    a1.set_ylim(-0.64, len(scen) - 0.36)
    a1.set_xlim(0, rmax * 1.30)
    a2.set_xlim(0, cmax * 1.32)

    hbars(fig, a1, [(len(scen) - 1 - i, abs(float(risk.get(s, 0.0))),
                     DIV_GOOD if float(risk.get(s, 0.0)) < 0 else DIV_MID)
                    for i, s in enumerate(scen)], 0.44)
    hbars(fig, a2, [(len(scen) - 1 - i, abs(float(cost.get(s, 0.0))),
                     DIV_BAD if float(cost.get(s, 0.0)) > 0 else DIV_MID)
                    for i, s in enumerate(scen)], 0.44)

    for i, s in enumerate(scen):
        y = len(scen) - 1 - i
        rv, cv = float(risk.get(s, 0.0)), float(cost.get(s, 0.0))
        a1.text(abs(rv) + rmax * 0.03, y, f"{rv:+.1f}" if rv else "no change",
                va="center", ha="left", fontsize=7.6, color=DIV_GOOD if rv else MUTED)
        a2.text(abs(cv) + cmax * 0.03, y, f"{cv:+.0f}", va="center", ha="left",
                fontsize=7.6, color=DIV_BAD if cv > 0 else MUTED)

    a1.set_title("hours of DO risk avoided", loc="left", fontsize=9.3, color=DIV_GOOD, pad=6)
    a2.set_title("extra operating cost incurred", loc="left", fontsize=9.3, color=DIV_BAD, pad=6)
    a1.set_xlabel("hours  (agent − thermostat)")
    a2.set_xlabel("cost units  (agent − thermostat)")
    for ax in (a1, a2):
        strip(ax, value_axis="x")
    fig.text(0.02, 0.96, "What the agent buys, and what it pays — vs. the static DO threshold",
             ha="left", va="top", fontsize=10.5, color=INK)
    save(fig, out, "f6_tradeoff")


# ======================================================================================
# F7 - channel coverage across the client set
# ======================================================================================
def fig_coverage(out: Path) -> None:
    mf = pd.read_csv(DATA_CACHE / "manifest.csv")
    counts: Dict[str, int] = {}
    for chans in mf["channels"]:
        for c in str(chans).split("|"):
            counts[c] = counts.get(c, 0) + 1
    ser = pd.Series(counts).sort_values(ascending=False)
    label = {"do": "dissolved oxygen", "temp_c": "temperature", "ph": "pH",
             "turbidity": "turbidity", "conductivity": "conductivity",
             "tan": "ammonium (as TAN)", "no3": "nitrate", "chlorophyll": "chlorophyll",
             "bga": "blue-green algae"}
    total = len(mf)

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.80, bottom=0.19)
    ax.set_yticks(range(len(ser)))
    ax.set_yticklabels([label.get(c, c) for c in reversed(ser.index)])
    ax.set_ylim(-0.64, len(ser) - 0.36)
    ax.set_xlim(0, total * 1.16)
    # One measure, one hue. The "n of N" labels carry the distinction, not colour.
    hbars(fig, ax, [(len(ser) - 1 - i, int(n), S1) for i, (c, n) in enumerate(ser.items())], 0.44)
    for i, (chan, n) in enumerate(ser.items()):
        ax.text(int(n) + total * 0.018, len(ser) - 1 - i, f"{n} of {total}", va="center",
                ha="left", fontsize=7.8, color=INK2)
    ax.set_xlabel(f"ponds reporting the channel  (of {total})")
    ax.set_title("Channel coverage is uneven — this is the non-IID structure,\n"
                 "not a simulation of it", loc="left", color=INK, pad=8)
    strip(ax, value_axis="x")
    save(fig, out, "f7_coverage")


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Render report figures (offline).")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    setup()
    print(f"Rendering figures -> {out}")
    for name, fn in [
        ("counterfactual", fig_counterfactual),
        ("controllers", fig_controllers),
        ("federated", fig_federated),
        ("horizon", fig_horizon),
        ("ablations", fig_ablations),
        ("tradeoff", fig_tradeoff),
        ("coverage", fig_coverage),
    ]:
        try:
            fn(out)
        except FileNotFoundError as exc:
            print(f"  ! skipped {name}: {exc}")
        except Exception as exc:
            print(f"  ! FAILED {name}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
