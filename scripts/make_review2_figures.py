"""Render the Review-2 report diagrams (Gantt, architecture, DFD, use case, class, sequence).

    python scripts/make_review2_figures.py [--out report/review2/figures]

These are *structural* diagrams - unlike ``scripts/make_figures.py`` they describe the system
rather than its measurements, so they are drawn rather than read from ``outputs/``. Every
component name here maps to a real module under ``src/``; if a box has no file behind it, that
is a bug in this script, not a design intention.

Palette and typography follow ``scripts/make_figures.py`` so the two figure sets sit together
in one document without looking like they came from different projects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# ---------------------------------------------------------------------------- data counts
def _counts():
    """(ponds, rows, training ponds) from the data and model manifests."""
    import csv
    import json

    root = Path(__file__).resolve().parents[1]
    with (root / "data_cache" / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    man = root / "models" / "manifest.json"
    n_train = len(json.loads(man.read_text(encoding="utf-8")).get("train_sites", [])) \
        if man.exists() else len(rows) - 1
    return len(rows), sum(int(r["rows"]) for r in rows), n_train


N_PONDS, N_ROWS, N_TRAIN = _counts()


# ---------------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
ST_CRITICAL = "#d03b3b"

FILL = {
    "data": "#e8f1fc",
    "twin": "#e6f6ef",
    "ml": "#fdeee7",
    "agent": "#fdf4e0",
    "ui": "#fbecf3",
    "plain": "#f4f3ef",
}
EDGE = {
    "data": S1,
    "twin": S3,
    "ml": S2,
    "agent": S4,
    "ui": S5,
    "plain": MUTED,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


# ---------------------------------------------------------------------------- primitives
def canvas(w: float, h: float, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    return fig, ax


def box(ax, x, y, w, h, text, kind="plain", fs=8.5, bold=False, rounding=0.6, ec=None, lw=1.2):
    """Rounded box centred on (x, y)."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=FILL[kind], edgecolor=ec or EDGE[kind], linewidth=lw, zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)


def sq(ax, x, y, w, h, text, kind="plain", fs=8.5, bold=False):
    """Square-cornered box - external entities."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=FILL[kind],
                           edgecolor=EDGE[kind], linewidth=1.2, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)


def store(ax, x, y, w, h, text, fs=8.5):
    """Open-ended data store (Gane-Sarson style)."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="#f4f3ef",
                           edgecolor="none", zorder=2))
    ax.plot([x - w / 2, x + w / 2], [y + h / 2, y + h / 2], color=MUTED, lw=1.2, zorder=3)
    ax.plot([x - w / 2, x + w / 2], [y - h / 2, y - h / 2], color=MUTED, lw=1.2, zorder=3)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3,
            linespacing=1.35)


def arrow(ax, p0, p1, text="", kind=None, fs=7.2, rad=0.0, style="-|>", ls="-",
          tx=None, ty=None, color=None, lw=1.15):
    c = color or (EDGE[kind] if kind else INK2)
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=c, lw=lw,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
                        shrinkA=2, shrinkB=2)
    )
    if text:
        mx = tx if tx is not None else (p0[0] + p1[0]) / 2
        my = ty if ty is not None else (p0[1] + p1[1]) / 2
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.4), zorder=4)


def elbow(ax, pts, text="", kind=None, fs=7.2, ls="-", tx=None, ty=None, lw=1.15):
    """Orthogonal multi-segment route with a single arrowhead at the last point."""
    c = EDGE[kind] if kind else INK2
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, ls=ls, zorder=1,
                solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=11, color=c,
                        lw=lw, linestyle=ls, zorder=1, shrinkA=0, shrinkB=2)
    )
    if text and tx is not None:
        ax.text(tx, ty, text, ha="center", va="center", fontsize=fs, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6), zorder=4)


def caption(ax, text, y=1.5):
    ax.text(50, y, text, ha="center", va="center", fontsize=8, color=MUTED, style="italic")


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  {name}.png")


# ---------------------------------------------------------------------------- 1. Gantt
def fig_gantt(out: Path):
    """Project-I plan. x is weeks from 1 July 2026; Review-2 falls on 09-09-2026 = week 10."""
    tasks = [
        ("Literature survey and gap identification", 0.0, 3.2, S1),
        ("Data-source study; pond dataset selection", 1.8, 1.8, S1),
        ("Requirement analysis and system design", 3.0, 2.0, S1),
        ("Review-1 (title, objectives, design)", 5.0, 0.6, MUTED),
        ("Digital twin: dynamics, scenarios, sensors", 4.6, 2.0, S3),
        ("Data collection and harmonisation (pond IoT)", 5.4, 1.6, S3),
        ("Twin calibration hooks", 6.6, 0.9, S3),
        ("Forecaster + regimes + risk + SHAP", 6.6, 1.6, S2),
        ("Federated learning: FedAvg / local / central", 7.4, 1.4, S2),
        ("Action-conditioned world model", 8.0, 1.2, S2),
        ("Agent: planner, guardrails, actuators", 7.4, 1.6, S4),
        ("Evaluation grid, ablations, statistics", 8.4, 1.1, S4),
        ("Operator console and counterfactual view", 8.4, 1.1, S5),
        ("Draft report and Review-2 (09-09-2026)", 9.2, 0.9, MUTED),
        ("Pond-domain data, DP-FedAvg, hardening", 10.4, 3.6, GRID),
        ("Final review and thesis (Project-II)", 14.4, 3.4, GRID),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    ax.set_facecolor(SURFACE)
    for i, (name, start, dur, colour) in enumerate(tasks):
        y = len(tasks) - i
        ax.barh(y, dur, left=start, height=0.55, color=colour, edgecolor="none",
                alpha=0.92, zorder=3)
        ax.text(start - 0.2, y, name, ha="right", va="center", fontsize=8.6, color=INK)
    ax.set_yticks([])
    ax.set_ylim(0.2, len(tasks) + 0.9)
    ax.set_xlim(-0.15, 18.6)
    months = [(0.0, "Jul"), (4.4, "Aug"), (8.9, "Sep"), (13.3, "Oct"), (17.7, "Nov")]
    ax.set_xticks([m[0] for m in months])
    ax.set_xticklabels([m[1] + "\n2026" for m in months], fontsize=8.5, color=INK2)
    for m, _ in months[1:]:
        ax.axvline(m, color=GRID, lw=1, zorder=1)
    ax.axvline(10.0, color=ST_CRITICAL, lw=1.4, ls="--", zorder=4)
    ax.text(10.2, len(tasks) + 0.6, "Review-2   09-09-2026", fontsize=8.4,
            color=ST_CRITICAL, fontweight="bold", va="center")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Coloured bars: delivered and demonstrated at Review-2.    "
                  "Pale bars: Project-II scope.", fontsize=8, color=MUTED, labelpad=10)
    save(fig, out, "fig1_gantt")


# ---------------------------------------------------------------------------- 2. architecture
def fig_architecture(out: Path):
    fig, ax = canvas(13.6, 9.4)

    lanes = [
        (93, "ONE-TIME, ONLINE", S1),
        (73, "OFFLINE PRE-TRAINING", S2),
        (40, "LIVE LOOP (every tick)", S3),
        (11, "OPERATOR INTERFACE", S5),
    ]
    for y, label, colour in lanes:
        ax.text(1.0, y, label, ha="center", va="center", fontsize=8.4, color=colour,
                fontweight="bold", rotation=90)
    for y in (84, 61, 22):
        ax.axhline(y, xmin=0.035, color=GRID, lw=1)

    # --- layer 1: acquisition
    sq(ax, 26, 96, 22, 6.5, "Kaggle aquaponics ponds", "data", 8.4)
    sq(ax, 26, 88, 22, 6.5, "Nsukka, Nigeria  (IoT, ~20 s)", "data", 8.4)
    box(ax, 54, 92, 20, 8, "collect + harmonise\nsrc/data/", "data", 8.4)
    store(ax, 84, 92, 26, 8, f"data_cache/*.parquet\n{N_ROWS:,} rows @ 15 min\n{N_PONDS} ponds", 8.2)
    arrow(ax, (37, 96), (44, 93.5), kind="data")
    arrow(ax, (37, 88), (44, 90.5), kind="data")
    arrow(ax, (64, 92), (71, 92), kind="data")

    # --- layer 2: federated pre-training
    box(ax, 22, 73, 22, 9, f"PondClient x {N_PONDS}\none pond = one client\nsrc/federated/", "ml", 8.2)
    box(ax, 46, 73, 18, 9, "FedAvg server\nweight averaging\n(weights only)", "ml", 8.2)
    box(ax, 70, 77, 20, 6, "LSTM forecaster\nsrc/perception/", "ml", 8.2)
    box(ax, 70, 68.5, 20, 6, "Action-conditioned\nworld model", "ml", 8.2)
    store(ax, 91, 73, 15, 9, "models/\n*.pt\nrisk_model.pkl\nregimes.pkl", 8.0)
    elbow(ax, [(84, 88), (84, 80.5), (22, 80.5), (22, 77.6)], kind="data",
          text="windows never leave the client", tx=45, ty=80.5)
    arrow(ax, (33, 73), (37, 73), kind="ml", text="w", ty=75.4)
    arrow(ax, (55, 74.5), (60, 77), kind="ml")
    arrow(ax, (55, 71.5), (60, 68.5), kind="ml")
    arrow(ax, (80, 76), (84, 74.5), kind="ml")
    arrow(ax, (80, 69.5), (84, 71.5), kind="ml")

    # --- layer 3: live loop
    box(ax, 22, 46, 22, 8, "PondTwin\nsrc/twin/pond.py", "twin", 8.4, bold=True)
    box(ax, 22, 32, 22, 8, "SensorStream\nnoise · dropout · fault", "twin", 8.4)
    box(ax, 48, 46, 22, 9, "Perception\nforecast · anomaly\nrisk + SHAP", "ml", 8.4)
    box(ax, 48, 31, 22, 9, "Planner (MPC)\n14 candidate actions\nrolled through world model",
        "agent", 8.0)
    box(ax, 72, 38, 19, 10, "GUARDRAILS\ncaps · rates · context\nemergency override",
        "agent", 8.4, bold=True, ec=ST_CRITICAL, lw=1.8)
    box(ax, 91, 46, 16, 8, "Actuators\naerator · feed\nlime · exchange", "agent", 8.0)
    box(ax, 91, 28, 16, 8, "Shadow twin\nagent off", "twin", 8.2)

    elbow(ax, [(91, 68.5), (91, 58.5), (48, 58.5), (48, 50.8)], kind="ml", ls="--",
          text="pre-trained weights, loaded at start-up", tx=68, ty=58.5)
    elbow(ax, [(91, 50), (91, 54.2), (22, 54.2), (22, 50.3)], kind="agent",
          text="applied action", tx=57, ty=54.2)
    elbow(ax, [(11, 44), (6, 44), (6, 19), (91, 19), (91, 23.7)], kind="twin", ls="--",
          text="fork at t = 0: same seed, same noise draw", tx=45, ty=19)

    arrow(ax, (22, 42), (22, 36), kind="twin", text="true state", tx=28.5, ty=39)
    arrow(ax, (33, 34), (37, 42), kind="twin", text="noisy reading", tx=41, ty=37.5)
    arrow(ax, (33, 30.5), (37, 30.5), kind="twin")
    arrow(ax, (48, 41.5), (48, 35.5), kind="ml", text="forecast · risk", tx=58, ty=38.6)
    arrow(ax, (59, 32.5), (63, 35.5), kind="agent", text="proposed", ty=30.5, tx=61.5)
    arrow(ax, (81, 41.5), (83, 44.5), kind="agent", text="allowed", ty=47.6, tx=78)

    # --- layer 4: dashboard
    box(ax, 25, 11, 28, 9, "Desktop operator console\nLive · Counterfactual · Agent · Scorecard",
        "ui", 8.4, bold=True)
    box(ax, 58, 11, 22, 9, "Evaluation harness\n5 controllers x 8 scenarios\nx 10 seeds",
        "ui", 8.2)
    store(ax, 86, 11, 20, 9, "outputs/*.csv\nRESULTS.md", 8.2)
    arrow(ax, (40, 27), (29, 15.8), kind="ui", rad=0.08, text="state · risk · rationale",
          tx=27, ty=24)
    arrow(ax, (83, 24.5), (66, 15.8), kind="ui", rad=-0.10, text="counterfactual gap",
          tx=76, ty=23.2)
    arrow(ax, (69, 11), (76, 11), kind="ui")

    caption(ax, "Every box maps to a module under src/. The guardrail block is the only "
                "component on the actuator path that cannot be bypassed.", y=2.5)
    save(fig, out, "fig2_architecture")


# ---------------------------------------------------------------------------- 3. DFD-0
def fig_dfd0(out: Path):
    fig, ax = canvas(11.0, 5.2, ylim=(10, 100))
    sq(ax, 14, 78, 22, 13, "Public pond\nsensor dataset\n(Kaggle, Nsukka)", "data", 9)
    sq(ax, 14, 32, 22, 11, "Farm operator", "ui", 9)
    sq(ax, 86, 55, 22, 13, "Pond actuators\n(aerator, feeder,\nlime, exchange)", "agent", 9)

    ax.add_patch(Circle((50, 55), 13, facecolor=FILL["twin"], edgecolor=EDGE["twin"],
                        lw=1.6, zorder=2))
    ax.text(50, 58.5, "0", ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(50, 52, "AquaSentinel\nmonitoring and\naction system", ha="center", va="center",
            fontsize=8.8, color=INK, linespacing=1.35)

    arrow(ax, (25, 74), (39, 61), kind="data", text="raw time series", tx=28, ty=67)
    arrow(ax, (25, 36), (39, 49), kind="ui", text="scenario, weights,\ncost preference",
          tx=29, ty=43)
    arrow(ax, (63, 58), (75, 58), kind="agent", text="allowed action", ty=62)
    arrow(ax, (75, 51), (64, 51), kind="twin", text="water-quality state", ty=47)
    arrow(ax, (44, 43), (23, 38), kind="ui", rad=-0.20,
          text="risk band, forecast, SHAP rationale,\ncounterfactual, audit log",
          tx=45, ty=27)

    caption(ax, "Fig. 3. Data flow diagram - Level 0 (context diagram).", y=13)
    save(fig, out, "fig3_dfd0")


# ---------------------------------------------------------------------------- 4. DFD-1
def fig_dfd1(out: Path):
    fig, ax = canvas(12.6, 7.8)

    def proc(x, y, num, label, kind, rx=10.0, ry=7.6):
        ax.add_patch(Ellipse((x, y), rx * 2, ry * 2, facecolor=FILL[kind],
                             edgecolor=EDGE[kind], lw=1.3, zorder=2))
        ax.text(x, y + 3.4, num, ha="center", va="center", fontsize=8.6, color=INK2,
                fontweight="bold", zorder=3)
        ax.text(x, y - 1.8, label, ha="center", va="center", fontsize=8.0, color=INK,
                zorder=3, linespacing=1.3)

    # --- acquisition and pre-training
    sq(ax, 8, 92, 15, 8, "Public\nAPIs", "data", 8.2)
    proc(28, 92, "1.0", "Collect and\nharmonise", "data")
    store(ax, 52, 92, 20, 7, "D1  data_cache/", 8.2)
    proc(76, 92, "2.0", "Federated\npre-training", "ml")
    store(ax, 76, 74, 20, 7, "D2  models/", 8.2)

    arrow(ax, (16, 92), (18, 92), kind="data")
    arrow(ax, (38, 92), (42, 92), kind="data", text="harmonised series", ty=97)
    arrow(ax, (62, 92), (66, 92), kind="data", text="windows", ty=97)
    arrow(ax, (76, 84), (76, 78), kind="ml", text="weights", tx=85, ty=81)
    arrow(ax, (44, 88.5), (21, 74.5), kind="data", ls="--", text="calibration statistics",
          tx=38, ty=82)

    # --- live loop
    proc(18, 66, "3.0", "Simulate pond\n(digital twin)", "twin")
    proc(18, 44, "4.0", "Sample sensors\n(noise, faults)", "twin")
    proc(48, 54, "5.0", "Perceive:\nforecast, risk,\nanomaly, SHAP", "ml", ry=8.6)
    proc(78, 54, "6.0", "Plan action\n(MPC)", "agent")
    proc(78, 28, "7.0", "Enforce\nguardrails", "agent")
    sq(ax, 78, 10, 18, 8, "Pond actuators", "agent", 8.2)

    arrow(ax, (18, 58), (18, 52), kind="twin", text="true state", tx=27, ty=55)
    arrow(ax, (27, 47), (38, 51), kind="twin", text="reading", tx=31, ty=51)
    arrow(ax, (72, 71), (54, 62), kind="ml", text="model", tx=66, ty=65)
    arrow(ax, (58, 54), (68, 54), kind="ml", text="forecast, risk", ty=58.5)
    arrow(ax, (78, 46), (78, 36), kind="agent", text="proposed action", tx=90, ty=41)
    arrow(ax, (78, 20), (78, 15), kind="agent", text="allowed action", tx=90, ty=17.5)
    arrow(ax, (68, 26), (26, 61), kind="agent", rad=0.10, text="applied action", tx=57, ty=39)

    # --- reporting
    proc(44, 20, "8.0", "Render and\nlog", "ui")
    sq(ax, 10, 20, 15, 8, "Operator", "ui", 8.2)
    store(ax, 44, 4, 26, 6, "D4  outputs/ · decision log", 8.2)
    arrow(ax, (42, 46), (44, 29), kind="ui", text="panel data", tx=36, ty=38)
    arrow(ax, (34, 21.5), (18, 21.5), kind="ui", text="dashboard", ty=25.5, tx=26)
    arrow(ax, (18, 18.5), (34, 18.5), kind="ui", text="scenario, weights", ty=14.5, tx=26)
    arrow(ax, (44, 12), (44, 7.5), kind="plain", ls="--")

    caption(ax, "Fig. 4. Data flow diagram - Level 1.  Dashed edges are offline or one-time; "
                "solid edges run every tick.", y=-3)
    save(fig, out, "fig4_dfd1")


# ---------------------------------------------------------------------------- 5. use case
def fig_usecase(out: Path):
    fig, ax = canvas(11.4, 7.6)

    def actor(x, y, name):
        ax.add_patch(Circle((x, y + 6), 2.0, facecolor="none", edgecolor=INK, lw=1.2, zorder=3))
        ax.plot([x, x], [y + 4, y - 1], color=INK, lw=1.2, zorder=3)
        ax.plot([x - 3, x + 3], [y + 2.5, y + 2.5], color=INK, lw=1.2, zorder=3)
        ax.plot([x, x - 2.6], [y - 1, y - 5.5], color=INK, lw=1.2, zorder=3)
        ax.plot([x, x + 2.6], [y - 1, y - 5.5], color=INK, lw=1.2, zorder=3)
        ax.text(x, y - 9, name, ha="center", va="center", fontsize=8.4, color=INK,
                linespacing=1.3)

    def uc(x, y, label, kind, w=27, h=9.2):
        ax.add_patch(Ellipse((x, y), w, h, facecolor=FILL[kind], edgecolor=EDGE[kind],
                             lw=1.2, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.0, color=INK, zorder=3,
                linespacing=1.28)

    ax.add_patch(Rectangle((22, 6), 56, 88, facecolor="none", edgecolor=MUTED, lw=1.1,
                           linestyle="--", zorder=1))
    ax.text(50, 96.5, "AquaSentinel", ha="center", va="center", fontsize=9.4, color=INK2,
            fontweight="bold")

    actor(10, 66, "Farm operator")
    actor(10, 22, "Researcher /\nevaluator")
    actor(90, 70, "Public pond dataset\n(Kaggle, Nsukka)")
    actor(90, 26, "Pond actuator\nhardware")

    uc(38, 86, "Select scenario\nand start run", "ui")
    uc(38, 72, "Monitor live water\nquality and risk", "ui")
    uc(38, 58, "View counterfactual\n(agent vs. no action)", "ui")
    uc(38, 44, "Inspect SHAP\nrationale for a decision", "ml")
    uc(38, 30, "Override or approve\nan action", "agent")
    uc(38, 16, "Switch driving weights\n(fed / local / central)", "ml")

    uc(64, 86, "Collect and cache\nreal site data", "data", w=24)
    uc(64, 72, "Pre-train models\nby FedAvg", "ml", w=24)
    uc(64, 58, "Forecast horizon\nand score risk", "ml", w=24)
    uc(64, 44, "Plan action\nover world model", "agent", w=24)
    uc(64, 30, "Enforce safety\nguardrails", "agent", w=24)
    uc(64, 16, "Run evaluation grid\nand ablations", "ui", w=24)

    assoc = dict(style="-", color=MUTED, lw=0.9)
    for y in (86, 72, 58, 44, 30):
        arrow(ax, (14, 66), (25, y), **assoc)
    arrow(ax, (14, 22), (25, 16), **assoc)
    arrow(ax, (14, 22), (25, 44), **assoc)
    arrow(ax, (14, 17), (52, 10), rad=-0.14, **assoc)
    arrow(ax, (86, 70), (76, 86), **assoc)
    arrow(ax, (86, 26), (76, 30), **assoc)

    inc = dict(ls="--", style="-|>", color=MUTED, fs=6.8, lw=0.9)
    arrow(ax, (64, 67.5), (64, 63), text="<<include>>", tx=73.5, ty=65.2, **inc)
    arrow(ax, (64, 39.5), (64, 35), text="<<include>>", tx=73.5, ty=37.2, **inc)
    arrow(ax, (50, 46), (52, 55), text="<<include>>", tx=44, ty=52, **inc)
    arrow(ax, (50, 32), (52, 41), text="<<include>>", tx=44, ty=38, **inc)

    caption(ax, "Fig. 5. Use case diagram.", y=1)
    save(fig, out, "fig5_usecase")


# ---------------------------------------------------------------------------- 6. class
def fig_class(out: Path):
    fig, ax = canvas(12.8, 7.8)

    def cls(x, y, name, attrs, ops, kind, w=24):
        ah, bh, oh = 4.6, 3.0 * len(attrs) + 1.6, 3.0 * len(ops) + 1.6
        top = y + (ah + bh + oh) / 2
        ax.add_patch(Rectangle((x - w / 2, top - ah), w, ah, facecolor=FILL[kind],
                               edgecolor=EDGE[kind], lw=1.2, zorder=2))
        ax.text(x, top - ah / 2, name, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color=INK, zorder=3)
        ax.add_patch(Rectangle((x - w / 2, top - ah - bh), w, bh, facecolor=SURFACE,
                               edgecolor=EDGE[kind], lw=1.2, zorder=2))
        for i, a in enumerate(attrs):
            ax.text(x - w / 2 + 1.2, top - ah - 2.4 - 3.0 * i, a, ha="left", va="center",
                    fontsize=6.9, color=INK2, zorder=3)
        ax.add_patch(Rectangle((x - w / 2, top - ah - bh - oh), w, oh, facecolor=SURFACE,
                               edgecolor=EDGE[kind], lw=1.2, zorder=2))
        for i, o in enumerate(ops):
            ax.text(x - w / 2 + 1.2, top - ah - bh - 2.4 - 3.0 * i, o, ha="left", va="center",
                    fontsize=6.9, color=INK2, zorder=3)

    cls(13, 80, "PondTwin", ["- state: State", "- cfg: dict", "- rng: Generator"],
        ["+ step(action)", "+ apply(action)", "+ fork_shadow()"], "twin", w=23)
    cls(13, 46, "SensorStream", ["- noise_sigma", "- dropout_p", "- fault_schedule"],
        ["+ sample(state)", "+ inject_fault()"], "twin", w=23)
    cls(13, 14, "Scenario", ["- name, seed", "- schedule"], ["+ perturb(t)"], "twin", w=23)

    cls(39, 80, "Forecaster (LSTM)", ["- hidden, layers", "- delta_scale"],
        ["+ predict(window)", "+ get_weights()"], "ml")
    cls(39, 46, "WorldModel", ["- encoder (federated)", "- action_dim"],
        ["+ rollout(win, act)"], "ml")
    cls(39, 14, "RiskModel", ["- lgbm booster", "- explainer"],
        ["+ score(obs, fc)", "+ attribute()"], "ml")

    cls(65, 80, "PondClient", ["- site_id", "- dataset", "- model"],
        ["+ local_train(E)", "+ get_weights()", "+ set_weights(w)", "+ evaluate()"], "ml")
    cls(65, 46, "Planner", ["- candidates: 14", "- cost_weights"],
        ["+ decide(obs, fc)", "+ score(rollout)"], "agent")
    cls(65, 14, "GuardrailStack", ["- caps, rate_limits", "- sensor_margin"],
        ["+ check(action)", "+ override(state)"], "agent")

    cls(89, 62, "AgentLoop", ["- planner", "- guardrails", "- twin, shadow"],
        ["+ tick()", "+ run(scenario)"], "agent", w=19)
    cls(89, 22, "Dashboard", ["- panels[]"], ["+ render()"], "ui", w=19)

    a = dict(style="-|>", color=MUTED, lw=1.0)
    arrow(ax, (13, 67), (13, 58), **a)                     # PondTwin   -> SensorStream
    arrow(ax, (24.5, 80), (27, 80), **a)                   # PondTwin   -> Forecaster
    arrow(ax, (39, 67), (39, 58), text="encoder warm-start", tx=52.5, ty=63, **a)
    arrow(ax, (51, 80), (53, 80), **a)                     # Forecaster -> PondClient
    arrow(ax, (51, 46), (53, 46), **a)                     # WorldModel -> Planner
    arrow(ax, (51, 16), (53, 36), **a)                     # RiskModel  -> Planner
    elbow(ax, [(1.5, 14), (-4, 14), (-4, 80), (1.4, 80)], kind="plain")  # Scenario -> PondTwin
    arrow(ax, (77, 50), (79.5, 56), **a)                   # Planner    -> AgentLoop
    arrow(ax, (77, 20), (79.5, 51), **a)                   # Guardrails -> AgentLoop
    arrow(ax, (89, 50.5), (89, 30), **a)                   # AgentLoop  -> Dashboard
    elbow(ax, [(13, 93), (13, 99), (89, 99), (89, 74)], kind="plain")   # twin -> AgentLoop

    ax.text(50, -3, "Fig. 6. Class diagram (principal collaborations; attribute and "
                    "operation lists abridged).", ha="center", va="center", fontsize=8,
            color=MUTED, style="italic")
    save(fig, out, "fig6_class")


# ---------------------------------------------------------------------------- 7. sequence
def fig_sequence(out: Path):
    fig, ax = canvas(12.4, 7.4)
    lifelines = [
        (8, "Dashboard", "ui"),
        (22, "PondTwin", "twin"),
        (37, "SensorStream", "twin"),
        (52, "Perception", "ml"),
        (66, "Planner", "agent"),
        (80, "Guardrails", "agent"),
        (93, "Shadow twin", "twin"),
    ]
    for x, name, kind in lifelines:
        box(ax, x, 93, 13.2, 6.5, name, kind, fs=7.8, bold=True)
        ax.plot([x, x], [89.5, 8], color=GRID, lw=1.0, ls=(0, (3, 3)), zorder=1)
        ax.add_patch(Rectangle((x - 1.1, 10), 2.2, 76, facecolor="#efeee9",
                               edgecolor=MUTED, lw=0.7, zorder=2))

    steps = [
        (8, 22, 84, "1: step(dt)"),
        (22, 37, 78, "2: true state"),
        (37, 52, 72, "3: noisy observation"),
        (52, 52, 66, "4: forecast · anomaly · risk · SHAP"),
        (52, 66, 60, "5: observation + forecast + risk"),
        (66, 66, 54, "6: roll 14 candidates through the world model"),
        (66, 80, 48, "7: proposed action"),
        (80, 80, 42, "8: caps · rates · context · predicted-critical"),
        (80, 22, 36, "9: allowed action   (or   blocked + reason)"),
        (22, 93, 30, "10: fork - same seed, same noise, agent disabled"),
        (93, 8, 24, "11: counterfactual trajectory"),
        (22, 8, 18, "12: realised trajectory + decision log"),
        (80, 8, 12, "13: guardrail events, unsafe-applied = 0"),
    ]
    for x0, x1, y, label in steps:
        if x0 == x1:
            ax.add_patch(FancyArrowPatch((x0 + 1.1, y + 2.5), (x0 + 1.1, y - 1.2),
                                         connectionstyle="arc3,rad=-2.6", arrowstyle="-|>",
                                         mutation_scale=9, color=INK2, lw=1.0, zorder=3))
            if x0 > 70:  # keep the long right-hand self-message labels inside the canvas
                ax.text(x0 - 5.5, y + 1.0, label, ha="right", va="center", fontsize=7.4,
                        color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2),
                        zorder=4)
            else:
                ax.text(x0 + 6.0, y + 1.0, label, ha="left", va="center", fontsize=7.4,
                        color=INK2)
        else:
            d = 1.1 if x1 > x0 else -1.1
            arrow(ax, (x0 + d, y), (x1 - d, y), color=INK2, lw=1.0)
            ax.text((x0 + x1) / 2, y + 2.3, label, ha="center", va="center", fontsize=7.4,
                    color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2),
                    zorder=4)

    ax.add_patch(Rectangle((2, 9), 97, 78, facecolor="none", edgecolor=EDGE["agent"],
                           lw=1.0, ls="--", zorder=1))
    ax.text(3, 5.5, "loop  [every sensor tick; the planner re-decides every "
                    "agent.decide_every_minutes]",
            ha="left", va="center", fontsize=7.6, color=EDGE["agent"], style="italic")

    caption(ax, "Fig. 7. Sequence diagram for one control tick.", y=0.5)
    save(fig, out, "fig7_sequence")


# --------------------------------------------------------------- 8-9. pipeline diagrams
# These two are drawn at their final print size (7.4 in wide, the text width of an A4 page
# with 1.6 cm side margins) and their font sizes are the sizes that reach the paper. That is
# the whole reason they are two figures rather than one: a single diagram carrying the same
# content would have to be shrunk to fit, and 4 pt boxes on a printed page are decoration, not
# information.
UPP = 0.1875          # data units per typographic point at the geometry below
CARD_PAD = 2.8        # vertical padding inside a card, in data units

_OVERRUNS: list = []


def _extent(ax, artist):
    """(width, height) of a drawn text artist, in data units.

    Estimating this from a character count and an average advance width is close but not
    close enough - '×', '→' and '·' are much wider than the mean, and a line of specifications
    is full of them. Asking the Agg renderer for the real extent turns both the box sizing and
    the overflow check below from guesses into measurements.
    """
    renderer = ax.figure.canvas.get_renderer()
    bb = artist.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return abs(x1 - x0), abs(y1 - y0)


def _measure(ax, artist) -> float:
    return _extent(ax, artist)[0]


def _check(ax, artist, w: float, where: str, text: str) -> None:
    """Record any text that is wider than the box drawn around it.

    Matplotlib draws text straight through a box edge without complaint, which is the single
    most common way a dense diagram goes wrong. Reporting it to the console makes a layout
    regression visible without re-reading every render.
    """
    need = _measure(ax, artist)
    if need > w - 1.2:
        _OVERRUNS.append(f"{where}: {need:.1f} > {w - 1.2:.1f} units  |  {text[:62]}")


def card(ax, x, y, w, title, body="", kind="plain", ts=8.5, bs=7.0, ec=None, lw=1.25,
         align="center"):
    """Auto-sized specification box centred on (x, y).

    The height is computed from the number of body lines rather than passed in, so a box can
    never be too short for its own text - the failure mode that makes dense diagrams unusable.
    Returns the height, so callers can stack against it.
    """
    lines = body.split("\n") if body else []
    gap = 1.0 if (title and lines) else 0.0
    tx = x if align == "center" else x - w / 2 + 1.6

    # Draw the text first, measure what the renderer actually produced, then size the box to
    # it. Sizing from a line count and an assumed line height is what lets a box end up too
    # short for the text inside it.
    t = b = None
    th = bh = 0.0
    if title:
        t = ax.text(x, y, title, ha="center", va="center", fontsize=ts,
                    fontweight="bold", color=INK, zorder=4)
        th = _extent(ax, t)[1]
        _check(ax, t, w, f"title @({x:.0f},{y:.0f})", title.replace("\n", " / "))
    if lines:
        b = ax.text(tx, y, body, ha=align, va="top", fontsize=bs, color=INK2,
                    linespacing=1.45, zorder=4)
        bh = _extent(ax, b)[1]
        _check(ax, b, w, f"body  @({x:.0f},{y:.0f})", max(lines, key=len))

    h = CARD_PAD + th + gap + bh
    top = y + h / 2 - CARD_PAD / 2
    if t is not None:
        t.set_position((x, top - th / 2))
    if b is not None:
        b.set_position((tx, top - th - gap))
    box(ax, x, y, w, h, "", kind, ec=ec, lw=lw, rounding=0.5)
    return h


def band(ax, y, label, colour):
    ax.text(2.2, y, label, ha="center", va="center", fontsize=8.0, color=colour,
            fontweight="bold", rotation=90)


def rule(ax, y, x0=7.0, x1=99.0):
    ax.plot([x0, x1], [y, y], color=GRID, lw=1.0, zorder=0)


# ---------------------------------------------------------------------------- 8. training
def fig_pipeline_train(out: Path):
    """Acquisition through both federations - every model that is *trained*, and on what."""
    fig, ax = canvas(7.4, 9.2, xlim=(0, 100), ylim=(-3, 121))

    band(ax, 110, "A · ACQUISITION", S1)
    band(ax, 84, "B · WINDOWS", S1)
    band(ax, 51, "C · FEDERATION 1", S2)
    band(ax, 14, "D · FEDERATION 2", S2)
    rule(ax, 98.5)
    rule(ax, 70.5)
    rule(ax, 31.5)

    # --- A. acquisition
    sq(ax, 25, 117, 32, 5.6, "Kaggle aquaponics ponds  —  Nsukka", "data", 8.0)
    sq(ax, 25, 110.5, 32, 5.6, "11 pond files  —  ESP32 probes, ~20 s", "data", 8.0)
    card(ax, 71, 113.7, 50, "collect · harmonise   (src/data/)",
         "one schema · per-channel availability mask\n"
         "QC ranges · stuck probes · outage limit · licence\n"
         "the only step in the project that uses the network", "data", bs=6.9)
    arrow(ax, (41, 117), (46, 115), kind="data")
    arrow(ax, (41, 110.5), (46, 112.4), kind="data")
    store(ax, 53, 102.5, 88, 7.4,
          f"D1   data_cache/*.parquet   —   {N_PONDS} ponds · {N_ROWS:,} rows · 15-minute cadence\n"
          "DO, temperature, pH, TAN and turbidity on every pond · no quality label read",
          6.9)
    arrow(ax, (71, 108.9), (60, 106.2), kind="data")

    # --- B. windows and features
    # W and H are counts of steps, not clock time: make_windows() slices the series by index,
    # so the same 36/24 is 9 h/6 h on the 15-minute real feed and 3 h/2 h in the 5-minute twin.
    card(ax, 25, 90, 32, "Windowing",
         "W = 36 steps in → H = 24 steps out\n"
         "9 h → 6 h on the 15-min real feed\n"
         "stride 6 · anchored on DO and temp", "data", bs=6.9)
    card(ax, 71, 90, 50, "Targets are per-channel Δ ÷ a FIXED scale",
         "DO 2.0 · temp 1.5 · pH 0.3 · TAN 0.3 · NO₂ 0.1 · turb 20\n"
         "fixed, not fitted — fitting would leak client statistics", "plain", bs=6.9)
    card(ax, 20, 76, 26, "Forecast tensor", "36 × 12  (6 values + 6 mask)", "ml", bs=6.9)
    card(ax, 53, 76, 32, "Regime features",
         "4 dimensionless derived × 4 stats", "ml", bs=6.9)
    card(ax, 85, 76, 26, "Risk features", "channels + forecast stats", "ml", bs=6.9)
    arrow(ax, (22, 85.2), (20, 80), kind="data")
    arrow(ax, (30, 85.2), (49, 80), kind="data")
    arrow(ax, (37, 86), (80, 80), kind="data")
    elbow(ax, [(53, 98.8), (53, 96), (25, 96), (25, 94.9)], kind="data")

    # --- C. federation over the real ponds
    card(ax, 33, 64, 52, f"PondClient × {N_PONDS}   —   one pond = one client",
         "owns its own dataset and model instance; exposes only\n"
         "local_train · get_weights · set_weights · evaluate", "ml", bs=6.7)
    card(ax, 79, 64, 38, "M1   LSTM Forecaster",
         "36×12 → LSTM(64, 1 layer)\n→ Linear(64 → 24×6)\n"
         "lr 3e-3 · batch 128 · 12 epochs · clip 1.0", "ml", bs=6.9)
    arrow(ax, (59.2, 64), (60, 64), kind="ml", ls="--")

    card(ax, 20, 51, 24, "independent-local",
         f"24 epochs, alone\n→ {N_TRAIN} per-pond models", "plain", bs=6.9)
    card(ax, 52, 51, 32, "FEDERATED  —  FedAvg",
         "12 rounds × 2 local epochs\nclient_fraction 1.0, sample-weighted",
         "ml", bs=6.9, ec=S2, lw=2.0)
    card(ax, 85, 51, 26, "central",
         "pooled windows, 24 epochs\nupper reference only", "plain", bs=6.9)
    arrow(ax, (22, 59.2), (20, 55.9), kind="ml")
    arrow(ax, (46, 59.2), (52, 55.9), kind="ml")
    arrow(ax, (72, 59.2), (85, 55.9), kind="ml")

    card(ax, 33, 39, 52, "What makes the regimes comparable",
         f"matched work (24 = 12 × 2) · shared init · {N_PONDS - N_TRAIN} ponds held out,\n"
         "never trained on · 2 data-poor clients keep 8 %",
         "plain", bs=6.7)
    card(ax, 82, 39, 32, "DP-FedAvg   (opt-in)",
         "clip 1.0 · σ 0.6 · δ 1e-5\nε from an RDP accountant,\nelse reported as UNACCOUNTED",
         "plain", bs=6.5)

    # --- D. federation over simulated ponds
    card(ax, 53, 26, 88, "Why a second, separate federation is unavoidable",
         "The real pond data logs no aerator, feeding or liming commands, and therefore "
         "zero (action → response) pairs.\nNo federation over that data can teach a model "
         "what an aerator does, so that half is learned in the twin.",
         "plain", bs=6.4, ec=ST_CRITICAL, lw=1.4)
    card(ax, 27, 11, 42, "8 heterogeneous simulated ponds",
         "+ 2 held out · volume ×0.55–1.8\ndepth ×0.75–1.45 · aeration efficiency\n"
         "algal productivity · temperature regime", "twin", bs=6.6)
    card(ax, 58, 11, 16, "FedAvg", "12 rounds over\ntwin rollouts\nunder random\nactions",
         "ml", bs=6.6)
    card(ax, 84, 11, 32, "M2   Action-conditioned\nWorld Model",
         "LSTM(64) encoder + 8 action features\n→ MLP(72 → 64 → 24×6), clipped to\n"
         "physical bounds · only the action\nhead is learned in the twin",
         "ml", ts=8.2, bs=6.5)
    arrow(ax, (48.2, 11), (50, 11), kind="ml")
    arrow(ax, (66.2, 11), (68, 11), kind="ml")
    elbow(ax, [(98, 64), (99.6, 64), (99.6, 16), (98, 16)], kind="ml", ls="--",
          text="encoder warm-started from the federated M1", tx=64, ty=20.0)

    store(ax, 50, -0.5, 92, 7.4,
          "models/    forecaster_{federated, central, local×32}.pt    ·    "
          "world_model_{federated, local×8}.pt\nregimes.pkl  (M3)    ·    risk_model.pkl  (M4)"
          "    —    all loaded by the live loop in Fig. 9", 6.6)

    save(fig, out, "fig8_pipeline_training")


# ---------------------------------------------------------------------------- 9. inference
def fig_pipeline_infer(out: Path):
    """Perception heads, the closed control loop, the safety layer and what comes out."""
    fig, ax = canvas(7.4, 9.2, xlim=(0, 100), ylim=(-3, 121))

    band(ax, 104, "E · PERCEPTION", S2)
    band(ax, 62, "F · LIVE LOOP", S3)
    band(ax, 12, "G · OUTPUT", S5)
    rule(ax, 89.0)
    rule(ax, 25.0)

    # --- E. perception heads
    store(ax, 50, 116, 92, 5.6,
          "models/   loaded at start-up — federated weights by default, switchable to "
          "independent-local or central in the console", 6.6)
    card(ax, 22, 104, 30, "M3   HDBSCAN",
         "unsupervised, 16-d dimensionless\nfeatures, min_cluster_size 40,\nmin_samples 10\n"
         "eom → noise label = anomaly flag", "ml", ts=8.0, bs=6.5)
    card(ax, 53, 104, 30, "M4   LightGBM risk",
         "220 trees · 31 leaves · depth 6\nlr 0.06 · min_child_samples 25\n"
         "→ {safe, warning, critical}", "ml", ts=8.0, bs=6.5)
    card(ax, 84.5, 104, 30, "M5   SHAP TreeExplainer",
         "per-tick attribution of M4's score\nto the input channels — this is\n"
         "the operator's stated rationale", "ml", ts=8.0, bs=6.5)
    card(ax, 50, 93, 92, "",
         "No dataset's quality labels are consumed anywhere. M3 returns 2 modes at DBCV 0.000, "
         "so it is used as a novelty\ndetector and not as a regime taxonomy; M4's target is a "
         "band rule from configs/twin.yaml, so it anticipates a rule.",
         "plain", bs=6.4)

    # --- F. the closed loop
    card(ax, 20, 82, 28, "PondTwin — true state",
         "6 channels + latent algae,\nbiomass and BOD pool\n8 stress scenarios, fixed seeds",
         "twin", bs=6.6)
    card(ax, 52, 82, 30, "Perception, every tick",
         "M1 forecast (2 h) · M3 anomaly flag\nM4 risk band + M5 attribution", "ml", bs=6.5)
    card(ax, 20, 68, 28, "SensorStream",
         "noise · dropout · injectable\nprobe faults (a fault corrupts\nthe reading, not the "
         "water)", "twin", bs=6.6)
    card(ax, 50, 64, 28, "M7   MPC Planner",
         "14 curated candidates (of 108),\n24 rollout steps each through M2\n"
         "cost = 1.0·risk + 3.0·critical\n+ 0.06·intervention + 0.20·stab.\n"
         "+ 0.35·feed_loss  →  argmin", "agent", ts=8.2, bs=6.4)
    card(ax, 50, 48, 28, "M6   InternalModel",
         "the agent's own PondTwin from\nnominal config, complementary\n"
         "filter α = 0.35 — it never reads\nthe latent state or the schedule",
         "agent", ts=8.2, bs=6.4)
    card(ax, 85, 70, 30, "GUARDRAILS",
         "1  dosing caps and min intervals\n     lime ≤ 2/day ≥ 6 h · salt ≤ 1/day\n"
         "2  rate limit — aerator ≤ 1.0/tick\n"
         "3  contextual blocks — no lime\n     above pH 8.6, no feed below\n"
         "     DO 3.5, no large cold exchange\n"
         "4  reject anything predicted\n     to drive a channel critical\n"
         "5  emergency override — DO < 2.5\n     forces max aeration, beats 1–4",
         "agent", ts=9.0, bs=6.4, ec=ST_CRITICAL, lw=2.0, align="left")
    card(ax, 85, 48, 30, "Actuators",
         "aerator {0, 0.5, 1.0}\nexchange {none, small, large}\nfeed {normal, reduced, skip}\n"
         "lime dose · salt dose", "agent", bs=6.5)
    card(ax, 20, 48, 28, "Shadow twin",
         "forked at t = 0 · same seed,\nsame perturbation schedule,\n"
         "same noise draw · agent off", "twin", bs=6.6)

    arrow(ax, (20, 76.3), (20, 73.8), kind="twin", text="true state", tx=29.5, ty=75)
    elbow(ax, [(34.2, 68), (35.6, 68), (35.6, 79), (36.6, 79)], kind="twin",
          text="noisy observation", tx=45, ty=74.6)
    arrow(ax, (50, 77.2), (50, 71.5), kind="ml", text="forecast · risk", tx=62, ty=74.6)
    arrow(ax, (64.2, 64), (69.8, 64), kind="agent", text="proposed", tx=67, ty=67.2)
    arrow(ax, (85, 58.2), (85, 54.8), kind="agent", text="allowed", tx=93.5, ty=56.5)
    elbow(ax, [(69.8, 48), (64.4, 48)], kind="agent", ls="--")
    elbow(ax, [(85, 41.4), (85, 38), (20, 38), (20, 42.2)], kind="agent",
          text="the applied action closes the loop on the pond", tx=52, ty=38)
    elbow(ax, [(6, 82), (4.4, 82), (4.4, 48), (6, 48)], kind="twin", ls="--")
    card(ax, 50, 31, 92, "",
         "The planner re-decides every 15 minutes; the guardrails run on every sensor tick "
         "— slow planner, fast safety layer.\nThe guardrails see only the noisy probe, so each "
         "threshold carries about 3σ of sensor noise as headroom.",
         "plain", bs=6.4)

    # --- G. output
    card(ax, 25, 18, 36, "Desktop operator console",
         "Live · Counterfactual · Agent · Scorecard\n"
         "every action against the shadow pond", "ui", bs=6.6)
    card(ax, 66, 18, 36, "Evaluation harness",
         "5 controllers × 8 scenarios × 10 seeds\n"
         "+ ablations · paired Wilcoxon tests", "ui", bs=6.6)
    store(ax, 50, 7, 64, 6.4,
          "outputs/*.csv   →   RESULTS.md\nevery reported number is generated, none typed by "
          "hand", 6.7)
    arrow(ax, (43.2, 18), (47.8, 18), kind="ui")
    arrow(ax, (50, 13.3), (50, 10.4), kind="ui")

    save(fig, out, "fig9_pipeline_inference")


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/review2/figures")
    args = ap.parse_args()
    out = Path(args.out)
    print(f"rendering Review-2 diagrams -> {out}")
    fig_gantt(out)
    fig_architecture(out)
    fig_dfd0(out)
    fig_dfd1(out)
    fig_usecase(out)
    fig_class(out)
    fig_sequence(out)
    fig_pipeline_train(out)
    fig_pipeline_infer(out)
    if _OVERRUNS:
        print(f"\n  {len(_OVERRUNS)} line(s) wider than their box:")
        for o in _OVERRUNS:
            print("   ", o)
    print("done")


if __name__ == "__main__":
    main()
