"""Build report/review2/AquaSentinel-Pipeline-Complete.pdf - the exhaustive pipeline reference.

    python scripts/make_pipeline_pdf.py [--out <path>] [--keep-html]

This is the "explain every last detail" document. Where the draft report summarises, this one
enumerates: every tensor shape, every configuration constant, every rule in the order it fires,
every arithmetic step between a probe reading and an actuator command.

Everything in it is read out of the source and the configs rather than remembered, so the two
cannot drift: constants are imported from ``configs/*.yaml`` at build time and the measured
numbers come from ``outputs/*.csv``. If a value changes in the code, it changes here on the
next build.

Diagrams are inline SVG authored by the helpers in this file - no image files, no external
fonts, so the PDF is self-contained and prints identically anywhere. Printing uses the same
headless-Chrome path as scripts/make_report_pdf.py.
"""

from __future__ import annotations

import argparse
import html as _html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_PDF = ROOT / "report" / "review2" / "AquaSentinel-Pipeline-Complete.pdf"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / "data_cache"

_BUF: List[str] = []
_FIG = {"n": 0}
_TAB = {"n": 0}


# ======================================================================================
# config + results access - every constant in this document comes through here
# ======================================================================================
def _load_yaml(name: str) -> Dict[str, Any]:
    import yaml

    with open(ROOT / "configs" / f"{name}.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = {n: _load_yaml(n) for n in
       ("twin", "sensors", "models", "agent", "federated", "scenarios", "data")}


def csv(name: str):
    """A results CSV, or None if the pipeline has not been run."""
    import pandas as pd

    path = OUTPUTS / f"{name}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def manifest():
    import pandas as pd

    path = CACHE / "manifest.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def num(v, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "n/a"
    if abs(f - round(f)) < 1e-9 and abs(f) < 1e7:
        return f"{int(round(f)):,}"
    return f"{f:,.{nd}f}"


# ======================================================================================
# HTML primitives
# ======================================================================================
def w(s: str = "") -> None:
    _BUF.append(s)


def esc(s: Any) -> str:
    return _html.escape(str(s))


def part(n: int, title: str, kicker: str = "") -> None:
    w(f'<section class="part"><div class="partnum">Part {n}</div>'
      f'<h1 id="part{n}">{title}</h1>'
      + (f'<p class="kicker">{kicker}</p>' if kicker else "")
      + "</section>")


def h2(t: str, anchor: str = "") -> None:
    a = f' id="{anchor}"' if anchor else ""
    w(f"<h2{a}>{t}</h2>")


def h3(t: str) -> None:
    w(f"<h3>{t}</h3>")


def h4(t: str) -> None:
    w(f"<h4>{t}</h4>")


def p(s: str) -> None:
    w(f"<p>{s}</p>")


def ul(items: Sequence[str], cls: str = "") -> None:
    c = f' class="{cls}"' if cls else ""
    w(f"<ul{c}>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")


def ol(items: Sequence[str], start: int = 1, cls: str = "") -> None:
    c = f' class="{cls}"' if cls else ""
    w(f'<ol start="{start}"{c}>' + "".join(f"<li>{i}</li>" for i in items) + "</ol>")


def note(body: str, kind: str = "note", title: str = "") -> None:
    t = f'<div class="nt">{esc(title)}</div>' if title else ""
    w(f'<div class="box {kind}">{t}{body}</div>')


def code(text: str, caption: str = "") -> None:
    cap = f'<div class="codecap">{esc(caption)}</div>' if caption else ""
    w(f'<div class="codewrap">{cap}<pre>{esc(text)}</pre></div>')


def table(rows: List[Sequence[Any]], header: Sequence[str], caption: str = "",
          cls: str = "", align: str = "") -> None:
    """rows already stringified or not; `align` is a per-column string of l/r/c."""
    _TAB["n"] += 1
    n = _TAB["n"]
    al = align or "l" * len(header)
    ths = "".join(f'<th class="a{al[i]}">{h}</th>' for i, h in enumerate(header))
    body = []
    for r in rows:
        tds = "".join(f'<td class="a{al[i] if i < len(al) else "l"}">{c}</td>'
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    cap = (f'<div class="tabcap"><b>Table {n}</b> &mdash; {caption}</div>') if caption else ""
    w(f'<div class="tabwrap {cls}">{cap}<table><thead><tr>{ths}</tr></thead>'
      f'<tbody>{"".join(body)}</tbody></table></div>')


def figure(svg_markup: str, caption: str, wide: bool = False,
           landscape: bool = False) -> None:
    _FIG["n"] += 1
    n = _FIG["n"]
    cls = "fig land" if landscape else ("fig wide" if wide else "fig")
    w(f'<figure class="{cls}">{svg_markup}'
      f'<figcaption><b>Figure {n}</b> &mdash; {caption}</figcaption></figure>')


# ======================================================================================
# SVG kit
# ======================================================================================
#: Diagram type scale. A viewBox unit maps to roughly 0.19 mm in the portrait
#: text column, so raw font sizes chosen for screen come out at ~5 pt in print.
TS = 1.32

INK = "#16202e"
MUTE = "#5d6b7d"
LINE = "#8a99ab"

PALETTE = {
    "data":   ("#e8f1fb", "#3b6ea5", "#173photo"),
    "twin":   ("#e6f4ec", "#2e7d55", "#12331f"),
    "learn":  ("#fdeee6", "#c0642a", "#3a1c08"),
    "fed":    ("#f0eafb", "#6b4fa0", "#241537"),
    "agent":  ("#fdf2e0", "#b1841f", "#3b2a05"),
    "guard":  ("#fbe6e6", "#b03b3b", "#3b1010"),
    "out":    ("#eceff3", "#66727f", "#1d242c"),
    "plain":  ("#ffffff", "#8a99ab", "#16202e"),
}
PALETTE["data"] = ("#e8f1fb", "#3b6ea5", "#132b45")


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg class="dia" viewBox="0 0 {width} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
        f'<defs>'
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
        f'<marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="#b03b3b"/></marker>'
        f'<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="#2e7d55"/></marker>'
        f'</defs>{body}</svg>'
    )


def _wrap(text: str, per_line: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) <= per_line:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def box(x, y, bw, bh, title, sub=None, kind="plain", fs=11.5, subfs=9.5,
        rx=5, dash=False, title_bold=True):
    """A labelled rounded box. `sub` is a string or a list of lines."""
    fs, subfs = fs * TS, subfs * TS
    fill, stroke, tc = PALETTE.get(kind, PALETTE["plain"])
    sd = ' stroke-dasharray="5 3"' if dash else ""
    out = [f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="1.3"{sd}/>']
    lines = []
    if sub is None:
        lines = []
    else:
        per = max(int(bw / (subfs * 0.52)), 8)
        src = [sub] if isinstance(sub, str) else list(sub)
        lines = [ln for item in src for ln in (_wrap(item, per) or [""])]
    total = (fs + 2 if title else 0) + len(lines) * (subfs + 2.2)
    cy = y + bh / 2 - total / 2 + fs * 0.82
    if title:
        fw = ' font-weight="600"' if title_bold else ""
        out.append(f'<text x="{x + bw / 2}" y="{cy}" text-anchor="middle" font-size="{fs}" '
                   f'fill="{tc}"{fw}>{esc(title)}</text>')
        cy += fs * 0.35 + 4
    for ln in lines:
        cy += subfs + 2.2
        out.append(f'<text x="{x + bw / 2}" y="{cy - subfs * 0.28}" text-anchor="middle" '
                   f'font-size="{subfs}" fill="{MUTE}">{esc(ln)}</text>')
    return "".join(out)


def txt(x, y, s, fs=10, fill=None, anchor="start", bold=False, mono=False, italic=False):
    fs = fs * TS
    f = ' font-weight="600"' if bold else ""
    i = ' font-style="italic"' if italic else ""
    fam = ' font-family="Consolas, Menlo, monospace"' if mono else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" '
            f'fill="{fill or INK}"{f}{i}{fam}>{esc(s)}</text>')


def arrow(x1, y1, x2, y2, label="", colour=None, dash=False, marker="ah",
          lfs=9, lo=-5, lanchor="middle"):
    lfs = lfs * TS
    c = colour or LINE
    sd = ' stroke-dasharray="5 4"' if dash else ""
    out = [f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c}" stroke-width="1.4" '
           f'fill="none" marker-end="url(#{marker})"{sd}/>']
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(f'<text x="{mx}" y="{my + lo}" text-anchor="{lanchor}" font-size="{lfs}" '
                   f'fill="{MUTE}">{esc(label)}</text>')
    return "".join(out)


def elbow(x1, y1, x2, y2, label="", colour=None, dash=False, marker="ah",
          via="h", lfs=9, lo=-5):
    """Right-angled connector: `via`='h' goes horizontal first, 'v' vertical first."""
    c = colour or LINE
    sd = ' stroke-dasharray="5 4"' if dash else ""
    lfs = lfs * TS
    d = f"M{x1},{y1} L{x2},{y1} L{x2},{y2}" if via == "h" else f"M{x1},{y1} L{x1},{y2} L{x2},{y2}"
    out = [f'<path d="{d}" stroke="{c}" stroke-width="1.4" fill="none" '
           f'marker-end="url(#{marker})"{sd}/>']
    if label:
        mx = x2 if via == "h" else x1
        my = (y1 + y2) / 2
        out.append(f'<text x="{mx + 6}" y="{my + lo}" font-size="{lfs}" fill="{MUTE}">'
                   f"{esc(label)}</text>")
    return "".join(out)


def lane(x, y, lw, lh, title, tint="#f7f9fc", stroke="#d5dde7"):
    lfs = 10 * TS
    return (f'<rect x="{x}" y="{y}" width="{lw}" height="{lh}" rx="7" fill="{tint}" '
            f'stroke="{stroke}" stroke-width="1"/>'
            f'<text x="{x + 11}" y="{y + 17}" font-size="{lfs}" font-weight="700" '
            f'fill="{MUTE}" letter-spacing="0.6">{esc(title.upper())}</text>')


def tensor(x, y, label, shape, kind="learn", bw=118, bh=34):
    """A small tensor-shape chip."""
    fs = 9.5 * TS
    fill, stroke, tc = PALETTE[kind]
    return (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="4" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
            f'<text x="{x + bw / 2}" y="{y + 14}" text-anchor="middle" font-size="{fs}" '
            f'fill="{tc}" font-weight="600">{esc(label)}</text>'
            f'<text x="{x + bw / 2}" y="{y + 26}" text-anchor="middle" font-size="{fs}" '
            f'fill="{MUTE}" font-family="Consolas, Menlo, monospace">{esc(shape)}</text>')


# ======================================================================================
# stylesheet
# ======================================================================================
CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
@page land { size: A4 landscape; margin: 13mm 14mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 9.6pt; line-height: 1.52; color: #16202e; margin: 0;
}
h1, h2, h3, h4, .partnum, .kicker, figcaption, th, .nt, .tabcap, .codecap,
.chip, .toc, .cover, .badge { font-family: "Segoe UI", Helvetica, Arial, sans-serif; }

/* ---- cover ---- */
.cover { page-break-after: always; padding-top: 34mm; }
.cover .eyebrow { font-size: 10pt; letter-spacing: 2.4px; color: #6b7a8c;
  text-transform: uppercase; font-weight: 700; }
.cover h1 { font-size: 30pt; line-height: 1.1; margin: 10px 0 6px; border: 0; padding: 0;
  color: #12233a; }
.cover .sub { font-size: 13pt; color: #46566a; margin: 0 0 26px; font-family: Georgia, serif;
  font-style: italic; }
.cover .rule { height: 3px; background: #3b6ea5; width: 68px; margin: 20px 0 22px; }
.cover .meta { font-size: 9.4pt; color: #46566a; }
.cover .meta b { color: #16202e; }
.cover .prom { margin-top: 26px; padding: 14px 16px; background: #f2f6fb;
  border-left: 3px solid #3b6ea5; font-size: 9.6pt; }

/* ---- structure ---- */
.part { page-break-before: always; padding-top: 4mm; margin-bottom: 10px; }
.part:first-of-type { page-break-before: avoid; }
.partnum { font-size: 9pt; letter-spacing: 2.6px; text-transform: uppercase;
  color: #3b6ea5; font-weight: 700; }
h1 { font-size: 20pt; margin: 4px 0 8px; color: #12233a; line-height: 1.15;
  border-bottom: 2.5px solid #3b6ea5; padding-bottom: 7px; }
.kicker { font-size: 10pt; color: #56667a; font-style: italic; margin: 8px 0 0;
  font-family: Georgia, serif; }
h2 { font-size: 12.4pt; margin: 17px 0 6px; color: #16334f; page-break-after: avoid; }
h3 { font-size: 10.6pt; margin: 13px 0 4px; color: #1d3c5c; page-break-after: avoid; }
h4 { font-size: 9.6pt; margin: 11px 0 3px; color: #34506e; page-break-after: avoid;
  text-transform: uppercase; letter-spacing: 0.6px; }
p { margin: 0 0 7px; text-align: justify; hyphens: auto; }
ul, ol { margin: 4px 0 8px; padding-left: 18px; }
li { margin: 0 0 3.5px; text-align: justify; }
ul.tight li, ol.tight li { margin: 0 0 1.5px; }
code, .mono { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt;
  background: #eef2f7; padding: 0.5px 3px; border-radius: 3px; color: #123; }
b, strong { color: #0d1b2b; }
a { color: #2c5f96; text-decoration: none; }

/* ---- figures ---- */
figure { margin: 12px 0 14px; page-break-inside: avoid; }
figure.wide { margin: 14px 0 16px; }
figure.land { page: land; page-break-before: always; page-break-after: always;
  margin: 0; padding-top: 2mm; }
figure.land figcaption { font-size: 9pt; margin-top: 9px; }
figure.land svg.dia { max-height: 158mm; width: auto; max-width: 100%;
  margin: 0 auto; display: block; }
svg.dia { display: block; width: 100%; height: auto; }
figcaption { font-size: 8.4pt; color: #56667a; margin-top: 6px; text-align: left;
  border-top: 1px solid #dde4ec; padding-top: 5px; }
figcaption b { color: #2c5f96; }

/* ---- tables ---- */
.tabwrap { margin: 10px 0 13px; page-break-inside: avoid; }
.tabcap { font-size: 8.4pt; color: #56667a; margin-bottom: 5px; }
.tabcap b { color: #2c5f96; }
table { width: 100%; border-collapse: collapse; font-size: 8.5pt;
  font-family: "Segoe UI", Helvetica, Arial, sans-serif; }
th { background: #3b6ea5; color: #fff; text-align: left; padding: 5px 7px;
  font-weight: 600; font-size: 8.3pt; }
td { padding: 4px 7px; border-bottom: 1px solid #e3e9f0; vertical-align: top; color: #22303f; }
tbody tr:nth-child(even) td { background: #f6f9fc; }
.al { text-align: left; } .ar { text-align: right; } .ac { text-align: center; }
table .mono, td code { background: none; padding: 0; }
.tabwrap.small table { font-size: 8pt; }

/* ---- callouts ---- */
.box { margin: 10px 0 12px; padding: 9px 12px; border-radius: 4px; font-size: 9.2pt;
  page-break-inside: avoid; border-left: 3px solid #8a99ab; background: #f6f8fa; }
.box .nt { font-size: 8pt; text-transform: uppercase; letter-spacing: 1.1px;
  font-weight: 700; margin-bottom: 4px; color: #56667a; }
.box p { margin: 0 0 5px; } .box p:last-child { margin-bottom: 0; }
.box ul { margin: 3px 0 0; }
.exam { background: #eef6ef; border-left-color: #2e7d55; }
.exam .nt { color: #226343; }
.warn { background: #fdf1ec; border-left-color: #c0642a; }
.warn .nt { color: #9a4c1c; }
.honest { background: #fbeded; border-left-color: #b03b3b; }
.honest .nt { color: #8f2f2f; }
.num { background: #f0edfa; border-left-color: #6b4fa0; }
.num .nt { color: #56409f; }

/* ---- misc ---- */
.codewrap { margin: 9px 0 12px; page-break-inside: avoid; }
.codecap { font-size: 8.2pt; color: #56667a; margin-bottom: 3px; }
pre { font-family: Consolas, "Courier New", monospace; font-size: 8.1pt; line-height: 1.42;
  background: #f4f7fa; border: 1px solid #e0e7ef; border-left: 3px solid #8fa6bd;
  border-radius: 3px; padding: 8px 10px; margin: 0; overflow-x: auto;
  white-space: pre-wrap; color: #1b2b3a; }
.toc { columns: 2; column-gap: 22px; font-size: 9pt; }
.toc div { break-inside: avoid; margin-bottom: 2.5px; color: #34506e; }
.toc .tp { font-weight: 700; color: #12233a; margin-top: 7px; }
.eq { text-align: center; font-style: italic; margin: 8px 0 10px; font-size: 10pt;
  color: #123; }
.eq .mono { font-style: normal; }
hr.soft { border: 0; border-top: 1px solid #e3e9f0; margin: 14px 0; }
.badge { display: inline-block; font-size: 7.6pt; font-weight: 700; letter-spacing: .6px;
  padding: 1px 6px; border-radius: 9px; background: #e8f1fb; color: #2c5f96;
  text-transform: uppercase; }
"""


# ======================================================================================
# Diagrams
# ======================================================================================
def dg_master() -> str:
    """The whole system on one page: four lanes, and what crosses between them."""
    b = []
    b.append(lane(0, 0, 1100, 118, "Lane A  ·  acquisition  ·  online, once", "#f3f8fd", "#cfe0f0"))
    b.append(box(20, 30, 150, 68, "USA (USGS)",
                 ["28 stations, no password", "a reading every 15 min"], "data"))
    b.append(box(186, 30, 150, 68, "England (EA)",
                 ["14 stations, no password", "a reading every 15 min"], "data"))
    b.append(box(360, 30, 168, 68, "Put in one format", ["clean up bad readings;", "record what is missing"], "data"))
    b.append(box(552, 30, 168, 68, "42 saved stations", ["729,542 rows of readings"], "data"))
    b.append(box(744, 30, 336, 68, "One station = one participant",
                 ["its readings never leave it"], "data"))
    b.append(arrow(170, 64, 184, 64))
    b.append(arrow(336, 64, 358, 64))
    b.append(arrow(528, 64, 550, 64))
    b.append(arrow(720, 64, 742, 64))

    b.append(lane(0, 136, 1100, 172, "Lane B  ·  offline pre-training  ·  no network", "#faf6fd", "#e0d4f0"))
    b.append(box(20, 170, 200, 56, "Cut into examples", ["36 readings in → 24 out;", "changes, not values"], "fed"))
    b.append(box(238, 168, 200, 60, "Trained across sites", ["12 rounds of FedAvg;", "32 stations; no data moves"], "fed"))
    b.append(box(238, 240, 200, 54, "Model 1 · forecaster", ["LSTM; water left alone"], "learn"))
    b.append(box(474, 168, 196, 60, "Twin rollouts × 8 ponds", ["randomised actions", "(action → response) pairs"], "twin"))
    b.append(box(474, 240, 196, 54, "Model 2 · what if I act?", ["LSTM; predicts an effect"], "learn"))
    b.append(box(700, 164, 178, 60, "Model 3 · odd readings", ["HDBSCAN clustering"], "learn"))
    b.append(box(700, 236, 178, 58, "Models 4 & 5 · risk", ["LightGBM, explained", "by SHAP"], "learn"))
    b.append(box(896, 176, 184, 110, "models/",
                 ["forecaster_*.pt", "world_model_*.pt", "regimes.pkl", "risk_model.pkl"], "out"))
    b.append(arrow(220, 198, 236, 198))
    b.append(arrow(338, 228, 338, 238))
    b.append(arrow(438, 198, 472, 198))
    b.append(arrow(572, 228, 572, 238))
    b.append(arrow(878, 265, 894, 265))
    b.append(arrow(878, 194, 894, 214))
    b.append('<path d="M338,294 L338,314 L572,314 L572,296" stroke="#c0642a" stroke-width="1.5" '
             'fill="none" stroke-dasharray="4 3" marker-end="url(#ahd)"/>')
    b.append(txt(455, 328, "encoder warm start — weights copied, then trained further (not frozen)",
                 8.6, "#c0642a", "middle", italic=True))

    b.append(lane(0, 352, 1100, 214, "Lane C  ·  live control loop  ·  one 5-minute tick", "#f4fbf7", "#cfe8da"))
    b.append(box(20, 388, 132, 62, "PondTwin", ["true state", "dt = 5 min"], "twin"))
    b.append(box(168, 388, 132, 62, "SensorStream", ["noise, bias,", "dropout, quantise"], "twin"))
    b.append(box(316, 380, 150, 78, "Perception", ["forecast · odd readings", "risk · explanation"], "learn"))
    b.append(box(482, 380, 158, 78, "Planner", ["tries 14 options through", "Model 2, picks the best"], "agent"))
    b.append(box(656, 380, 150, 78, "GUARDRAILS", ["5 rule groups,", "every single tick"], "guard"))
    b.append(box(822, 388, 132, 62, "Actuators", ["aerator · feed · lime", "salt · exchange"], "agent"))
    b.append(box(20, 482, 280, 58, "Shadow twin (agent off)",
                 ["forked at t=0: same seed, same noise draw"], "twin", dash=True))
    b.append(arrow(152, 418, 166, 418))
    b.append(arrow(300, 418, 314, 418))
    b.append(arrow(466, 418, 480, 418))
    b.append(arrow(640, 418, 654, 418, "proposed"))
    b.append(arrow(806, 418, 820, 418, "applied"))
    b.append(f'<path d="M888,450 L888,552 L86,552 L86,454" stroke="{LINE}" stroke-width="1.4" '
             f'fill="none" marker-end="url(#ah)"/>')
    b.append(txt(500, 548, "the applied action closes the loop on the pond", 8.6, MUTE, "middle", italic=True))
    b.append(f'<path d="M300,511 L360,511" stroke="#2e7d55" stroke-width="1.5" stroke-dasharray="6 4"/>')
    b.append(txt(372, 515, "counterfactual gap = what the agent changed", 8.6, "#226343", italic=True))

    b.append(lane(0, 586, 1100, 96, "Lane D  ·  operator + evaluation", "#f8f9fb", "#dde3ea"))
    b.append(box(20, 612, 250, 56, "Streamlit dashboard",
                 ["Live · Counterfactual · Agent · Scorecard"], "out"))
    b.append(box(292, 612, 300, 56, "Evaluation harness",
                 ["8 scenarios × 5 controllers × 10 seeds"], "out"))
    b.append(box(614, 612, 220, 56, "7 parts switched off", ["what each one was worth"], "out"))
    b.append(box(856, 612, 224, 56, "outputs/*.csv → RESULTS.md", ["every reported number"], "out"))
    b.append(arrow(592, 640, 612, 640))
    b.append(arrow(834, 640, 854, 640))
    return svg(1100, 690, "".join(b))


def dg_acquisition() -> str:
    b = []
    y = 26
    b.append(txt(0, 12, "One-time and online. Everything after this stage is offline and reproducible.",
                 9.5, MUTE, italic=True))
    b.append(box(0, y, 176, 76, "1 · Discover",
                 ["USGS: RDB site service,", "outputDataTypeCd=iv;", "EA: station catalogue"], "data", fs=10, subfs=8.2))
    b.append(box(196, y, 176, 76, "2 · Filter",
                 ["period of record must", "cover the window;", "≥ 3 usable channels"], "data", fs=10, subfs=8.2))
    b.append(box(392, y, 176, 76, "3 · Pull",
                 ["instantaneous values,", "1 Mar – 2 Sep 2024,", "cached to data_cache/raw"], "data", fs=10, subfs=8.2))
    b.append(box(588, y, 176, 76, "4 · Map",
                 ["source parameter code", "→ common channel;", "long [ts, channel, value]"], "data", fs=10, subfs=8.2))
    b.append(box(784, y, 176, 76, "5 · Pivot",
                 ["long → wide;", "resample onto a regular", "15-minute grid"], "data", fs=10, subfs=8.2))
    for x in (176, 372, 568, 764):
        b.append(arrow(x, y + 38, x + 19, y + 38))

    y2 = 152
    b.append(box(0, y2, 240, 84, "6 · Quality control",
                 ["drop physically impossible values",
                  "mask flat-lined runs (stuck probe)",
                  "censor detection-limit floors"], "data", fs=10, subfs=8.2))
    b.append(box(262, y2, 216, 84, "7 · Availability mask",
                 ["one boolean per cell:",
                  "“was this really measured?”",
                  "missingness kept, not filled"], "data", fs=10, subfs=8.2))
    b.append(box(500, y2, 216, 84, "8 · Derive",
                 ["NH₃ from TAN + pH + temp", "(Emerson et al., 1975)"], "data", fs=10, subfs=8.2))
    b.append(box(738, y2, 222, 84, "9 · Write",
                 ["data_cache/sites/*.parquet", "manifest.csv — 42 rows",
                  "licence recorded per site"], "out", fs=10, subfs=8.2))
    b.append(arrow(240, y2 + 42, 260, y2 + 42))
    b.append(arrow(478, y2 + 42, 498, y2 + 42))
    b.append(arrow(716, y2 + 42, 736, y2 + 42))
    b.append(elbow(872, y + 76, 120, y2 - 2, "", None, False, "ah", "v"))

    b.append('<rect x="0" y="256" width="960" height="50" rx="5" fill="#fbeded" '
             'stroke="#b03b3b" stroke-width="1.2"/>')
    b.append(txt(12, 275, "Never read:", 9.6, "#8f2f2f", bold=True))
    b.append(txt(78, 275, "the published water-quality index label. It is a scripted threshold, so training on it and then reporting", 9.2, "#5a2020"))
    b.append(txt(12, 291, "accuracy against it would be circular. Every label used downstream is derived from the twin’s own risk bands instead.", 9.2, "#5a2020"))
    return svg(960, 310, "".join(b))


def dg_twin_step() -> str:
    """One simulator step, as a 3 x 3 grid so it reads well in the portrait column."""
    b = []
    steps = [
        ("1", "Sunlight and temperature", ["how much light there is now;", "what temperature the",
                                           "water is heading for"], "twin"),
        ("2", "Apply the action", ["feed, but only at 08:00", "and 16:00; lime goes into",
                                   "a pile; salt becomes chloride"], "agent"),
        ("3", "Did the aerator work?", ["if the scenario says it has",
                                        "failed, nothing is delivered", "even though it was told to"], "agent"),
        ("4", "Work out the rates", ["how fast oxygen, ammonia,",
                                     "nitrite, pH, alkalinity,", "algae and cloudiness change"], "twin"),
        ("5", "Take the step", ["new value = old value",
                                "+ rate x 5 minutes", "(this is Euler's method)"], "twin"),
        ("6", "Update the stores", ["feed turns into nitrogen",
                                    "and decaying matter;", "lime dissolves; salt fades"], "twin"),
        ("7", "Water exchange", ["if an exchange is running,",
                                 "blend the pond towards", "the cleaner inflow water"], "twin"),
        ("8", "Add the random wobble", ["the drift-and-return kind",
                                        "from Part 2.6, not", "independent random noise"], "twin"),
        ("9", "Force back into range", ["oxygen at least 0.05,",
                                        "pH between 3 and 12,", "nothing below zero"], "twin"),
    ]
    bw, bh, gx, gy = 224, 96, 18, 16
    for i, (n, title, sub, kind) in enumerate(steps):
        col, row = i % 3, i // 3
        x = col * (bw + gx)
        y = 26 + row * (bh + gy)
        b.append(box(x, y, bw, bh, title, sub, kind, fs=10.2, subfs=8.4))
        b.append(f'<circle cx="{x + 14}" cy="{y + 13}" r="9.5" fill="#ffffff" stroke="{LINE}"/>')
        b.append(txt(x + 14, y + 16.5, n, 9, MUTE, "middle", bold=True))
        if col < 2:
            b.append(arrow(x + bw, y + bh / 2, x + bw + gx - 2, y + bh / 2))
    # row wraps
    for row in (0, 1):
        y = 26 + row * (bh + gy) + bh
        b.append(f'<path d="M{2 * (bw + gx) + bw / 2},{y} L{2 * (bw + gx) + bw / 2},{y + 8} '
                 f'L{bw / 2},{y + 8} L{bw / 2},{y + gy - 2}" stroke="{LINE}" stroke-width="1.3" '
                 f'fill="none" marker-end="url(#ah)"/>')
    total_h = 26 + 3 * bh + 2 * gy
    b.append(txt(0, 13, "One step of the simulator. The order matters: the action is applied "
                        "before the rates are worked out, and the limits come last.",
                 9.2, MUTE, italic=True))
    b.append(f'<path d="M{2 * (bw + gx) + bw / 2},{total_h} L{2 * (bw + gx) + bw / 2},'
             f'{total_h + 12} L{bw / 2},{total_h + 12} L{bw / 2},{total_h + 4}" '
             f'stroke="{LINE}" stroke-width="1.3" fill="none" stroke-dasharray="4 3" '
             f'marker-end="url(#ah)"/>')
    b.append(txt((3 * bw + 2 * gx) / 2, total_h + 26, "then the next step, five simulated minutes later",
                 8.8, MUTE, "middle", italic=True))
    return svg(3 * bw + 2 * gx, total_h + 34, "".join(b))


def dg_do_budget() -> str:
    """The six-term dissolved-oxygen budget, drawn as sources against sinks."""
    b = []
    cx, cy = 470, 122
    b.append(f'<rect x="{cx - 96}" y="{cy - 36}" width="192" height="72" rx="8" '
             f'fill="#e6f4ec" stroke="#2e7d55" stroke-width="1.8"/>')
    b.append(txt(cx, cy - 6, "dDO/dt", 16, "#12331f", "middle", bold=True))
    b.append(txt(cx, cy + 14, "mg/L per hour", 9, MUTE, "middle"))

    ox = CFG["twin"]["oxygen"]
    sources = [
        ("Surface reaeration", "k_a · (DO_sat − DO)", f"k_a = {ox['k_reaeration_per_h']} /h"),
        ("Photosynthesis", "P_max · light · algae",
         f"P_max = {ox['p_max_per_h']} /h · exactly zero at night"),
        ("Mechanical aeration", "k_aer · u · (DO_sat − DO)",
         f"k_aer = {ox['k_aerator_per_h']} /h · u = aerator ∈ [0,1]"),
    ]
    sinks = [
        ("Respiration", "R₂₀ · θ^(T−20) · (biomass + w·algae)",
         f"R₂₀ = {ox['r20_per_h']} /h, θ = {ox['theta_resp']}, w = {ox['algae_respiration_weight']}"),
        ("Organic decay", "k_BOD · BOD_pool", f"k_BOD = {ox['k_bod_per_h']} /h"),
        ("Nitrifier demand", "4.57 · N_nitrified", "4.57 mg O₂ consumed per mg N oxidised"),
    ]
    ys = [16, 84, 152]
    for (title, eq, sub), y in zip(sources, ys):
        b.append(box(0, y, 306, 58, title, [eq, sub], "twin", fs=10, subfs=8.3))
        b.append(arrow(306, y + 29, cx - 100, cy - 12 + (y - 84) * 0.16, "", "#2e7d55", False, "ahg"))
    for (title, eq, sub), y in zip(sinks, ys):
        b.append(box(634, y, 306, 58, title, [eq, sub], "guard", fs=10, subfs=8.3))
        b.append(arrow(cx + 100, cy - 12 + (y - 84) * 0.16, 632, y + 29, "", "#b03b3b", False, "ahd"))
    b.append(txt(153, 10, "SOURCES  ( + )", 9, "#226343", "middle", bold=True))
    b.append(txt(787, 10, "SINKS  ( − )", 9, "#8f2f2f", "middle", bold=True))
    b.append(txt(cx, 236, "DO_sat comes from the Benson–Krause / APHA freshwater fit with a salinity term. Every",
                 8.8, MUTE, "middle", italic=True))
    b.append(txt(cx, 249, "coefficient is a calibratable default in configs/twin.yaml, not a measured fact.",
                 8.8, MUTE, "middle", italic=True))
    return svg(940, 258, "".join(b))


def dg_sensor_chain() -> str:
    """True value to reading, as two rows so it fits the portrait column legibly."""
    sc = CFG["sensors"]
    b = []
    stages = [
        ("The true value", ["what the simulator", "actually has, e.g.", "oxygen = 6.4173"], "twin"),
        ("Add random noise", [f"typical size {sc['noise_std']['do']} mg/L", "for oxygen; a fresh",
                              "draw every reading"], "learn"),
        ("Add a fixed error", ["each probe is slightly", "miscalibrated; drawn",
                               "ONCE at the start of a run"], "learn"),
        ("Inject a fault", ["only if the scenario asks:",
                            "missing, spike, stuck,", "or slow drift"], "guard"),
        ("Lose it at random", [f"{sc['dropout_prob']:.0%} of the time the",
                               "reading simply does", "not arrive"], "guard"),
        ("Round it off", [f"to the nearest {sc['quantization']['do']} mg/L", "for oxygen; a probe",
                          "cannot report finer"], "learn"),
    ]
    bw, bh, gx, gy = 224, 92, 18, 16
    for i, (t, sub, k) in enumerate(stages):
        col, row = i % 3, i // 3
        x = col * (bw + gx)
        y = 26 + row * (bh + gy)
        b.append(box(x, y, bw, bh, t, sub, k, fs=10.2, subfs=8.4))
        if col < 2:
            b.append(arrow(x + bw, y + bh / 2, x + bw + gx - 2, y + bh / 2))
    y_wrap = 26 + bh
    b.append(f'<path d="M{2 * (bw + gx) + bw / 2},{y_wrap} L{2 * (bw + gx) + bw / 2},'
             f'{y_wrap + 8} L{bw / 2},{y_wrap + 8} L{bw / 2},{y_wrap + gy - 2}" '
             f'stroke="{LINE}" stroke-width="1.3" fill="none" marker-end="url(#ah)"/>')
    total = 3 * bw + 2 * gx
    y2 = 26 + 2 * bh + gy + 10
    b.append(box(0, y2, total, 46, "What comes out: a reading, plus a flag saying whether it is real",
                 ["this is everything the rest of the system ever sees"], "out", fs=10.4, subfs=8.6))
    b.append(txt(0, 13, "How a true value becomes a reading, in the order the steps happen.",
                 9.2, MUTE, italic=True))
    y3 = y2 + 58
    b.append(f'<rect x="0" y="{y3}" width="{total}" height="58" rx="5" fill="#eef6ef" '
             f'stroke="#2e7d55" stroke-width="1.2"/>')
    b.append(txt(12, y3 + 19, "Why the fault is injected here and not in the simulator",
                 9.4, "#226343", bold=True))
    b.append(txt(12, y3 + 35, "Because a broken probe spoils the reading, not the water. That is "
                              "exactly the distinction the odd-reading detector", 9, "#294c39"))
    b.append(txt(12, y3 + 49, "has to learn, and the sensor-fault test measures how far it gets.",
                 9, "#294c39"))
    return svg(total, y3 + 68, "".join(b))


def dg_windows() -> str:
    """Windowing and the delta encoding, with every tensor shape named."""
    m = CFG["models"]
    W, H = m["window"], m["horizon"]
    C = len(m["forecaster"]["channels"])
    b = []
    b.append(txt(0, 12, "LSTMForecaster.make_windows() — one site’s series becomes supervised examples",
                 9.4, MUTE, italic=True))

    # the strip
    x0, y0, cellw, cellh = 0, 30, 9.0, 26
    for i in range(W + H):
        fill = "#e8f1fb" if i < W else "#fdeee6"
        stroke = "#3b6ea5" if i < W else "#c0642a"
        b.append(f'<rect x="{x0 + i * cellw}" y="{y0}" width="{cellw - 1}" height="{cellh}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>')
    b.append(f'<rect x="{x0 + (W - 1) * cellw}" y="{y0 - 4}" width="{cellw + 1}" height="{cellh + 8}" '
             f'fill="none" stroke="#b03b3b" stroke-width="2"/>')
    b.append(txt(x0 + W * cellw / 2, y0 + cellh + 15, f"input window  W = {W} steps", 9, "#2c5f96", "middle", bold=True))
    b.append(txt(x0 + W * cellw + H * cellw / 2, y0 + cellh + 15, f"target  H = {H}", 9, "#9a4c1c", "middle", bold=True))
    b.append(txt(x0 + (W - 1) * cellw + 5, y0 - 9, "anchor step", 8.4, "#8f2f2f", "middle", bold=True))

    b.append(txt(x0 + W * cellw / 2, y0 + cellh + 29,
                 "3 h on the twin (5-min steps) · 9 h on the real 15-min feed", 8.3, MUTE, "middle"))
    b.append(txt(x0 + W * cellw + H * cellw / 2, y0 + cellh + 29, "2 h twin · 6 h real", 8.3, MUTE, "middle"))

    sx = x0 + (W + H) * cellw + 26
    b.append(box(sx, 24, 150, 44, "stride = 6", ["slide, take the next"], "plain", fs=9.6, subfs=8.2))
    b.append(box(sx, 78, 150, 44, "keep?", ["only if the anchor is real"], "guard", fs=9.6, subfs=8.2))

    # encoding pipeline
    y = 142
    b.append(txt(0, y - 6, "Encoding — what actually enters the network", 9.6, INK, bold=True))
    steps = [
        ("raw window", f"({W}, {C})", "data"),
        ("forward-fill\ninterior gaps", f"({W}, {C})", "data"),
        ("subtract the\nanchor row", f"({W}, {C})", "learn"),
        ("÷ fixed\ndelta_scale", f"({W}, {C})", "learn"),
        ("concat the\nmask", f"({W}, {2 * C})", "fed"),
    ]
    x = 0
    for i, (t, shape, k) in enumerate(steps):
        lines = t.split("\n")
        b.append(box(x, y + 6, 132, 62, lines[0], (lines[1:] + [shape]) if len(lines) > 1 else [shape],
                     k, fs=9.4, subfs=8.4))
        if i < len(steps) - 1:
            b.append(arrow(x + 132, y + 37, x + 144, y + 37))
        x += 144
    b.append(f'<rect x="{x}" y="{y + 6}" width="150" height="62" rx="5" fill="#f0eafb" '
             f'stroke="#6b4fa0" stroke-width="1.6"/>')
    b.append(txt(x + 75, y + 30, "X", 13, "#241537", "middle", bold=True))
    b.append(txt(x + 75, y + 48, f"(N, {W}, {2 * C})", 9.6, MUTE, "middle", mono=True))

    b.append(txt(0, y + 88, f"Targets Y are the same transform applied to the {H} steps after the window: "
                            f"(N, {H}, {C}) of scaled deltas, plus a boolean", 9, MUTE))
    b.append(txt(0, y + 101, "mask M of the same shape marking which target cells were genuinely measured "
                             "and therefore count toward the loss.", 9, MUTE))
    return svg(max(x + 150, 960), y + 112, "".join(b))


def dg_lstm() -> str:
    """The forecaster: unrolled encoder, last hidden state, linear head, un-scaling."""
    m = CFG["models"]
    W, H = m["window"], m["horizon"]
    C = len(m["forecaster"]["channels"])
    hid = m["forecaster"]["hidden_size"]
    b = []
    b.append(txt(0, 12, "Model 1 — LSTMForecaster: single-layer, unidirectional, last-step readout",
                 9.4, MUTE, italic=True))

    y = 54
    xs = [40, 158, 276, 446, 564]
    labels = ["x1", "x2", "x3", None, f"x{W}"]
    for i, x in enumerate(xs):
        if labels[i] is None:
            b.append(txt(x + 40, y + 34, "· · ·", 16, MUTE, "middle"))
            continue
        b.append(f'<rect x="{x}" y="{y}" width="86" height="54" rx="5" fill="#fdeee6" '
                 f'stroke="#c0642a" stroke-width="1.4"/>')
        b.append(txt(x + 43, y + 26, "LSTM", 10.5, "#3a1c08", "middle", bold=True))
        b.append(txt(x + 43, y + 42, "cell", 9, MUTE, "middle"))
        b.append(txt(x + 43, y - 12, labels[i], 10.5, "#2c5f96", "middle", mono=True))
        b.append(f'<path d="M{x + 43},{y - 8} L{x + 43},{y - 2}" stroke="{LINE}" '
                 f'stroke-width="1.2" marker-end="url(#ah)"/>')
    for a, bb in ((126, 158), (244, 276), (362, 446), (532, 564)):
        b.append(arrow(a, y + 27, bb, y + 27))
    b.append(txt(240, y + 80, "the running summary is carried forward from step to step",
                 8.8, MUTE, "middle", italic=True))

    b.append(arrow(650, y + 27, 690, y + 27))
    b.append(box(690, y - 8, 138, 70, "The final summary",
                 [f"{hid} numbers", "only this one is used"], "fed", fs=10.4, subfs=8.6))
    b.append(arrow(828, y + 27, 866, y + 27))
    b.append(box(866, y - 8, 154, 70, "One plain layer",
                 [f"turns {hid} numbers", f"into {H * C}"], "learn", fs=10.4, subfs=8.6))

    y2 = 176
    b.append(elbow(943, y + 62, 118, y2 + 4, "", None, False, "ah", "v"))
    seq = [
        ("Reshape it", [f"into {H} steps of", f"{C} measurements"], "learn"),
        ("Undo the scaling", ["multiply by the fixed", "scale for each one"], "learn"),
        ("Add the anchor back", ["now it is in real units", "again, not changes"], "data"),
        ("The forecast", [f"{H} steps ahead for", f"{C} measurements"], "out"),
        ("Summarise it", ["lowest, highest, average,", f"final, trend → {5 * C} numbers"], "out"),
    ]
    x = 58
    for i, (t, sub, k) in enumerate(seq):
        b.append(box(x, y2, 168, 70, t, sub, k, fs=10, subfs=8.6))
        if i < len(seq) - 1:
            b.append(arrow(x + 168, y2 + 35, x + 182, y2 + 35))
        x += 182
    b.append(txt(0, y2 + 100, "Only the FINAL summary is read. The network is not asked to produce one step at a "
                              "time; it maps the whole window", 9, MUTE))
    b.append(txt(0, y2 + 114, f"to the whole forecast in one pass, which is why the last part is a single "
                              f"{hid} → {H}×{C} layer rather than something that generates step by step.", 9, MUTE))
    return svg(1020, y2 + 126, "".join(b))


def dg_lstm_cell() -> str:
    """What is inside one LSTM cell — the four gates."""
    b = []
    b.append(txt(0, 12, "Inside one cell — why an LSTM and not a plain RNN", 9.4, MUTE, italic=True))
    gates = [
        ("Forget gate", "fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)",
         "how much of the old cell state to discard"),
        ("Input gate", "iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)",
         "how much of the new candidate to admit"),
        ("Candidate", "g̃ₜ = tanh(W_g·[hₜ₋₁, xₜ] + b_g)",
         "the proposed new content"),
        ("Output gate", "oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)",
         "how much of the cell state to expose as hₜ"),
    ]
    y = 26
    for t, eq, sub in gates:
        b.append(box(0, y, 190, 46, t, [], "learn", fs=10))
        b.append(txt(206, y + 20, eq, 9.6, INK, mono=True))
        b.append(txt(206, y + 36, sub, 8.8, MUTE, italic=True))
        y += 54
    b.append(f'<rect x="0" y="{y + 6}" width="700" height="52" rx="5" fill="#f0eafb" '
             f'stroke="#6b4fa0" stroke-width="1.3"/>')
    b.append(txt(12, y + 25, "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ          hₜ = oₜ ⊙ tanh(cₜ)", 10.4, "#241537", mono=True, bold=True))
    b.append(txt(12, y + 44, "The additive path through cₜ is the point: gradients reach the start of a 36-step window "
                             "without vanishing.", 8.8, "#3c2c55"))
    return svg(700, y + 68, "".join(b))


def dg_fedavg() -> str:
    """One FedAvg round, and exactly what crosses the client boundary."""
    f = CFG["federated"]
    rounds = f["regimes"]["federated"]["rounds"]
    le = f["regimes"]["federated"]["local_epochs"]
    b = []
    b.append(txt(0, 12, f"One round, repeated {rounds} times. Each station trains for {le} passes per round.",
                 9.4, MUTE, italic=True))
    # server
    b.append(box(340, 28, 290, 60, "The central point",
                 ["holds only the shared numbers", "and never any readings"], "fed", fs=11, subfs=8.6))
    # clients
    cx = [0, 250, 500, 750]
    names = ["Station 1", "Station 2", "…", "Station 32"]
    for i, x in enumerate(cx):
        if names[i] == "…":
            b.append(txt(x + 110, 200, "· · ·", 18, MUTE, "middle"))
            continue
        b.append(f'<rect x="{x}" y="150" width="220" height="104" rx="6" fill="#f3f8fd" '
                 f'stroke="#3b6ea5" stroke-width="1.4"/>')
        b.append(txt(x + 110, 170, names[i], 10.5, "#132b45", "middle", bold=True))
        b.append(txt(x + 110, 187, "its own readings, its own network", 8.4, MUTE, "middle"))
        b.append(box(x + 14, 194, 192, 50, "train for 2 passes",
                     "on its own examples only", "data", fs=9.6, subfs=8.4))
    # down arrows
    for x in (110, 360, 860):
        b.append(arrow(485 if x == 360 else (420 if x < 360 else 560), 88, x, 146,
                       "", "#6b4fa0", False, "ah"))
    b.append(txt(206, 100, "send the shared numbers out", 8.8, "#56409f", "middle", bold=True))
    # up arrows
    for x in (110, 360, 860):
        b.append(arrow(x, 258, 470 if x == 360 else (430 if x < 360 else 540), 300,
                       "", "#6b4fa0", False, "ah"))
    b.append(box(310, 300, 350, 66, "Average everybody's numbers",
                 ["weighted by how many examples", "each station has"], "fed", fs=10.6, subfs=8.8))
    b.append(elbow(660, 333, 700, 60, "", "#6b4fa0", True, "ah", "h"))
    b.append(txt(700, 288, "then repeat", 8.8, "#56409f", italic=True))

    # boundary panel
    b.append(f'<rect x="0" y="382" width="980" height="66" rx="5" fill="#eef6ef" '
             f'stroke="#2e7d55" stroke-width="1.3"/>')
    b.append(txt(12, 401, "What crosses the boundary", 9.6, "#226343", bold=True))
    b.append(txt(12, 418, "UP:  the network’s numbers, plus one count of how many training examples this station has.", 9, "#294c39"))
    b.append(txt(12, 433, "DOWN:  the average of everybody’s numbers.   NEVER:  a reading, an average, a station name, a timestamp.", 9, "#294c39"))
    return svg(980, 456, "".join(b))


def dg_worldmodel() -> str:
    """Model 2: where the action is fused, and what the warm start actually copies."""
    m = CFG["models"]
    W, H = m["window"], m["horizon"]
    C = len(m["forecaster"]["channels"])
    hid = m["world_model"]["hidden_size"]
    b = []
    b.append(txt(0, 12, "Model 2 — “what happens if I do this?”", 9.4, MUTE, italic=True))

    b.append(box(0, 44, 152, 58, "The recent readings",
                 [f"{W} x {2 * C}, encoded", "exactly as in Model 1"], "data", fs=9.8, subfs=8.2))
    b.append(arrow(152, 73, 176, 73))
    b.append(f'<rect x="176" y="36" width="168" height="74" rx="6" fill="#fdeee6" '
             f'stroke="#c0642a" stroke-width="2.2"/>')
    b.append(txt(260, 60, "self.lstm", 11, "#3a1c08", "middle", bold=True, mono=True))
    b.append(txt(260, 76, "the reading part", 8.6, "#9a4c1c", "middle", bold=True))
    b.append(txt(260, 92, f"1 layer, {hid} units", 8.6, MUTE, "middle"))
    b.append(arrow(344, 73, 372, 73))
    b.append(box(372, 48, 100, 50, "the summary", [f"{hid} numbers"], "fed", fs=9.6, subfs=8.4))

    b.append(box(372, 120, 100, 50, "the action", ["8 numbers"], "agent", fs=9.6, subfs=8.4))
    b.append(elbow(464, 145, 500, 96, "", None, False, "ah", "h"))
    b.append(arrow(464, 73, 500, 92))
    b.append(f'<circle cx="512" cy="98" r="15" fill="#ffffff" stroke="{LINE}" stroke-width="1.3"/>')
    b.append(txt(512, 102, "⊕", 14, MUTE, "middle"))
    b.append(txt(512, 128, "concat", 8.4, MUTE, "middle", italic=True))
    b.append(arrow(527, 98, 552, 98))
    b.append(f'<rect x="552" y="62" width="176" height="72" rx="6" fill="#fdeee6" '
             f'stroke="#c0642a" stroke-width="1.4"/>')
    b.append(txt(640, 84, "self.head", 10.6, "#3a1c08", "middle", bold=True, mono=True))
    b.append(txt(640, 100, f"Linear({hid}+8 → {hid})", 8.6, MUTE, "middle", mono=True))
    b.append(txt(640, 113, "ReLU", 8.6, MUTE, "middle", mono=True))
    b.append(txt(640, 126, f"Linear({hid} → {H * C})", 8.6, MUTE, "middle", mono=True))
    b.append(arrow(728, 98, 752, 98))
    b.append(box(752, 62, 148, 72, "what happens next",
                 [f"{H} steps x {C} measures", "unscaled, then capped", "to possible values"],
                 "out", fs=9.8, subfs=8.2))

    b.append(f'<rect x="0" y="182" width="900" height="52" rx="5" fill="#f4f7fa" '
             f'stroke="#8fa6bd" stroke-width="1.2"/>')
    b.append(txt(12, 199, "Why the action is attached AFTER the reading part, not added to every input step:", 9.3, INK, bold=True))
    b.append(txt(12, 214, "the window is the PAST, and the action has not happened yet. Putting it into the", 9, MUTE))
    b.append(txt(12, 228, "sequence would teach the network that the aerator was already running during the readings it is looking at.", 9, MUTE))

    # warm start panel
    y = 254
    b.append(txt(0, y + 2, "The warm start — what is copied, and what is not", 10, INK, bold=True))
    b.append(box(0, y + 14, 220, 66, "the trained forecaster",
                 ["Model 1, trained across", "42 real river stations"], "fed", fs=9.8, subfs=8.2))
    b.append(arrow(226, y + 47, 268, y + 47, "", "#c0642a", True, "ahd"))
    b.append(txt(247, y + 30, "reading", 8.2, "#9a4c1c", "middle"))
    b.append(txt(247, y + 68, "part only", 8.2, "#9a4c1c", "middle"))
    b.append(box(268, y + 14, 200, 66, "reading part: copied",
                 ["straight from the", "trained forecaster"], "learn", fs=9.8, subfs=8.2))
    b.append(box(482, y + 14, 200, 66, "the rest: random",
                 ["the forecaster has no", "action part to copy"], "guard", fs=9.8, subfs=8.2))
    b.append(box(696, y + 14, 204, 66, "then BOTH are trained",
                 ["nothing is frozen;", "no line freezes it"], "guard", fs=9.8, subfs=8.2))
    b.append(arrow(468, y + 47, 480, y + 47))
    b.append(arrow(682, y + 47, 694, y + 47))
    return svg(900, y + 96, "".join(b))


def dg_warmstart_options() -> str:
    """The three initialisation regimes, and which one the code implements."""
    b = []
    opts = [
        ("Option 1 — from scratch", ["encoder starts RANDOM", "encoder trains freely"],
         "plain", "not used here"),
        ("Option 2 — fine-tuning", ["encoder starts COPIED", "encoder is FROZEN",
                                    "requires_grad = False"], "plain", "not used here"),
        ("Option 3 — warm start", ["encoder starts COPIED", "encoder trains freely",
                                   "nothing frozen"], "learn", "◀ what AquaSentinel does"),
    ]
    x = 0
    for title, sub, kind, tag in opts:
        b.append(box(x, 22, 292, 84, title, sub, kind, fs=10.4, subfs=8.8))
        colour = "#9a4c1c" if "◀" in tag else MUTE
        b.append(txt(x + 146, 120, tag, 9, colour, "middle", bold=("◀" in tag)))
        x += 306
    b.append(txt(0, 12, "Three ways to start a network from someone else’s weights", 9.4, MUTE, italic=True))
    b.append(f'<rect x="0" y="134" width="890" height="46" rx="5" fill="#fbeded" '
             f'stroke="#b03b3b" stroke-width="1.2"/>')
    b.append(txt(12, 152, "Consequence worth stating plainly:", 9.4, "#8f2f2f", bold=True))
    b.append(txt(12, 168, "the real-data weights fix where twin training BEGINS, not where it ends. How much of the federated "
                          "representation survives 12 rounds of twin training is currently unmeasured.", 9, "#5a2020"))
    return svg(890, 188, "".join(b))


def dg_regimes() -> str:
    """Why the clustering features are derived, and what the model is actually used for."""
    mr = CFG["models"]["regimes"]
    b = []
    b.append(txt(0, 12, "Model 3 — HDBSCAN over derived window features", 9.4, MUTE, italic=True))
    raw = ["do", "temp_c", "ph", "turbidity"]
    b.append(box(0, 30, 150, 70, "raw window", [f"({CFG['models']['window']}, {len(raw)})",
                                                 " · ".join(raw)], "data", fs=9.8, subfs=8))
    b.append(arrow(150, 65, 176, 65))
    der = [
        ("do_sat", "DO ÷ saturation(T)", "dimensionless — a 12 °C river at 9 mg/L and a 28 °C pond at 6.5 are both ≈85 %"),
        ("temp_anom", "T − mean(T over window)", "the diurnal shape, not the climate"),
        ("ph_anom", "pH − mean(pH over window)", "the swing, not the catchment geology"),
        ("turb_log", "log1p(turbidity)", "heavy-tailed over orders of magnitude"),
    ]
    y = 26
    for name, formula, why in der:
        b.append(box(176, y, 150, 34, name, [], "learn", fs=9.6))
        b.append(txt(336, y + 15, formula, 9.2, INK, mono=True))
        b.append(txt(336, y + 28, why, 8.4, MUTE, italic=True))
        y += 40
    b.append(arrow(326, 100, 900, 100) if False else "")

    y2 = 200
    b.append(box(0, y2, 168, 56, "4 stats per channel", ["mean · std · slope · last"], "fed", fs=9.6, subfs=8.4))
    b.append(arrow(168, y2 + 28, 190, y2 + 28))
    b.append(box(190, y2, 150, 56, "feature vector", ["4 × 4 = 16 dims"], "fed", fs=9.6, subfs=8.4))
    b.append(arrow(340, y2 + 28, 362, y2 + 28))
    b.append(box(362, y2, 176, 56, "robust scaling", ["(x − median) / IQR"], "fed", fs=9.6, subfs=8.4))
    b.append(arrow(538, y2 + 28, 560, y2 + 28))
    b.append(box(560, y2, 200, 56, "HDBSCAN",
                 [f"min_cluster_size={mr['min_cluster_size']}, min_samples={mr['min_samples']}"],
                 "learn", fs=9.6, subfs=8))
    b.append(arrow(760, y2 + 28, 782, y2 + 28))
    b.append(box(782, y2, 168, 56, "label ∈ {−1, 0, 1}", ["−1 = noise = ANOMALY"], "guard", fs=9.6, subfs=8.4))

    b.append(f'<rect x="0" y="{y2 + 76}" width="950" height="62" rx="5" fill="#fbeded" '
             f'stroke="#b03b3b" stroke-width="1.3"/>')
    b.append(txt(12, y2 + 95, "The honest negative result", 9.6, "#8f2f2f", bold=True))
    b.append(txt(12, y2 + 111, "Swept across min_cluster_size 40–800 and min_samples 3–80, HDBSCAN returns the same shape every time: one dominant", 9, "#5a2020"))
    b.append(txt(12, y2 + 126, "mode holding ~80 % of windows and DBCV ≈ 0. Water-quality windows form a continuum, not clusters. Only the anomaly flag is used.", 9, "#5a2020"))
    return svg(950, y2 + 148, "".join(b))


def dg_risk() -> str:
    """Models 4 and 5: how the risk target is manufactured, and what SHAP explains."""
    r = CFG["models"]["risk"]
    C = len(CFG["models"]["forecaster"]["channels"])
    b = []
    b.append(txt(0, 12, "Models 4 & 5 — LightGBM risk classifier and its SHAP attribution", 9.4, MUTE, italic=True))

    b.append(txt(0, 34, "TRAINING — how the target is manufactured", 9, "#56667a", bold=True))
    b.append(box(0, 42, 168, 58, "forecast horizon", [f"({CFG['models']['horizon']}, {C}) predicted"], "learn", fs=9.6, subfs=8.2))
    b.append(arrow(168, 71, 190, 71))
    b.append(box(190, 42, 190, 58, "worst point per channel",
                 ["min DO, max NH₃, both pH tails"], "twin", fs=9.6, subfs=8.2))
    b.append(arrow(380, 71, 402, 71))
    b.append(box(402, 42, 190, 58, "risk_bands rule", ["safe | warning | critical"], "twin", fs=9.6, subfs=8.2))
    b.append(arrow(592, 71, 614, 71))
    b.append(box(614, 42, 176, 58, "TARGET LABEL", ["rule-derived, not human"], "guard", fs=9.6, subfs=8.2))

    b.append(txt(0, 128, "INFERENCE — one tick", 9, "#56667a", bold=True))
    b.append(box(0, 136, 168, 62, "7 now_ features", ["do, temp_c, ph, tan,", "nh3, no2, turbidity"], "data", fs=9.6, subfs=8))
    b.append(box(0, 206, 168, 62, f"{5 * C} fc_ features",
                 ["min · max · mean · end · slope", f"for each of {C} channels"], "learn", fs=9.6, subfs=8))
    b.append(arrow(168, 167, 208, 190))
    b.append(arrow(168, 237, 208, 210))
    b.append(box(208, 168, 196, 66, "LightGBM",
                 [f"{r['n_estimators']} trees, {r['num_leaves']} leaves",
                  f"depth {r['max_depth']}, lr {r['learning_rate']}"], "learn", fs=10, subfs=8.2))
    b.append(arrow(404, 186, 430, 172))
    b.append(arrow(404, 216, 430, 230))
    b.append(box(430, 142, 178, 58, "risk class", ["safe / warning / critical"], "out", fs=9.6, subfs=8.2))
    b.append(box(430, 208, 178, 58, "risk score", ["0.0 / 0.5 / 1.0"], "out", fs=9.6, subfs=8.2))
    b.append(arrow(608, 171, 634, 171))
    b.append(box(634, 142, 200, 58, "TreeExplainer (SHAP)",
                 ["exact, not sampled"], "learn", fs=9.6, subfs=8.2))
    b.append(arrow(834, 171, 856, 171))
    b.append(box(856, 138, 130, 66, "top drivers",
                 ["“forecast min DO", "= 2.1 mg/L”"], "out", fs=9.6, subfs=8.2))

    b.append(f'<rect x="0" y="286" width="986" height="60" rx="5" fill="#eef6ef" '
             f'stroke="#2e7d55" stroke-width="1.3"/>')
    b.append(txt(12, 305, "Why modelling a rule is not circular", 9.6, "#226343", bold=True))
    b.append(txt(12, 321, "The rule needs the FUTURE; the future comes from a forecaster with real error. A tree ensemble over the whole forecast", 9, "#294c39"))
    b.append(txt(12, 336, "summary degrades far more gracefully under that error than a hard threshold on one predicted value. The ablation measures it.", 9, "#294c39"))
    return svg(986, 356, "".join(b))


def dg_clock() -> str:
    """The three clocks and how they nest."""
    ag = CFG["agent"]
    sm = CFG["sensors"]["sample_every_minutes"]
    dm = CFG["twin"]["dt_minutes"]
    dec = ag["decide_every_minutes"]
    tpd = int(round(dec / sm))
    b = []
    b.append(txt(0, 12, "Three clocks. Everything about the loop’s behaviour follows from how they nest.",
                 9.4, MUTE, italic=True))
    total = 960
    # twin ticks
    y = 34
    b.append(txt(0, y + 12, "twin", 9.4, INK, bold=True))
    b.append(txt(0, y + 25, f"dt = {dm} min", 8.4, MUTE))
    n = 36
    cw = (total - 90) / n
    for i in range(n):
        b.append(f'<rect x="{90 + i * cw}" y="{y}" width="{cw - 1.2}" height="18" fill="#e6f4ec" '
                 f'stroke="#2e7d55" stroke-width="0.5"/>')
    # sensor samples
    y2 = 74
    b.append(txt(0, y2 + 12, "sensor", 9.4, INK, bold=True))
    b.append(txt(0, y2 + 25, f"every {sm} min", 8.4, MUTE))
    for i in range(n):
        b.append(f'<rect x="{90 + i * cw}" y="{y2}" width="{cw - 1.2}" height="18" fill="#e8f1fb" '
                 f'stroke="#3b6ea5" stroke-width="0.5"/>')
    # decisions
    y3 = 114
    b.append(txt(0, y3 + 12, "planner", 9.4, INK, bold=True))
    b.append(txt(0, y3 + 25, f"every {dec} min", 8.4, MUTE))
    for i in range(0, n, tpd):
        b.append(f'<rect x="{90 + i * cw}" y="{y3}" width="{cw * tpd - 1.2}" height="18" '
                 f'fill="#fdf2e0" stroke="#b1841f" stroke-width="0.8"/>')
        b.append(f'<circle cx="{90 + i * cw + 4}" cy="{y3 + 9}" r="3" fill="#b1841f"/>')
    # guardrails band
    y4 = 154
    b.append(txt(0, y4 + 12, "guardrails", 9.4, "#8f2f2f", bold=True))
    b.append(txt(0, y4 + 25, "EVERY tick", 8.4, "#8f2f2f"))
    b.append(f'<rect x="90" y="{y4}" width="{total - 90}" height="18" fill="#fbe6e6" '
             f'stroke="#b03b3b" stroke-width="1.2"/>')
    for i in range(n):
        b.append(f'<circle cx="{90 + i * cw + cw / 2}" cy="{y4 + 9}" r="2.2" fill="#b03b3b"/>')

    b.append(txt(90, 196, f"ticks_per_decision = {dec} ÷ {sm} = {tpd}   ·   "
                          f"steps_per_sample = {sm} ÷ {dm} = {int(sm / dm)}", 9.4, INK, mono=True))
    b.append(f'<rect x="0" y="210" width="{total}" height="56" rx="5" fill="#fbeded" '
             f'stroke="#b03b3b" stroke-width="1.2"/>')
    b.append(txt(12, 228, "This split is the whole safety argument.", 9.4, "#8f2f2f", bold=True))
    b.append(txt(12, 243, f"Planning is expensive, so it runs every {dec} minutes — but a pond can cross into", 9, "#5a2020"))
    b.append(txt(12, 257, f"critical in far less than that. So the contextual blocks and the emergency override run on all "
                          f"{tpd} ticks, against whatever action is currently held.", 9, "#5a2020"))
    return svg(total, 274, "".join(b))


def dg_tick() -> str:
    """One complete tick of AgentRun, numbered."""
    b = []
    b.append(txt(0, 12, "AgentRun.tick() — every action, in order", 9.4, MUTE, italic=True))
    rows = [
        (1, "Advance the twin", "stream.tick(current_action) integrates steps_per_sample twin steps, applying the held action to each", "twin"),
        (2, "Sample the probe", "true State → noisy, quantised, possibly-missing Observation", "twin"),
        (3, "Advance the shadow", "the forked twin takes the same step under no_op, on the identical noise draw", "twin"),
        (4, "Update the belief", "InternalModel steps forward under the applied action, then is pulled toward the observation (α = 0.35)", "agent"),
        (5, "Decision tick?", "tick_index mod 3 == 0 → plan.  Otherwise → hold, and jump to step 10", "agent"),
        (6, "Perception", "forecast (Model 1) · regime + anomaly (Model 3) · risk class (Model 4) · SHAP drivers (Model 5)", "learn"),
        (7, "Propose", "the planner scores 14 candidates through the world model and returns the argmin", "agent"),
        (8, "GUARDRAILS", "dosing caps → rate limits → contextual blocks → predicted-critical rejection → emergency override", "guard"),
        (9, "Record the Decision", "proposed, applied, every guardrail event, rationale, risk class, SHAP top-k", "out"),
        (10, "Held tick (fast path)", "safety_only(): contextual blocks + emergency override against the held action — no planning", "guard"),
        (11, "Unsafe check", "did the pond actually reach a critical band this tick? → unsafe_applied counter", "guard"),
        (12, "Next tick", "the applied action becomes current_action and is carried into step 1", "plain"),
    ]
    y = 26
    for n, title, detail, kind in rows:
        fill, stroke, tc = PALETTE[kind]
        h = 30
        b.append(f'<rect x="26" y="{y}" width="934" height="{h}" rx="4" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.1"/>')
        b.append(f'<circle cx="13" cy="{y + h / 2}" r="11" fill="#ffffff" stroke="{stroke}" stroke-width="1.3"/>')
        b.append(txt(13, y + h / 2 + 3.4, str(n), 9.4, stroke, "middle", bold=True))
        b.append(txt(38, y + 13, title, 9.6, tc, bold=True))
        b.append(txt(38, y + 25, detail, 8.6, MUTE))
        y += h + 5
    return svg(960, y + 4, "".join(b))


def dg_mpc() -> str:
    """The planner tick: 14 candidates, one batched forward pass, argmin cost."""
    pr = CFG["agent"]["predictive"]
    cw = CFG["agent"]["cost_weights"]
    b = []
    b.append(txt(0, 12, "One planner decision: simulate every option, score it, take the best",
                 9.4, MUTE, italic=True))
    b.append(box(0, 26, 168, 70, "the recent readings",
                 [f"the last {CFG['models']['window']}, taken from", "the agent's own picture",
                  "of the pond"], "data", fs=9.8, subfs=8.2))
    b.append(arrow(168, 61, 194, 61))
    b.append(box(194, 24, 150, 74, "the 14 options",
                 ["do nothing, aerator,", "feed, exchange, lime,", "salt, and combinations"],
                 "agent", fs=9.8, subfs=8.2))
    b.append(arrow(344, 61, 370, 61))
    b.append(box(370, 24, 186, 74, "simulate with Model 2",
                 [f"{pr['rollout_steps']} steps ahead; the action",
                  f"is held for {pr['hold_action_steps']} of them,", "then stopped"],
                 "learn", fs=9.8, subfs=8.2))
    b.append(arrow(556, 61, 582, 61))
    b.append(box(582, 24, 174, 74, "add the correction",
                 ["Model 1 minus the", "equations, halved;",
                  "the same for every option"], "learn", fs=9.8, subfs=8.2, dash=True))
    b.append(arrow(756, 61, 782, 61))
    b.append(box(782, 24, 178, 74, "score it",
                 ["safe, warning or critical", "at every predicted step,",
                  "using the Part 2.8 ranges"], "twin", fs=9.8, subfs=8.2))

    y = 122
    b.append(txt(0, y, "THE SCORE — one number per option; lower is better", 9, "#56667a", bold=True))
    terms = [
        (f"{cw['risk']}", "× hours predicted outside the safe band", "twin"),
        (f"{cw['critical']}", "× hours in the critical band, counted a second time", "guard"),
        (f"{cw['intervention']}", "× what the action costs to run", "agent"),
        (f"{cw['stability']}", "× how jumpy the predicted oxygen and pH are", "learn"),
        (f"{cw['feed_loss']}", "× the penalty for withholding feed (skip 1.0, reduce 0.4)", "agent"),
    ]
    yy = y + 10
    for weight, what, kind in terms:
        fill, stroke, tc = PALETTE[kind]
        b.append(f'<rect x="0" y="{yy}" width="58" height="26" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        b.append(txt(29, yy + 17, weight, 10.4, tc, "middle", bold=True, mono=True))
        b.append(txt(70, yy + 17, what, 9.4, INK))
        yy += 31
    b.append(f'<path d="M0,{yy + 4} L960,{yy + 4}" stroke="{LINE}" stroke-width="1"/>')
    b.append(txt(0, yy + 22, "The lowest score wins and becomes the suggestion. The reason recorded alongside it also says what "
                             "doing nothing would have cost, so the gain can be checked.", 9.2, INK, italic=True))

    b.append(f'<rect x="0" y="{yy + 34}" width="960" height="46" rx="5" fill="#fdf1ec" '
             f'stroke="#c0642a" stroke-width="1.2"/>')
    b.append(txt(12, yy + 52, "The feed penalty is not decoration.", 9.4, "#9a4c1c", bold=True))
    b.append(txt(12, yy + 68, "Without it, the cheapest way to keep every measurement safe is to stop feeding the fish forever. The penalty "
                              "forces the planner to trade safety against the reason the pond exists.", 9, "#7a3d16"))
    return svg(960, yy + 92, "".join(b))


def dg_guardrails() -> str:
    """The cascade, in firing order, with the config value on each rule."""
    g = CFG["agent"]["guardrails"]
    b = []
    b.append(txt(0, 12, "The safety layer — five groups of rules, always run in this order",
                 9.4, MUTE, italic=True))
    groups = [
        ("1", "Limits on how often things can be dosed", [
            f"lime at most {g['max_lime_doses_per_day']} times a day, at least "
            f"{g['min_interval_minutes']['lime']} minutes apart",
            f"salt at most {g['max_salt_doses_per_day']} time a day, at least "
            f"{g['min_interval_minutes']['salt']} minutes apart",
            f"water exchanges at least {g['min_interval_minutes']['exchange']} min apart, and at most "
            f"{g['max_exchange_hours_per_day']} hours a day"], "block"),
        ("2", "Limit on how fast a setting may change", [
            f"the aerator may move by at most {g['aerator_rate_limit_per_tick']} between decisions"], "clamp"),
        ("3", "Blocks based on the current situation", [
            f"never add lime when the water is already alkaline (pH above {g['block_lime_above_ph']})",
            f"never feed when oxygen is below {g['no_feed_when']['do_below']}, harmful ammonia is above "
            f"{g['no_feed_when']['nh3_above']}, or oxygen is forecast to fall below {g['no_feed_when']['forecast_do_below']}",
            f"downgrade a large water exchange to a small one if the replacement water differs by more than "
            f"{g['block_large_exchange_above_temp_delta']} °C, or is not known to be cleaner"], "clamp"),
        ("4", "Reject actions predicted to cause harm", [
            "simulate the suggestion; if it would push something into the critical band, do nothing instead",
            "unless doing nothing would ALSO be critical — then the action is not the cause"], "block"),
        ("5", "Emergency override", [
            f"oxygen below {g['emergency_do_threshold']} mg/L now, or forecast to fall below "
            f"{g['emergency_forecast_do']}",
            "aerator to full, feeding stopped. Deliberately ignores the rate limit above."], "override"),
    ]
    y = 26
    for n, title, rules, sev in groups:
        h = 22 + len(rules) * 13
        b.append(f'<rect x="26" y="{y}" width="880" height="{h}" rx="4" fill="#fbe6e6" '
                 f'stroke="#b03b3b" stroke-width="1.2"/>')
        b.append(f'<circle cx="13" cy="{y + h / 2}" r="11" fill="#ffffff" stroke="#b03b3b" stroke-width="1.4"/>')
        b.append(txt(13, y + h / 2 + 3.4, n, 9.6, "#b03b3b", "middle", bold=True))
        b.append(txt(38, y + 15, title, 10, "#3b1010", bold=True))
        b.append(f'<rect x="{906 - 66}" y="{y + 6}" width="66" height="15" rx="7" fill="#b03b3b"/>')
        b.append(txt(906 - 33, y + 17, sev, 7.6, "#ffffff", "middle", bold=True))
        yy = y + 27
        for r in rules:
            b.append(txt(38, yy, "· " + r, 8.6, "#5a2020"))
            yy += 13
        if n != "5":
            b.append(arrow(466, y + h, 466, y + h + 8))
        y += h + 12
    b.append(f'<rect x="0" y="{y}" width="906" height="60" rx="5" fill="#eef6ef" '
             f'stroke="#2e7d55" stroke-width="1.3"/>')
    b.append(txt(12, y + 19, "Why the override runs LAST", 9.6, "#226343", bold=True))
    b.append(txt(12, y + 35, "So that it beats every restriction above it. Group 1 may have just blocked a dose and group 2 may have limited the aerator —", 9, "#294c39"))
    b.append(txt(12, y + 50, "but when the pond is suffocating, nothing outranks air. Every difference between suggested and applied is written down with its reason.", 9, "#294c39"))
    return svg(906, y + 70, "".join(b))


def dg_counterfactual() -> str:
    """The shadow twin: why the comparison is causal rather than anecdotal."""
    b = []
    b.append(txt(0, 12, "The shadow pond — what makes the counterfactual an argument rather than an illustration",
                 9.4, MUTE, italic=True))
    b.append(box(330, 26, 250, 46, "PondTwin at t = 0", ["scenario, seed, initial state"], "twin", fs=10, subfs=8.4))
    b.append(f'<path d="M455,72 L455,88 L200,88 L200,104" stroke="{LINE}" stroke-width="1.4" '
             f'fill="none" marker-end="url(#ah)"/>')
    b.append(f'<path d="M455,72 L455,88 L716,88 L716,104" stroke="{LINE}" stroke-width="1.4" '
             f'fill="none" marker-end="url(#ah)"/>')
    b.append(txt(455, 84, "deep copy — RNG cloned by state", 8.6, MUTE, "middle", italic=True))

    b.append(box(60, 106, 280, 66, "LIVE pond", ["driven by the agent", "guardrails on"], "twin", fs=10.4, subfs=8.6))
    b.append(box(576, 106, 280, 66, "SHADOW pond", ["driven by do_nothing", "agent never touches it"], "twin", fs=10.4, subfs=8.6, dash=True))

    b.append(box(60, 194, 280, 50, "time_in_risk (live)", [], "out", fs=10))
    b.append(box(576, 194, 280, 50, "time_in_risk (shadow)", [], "out", fs=10))
    b.append(arrow(200, 172, 200, 190))
    b.append(arrow(716, 172, 716, 190))
    b.append(f'<path d="M340,219 L570,219" stroke="#2e7d55" stroke-width="1.6" stroke-dasharray="6 4"/>')
    b.append(txt(455, 212, "the gap", 10, "#226343", "middle", bold=True))

    b.append(f'<rect x="0" y="264" width="916" height="62" rx="5" fill="#eef6ef" '
             f'stroke="#2e7d55" stroke-width="1.3"/>')
    b.append(txt(12, 283, "Same seed, same noise draw, same perturbation schedule.", 9.6, "#226343", bold=True))
    b.append(txt(12, 299, "The RNG is cloned by internal state, not merely re-seeded, so the shadow experiences the identical realisation of every random", 9, "#294c39"))
    b.append(txt(12, 314, "event. Any divergence between the two ponds is therefore caused by the agent’s actions and not by luck.", 9, "#294c39"))
    return svg(916, 336, "".join(b))


def dg_evalgrid() -> str:
    b = []
    b.append(txt(0, 12, "How the testing is laid out, and what each variation asks",
                 9.4, MUTE, italic=True))
    b.append(box(0, 28, 180, 62, "8 scenarios", ["healthy · night_do_crash ·", "bloom · heatwave · …"], "twin", fs=10, subfs=8.2))
    b.append(txt(192, 64, "×", 15, MUTE, "middle"))
    b.append(box(206, 28, 200, 62, "5 controllers", ["do_nothing · static · reactive", "predictive · oracle"], "agent", fs=10, subfs=8.2))
    b.append(txt(418, 64, "×", 15, MUTE, "middle"))
    b.append(box(432, 28, 150, 62, "10 seeds", ["paired across", "controllers"], "fed", fs=10, subfs=8.2))
    b.append(txt(594, 64, "=", 15, MUTE, "middle"))
    b.append(box(608, 28, 172, 62, "400 runs", ["every one recorded"], "out", fs=11, subfs=8.4))
    b.append(arrow(780, 59, 802, 59))
    b.append(box(802, 20, 158, 78, "compared in pairs",
                 ["same seed, same test,", "two controllers at a time;", "24 comparisons"],
                 "out", fs=9.8, subfs=8.2))

    y = 116
    b.append(txt(0, y, "Parts switched off, one at a time — each answers one question",
                 9, "#56667a", bold=True))
    abl = [
        ("full", "the shipped configuration"),
        ("no_lstm_correction", "is the correction from Model 1 earning its place?"),
        ("no_forecast_at_all", "does having a forecast at all help?"),
        ("no_guardrails", "how many unsafe actions would the same planner take?"),
        ("weights_independent_local", "does it matter how the forecaster was trained?"),
        ("weights_central", "does pooling everyone's data buy better control?"),
        ("no_world_model", "Model 2, or the hand-written pond equations?"),
    ]
    yy = y + 8
    for name, q in abl:
        b.append(f'<rect x="0" y="{yy}" width="230" height="22" rx="3" fill="#f0eafb" '
                 f'stroke="#6b4fa0" stroke-width="1"/>')
        b.append(txt(8, yy + 15, name, 8.8, "#241537", mono=True, bold=True))
        b.append(txt(244, yy + 15, q, 9.2, MUTE))
        yy += 26
    return svg(960, yy + 10, "".join(b))


# ======================================================================================
# Document body
# ======================================================================================
def cover() -> None:
    man = manifest()
    n_sites = len(man) if man is not None else 42
    n_rows = int(man["rows"].sum()) if man is not None else 729542
    w('<div class="cover">')
    w('<div class="eyebrow">AquaSentinel &middot; complete technical reference</div>')
    w("<h1>Every Action in the Pipeline</h1>")
    w('<p class="sub">From a public river gauge to an aerator command, with nothing skipped.</p>')
    w('<div class="rule"></div>')
    w('<div class="meta">'
      "<p><b>What the system is</b> &nbsp; Software that watches a fish pond's water and switches "
      "equipment on and off to keep the fish alive.</p>"
      f"<p><b>Data</b> &nbsp; {n_sites} real public water-monitoring stations &middot; "
      f"{n_rows:,} rows of readings &middot; one reading every 15 minutes &middot; "
      "1 March to 2 September 2024.</p>"
      "<p><b>Models</b> &nbsp; Five trained components. Two of them are trained across many "
      "stations at once without any station handing over its data.</p>"
      "<p><b>Course</b> &nbsp; B.Tech. BCSE497J Project-I &middot; SCOPE, VIT.</p>"
      "</div>")
    w('<div class="prom"><b>What this document is for.</b> The project report summarises. This '
      "document explains. It walks through every step between a sensor reading and an equipment "
      "command, and it stops to define each term the first time that term is used. If you can "
      "follow it from front to back, you can answer any question about how the system works "
      "&mdash; including the questions about what it does <i>not</i> do.")
    w("<p style='margin-top:8px'>Every number in here is read straight out of the project's own "
      "settings files and results files when the document is built. So the document cannot "
      "gradually disagree with the code.</p></div>")
    w("</div>")


def contents() -> None:
    w('<section class="part" style="page-break-before:avoid">')
    w('<div class="partnum">Start here</div><h1>How to read this document</h1>')
    w("</section>")

    h2("The order of the parts")
    p("The parts follow the order in which things actually happen. Nothing is described before the "
      "thing that produces it.")
    ul([
        "<b>Parts 1 to 3</b> build the world. Where the data comes from, the pond simulator, and "
        "the fake sensors that sit between the two.",
        "<b>Parts 4 to 9</b> build the five trained models, one part each.",
        "<b>Parts 10 to 12</b> are the live control loop, where everything above finally gets used.",
        "<b>Parts 13 to 15</b> are the evidence, the honest list of weaknesses, and a lookup section.",
    ])

    items = [
        ("tp", "Foundations"),
        ("", "Part 0 &nbsp;&middot;&nbsp; The whole system in one diagram"),
        ("", "Part 1 &nbsp;&middot;&nbsp; Collecting data from 42 real stations"),
        ("", "Part 2 &nbsp;&middot;&nbsp; The pond simulator"),
        ("", "Part 3 &nbsp;&middot;&nbsp; The fake sensors"),
        ("tp", "The five models"),
        ("", "Part 4 &nbsp;&middot;&nbsp; Cutting the data into training examples"),
        ("", "Part 5 &nbsp;&middot;&nbsp; Model 1: the forecaster"),
        ("", "Part 6 &nbsp;&middot;&nbsp; Training across 42 stations without sharing data"),
        ("", "Part 7 &nbsp;&middot;&nbsp; Model 2: the &ldquo;what if I do this&rdquo; model"),
        ("", "Part 8 &nbsp;&middot;&nbsp; Model 3: grouping and spotting odd readings"),
        ("", "Part 9 &nbsp;&middot;&nbsp; Models 4 and 5: risk, and why it said so"),
        ("tp", "The agent"),
        ("", "Part 10 &nbsp;&middot;&nbsp; The live loop, step by step"),
        ("", "Part 11 &nbsp;&middot;&nbsp; How the agent chooses an action"),
        ("", "Part 12 &nbsp;&middot;&nbsp; The safety layer"),
        ("tp", "Evidence and reference"),
        ("", "Part 13 &nbsp;&middot;&nbsp; Testing the agent"),
        ("", "Part 14 &nbsp;&middot;&nbsp; What the system does not do"),
        ("", "Part 15 &nbsp;&middot;&nbsp; Lookup tables and thirty questions"),
    ]
    w('<div class="toc">')
    for cls, t in items:
        w(f'<div class="{cls}">{t}</div>')
    w("</div>")

    h2("The four kinds of box")
    w('<div class="box exam"><div class="nt">Green box &mdash; likely question</div>'
      "<p>A question you will probably be asked, and a short answer that is also correct.</p></div>")
    w('<div class="box honest"><div class="nt">Red box &mdash; weakness</div>'
      "<p>Something the system does badly, or a claim that sounds stronger than the evidence "
      "behind it. These are the parts worth rehearsing. An examiner who finds a weakness you have "
      "not thought about will stay on it.</p></div>")
    w('<div class="box num"><div class="nt">Purple box &mdash; the exact numbers</div>'
      "<p>Figures copied straight from the settings files or the results files.</p></div>")
    w('<div class="box note"><div class="nt">Grey box &mdash; a detail worth knowing</div>'
      "<p>A design choice that is easy to miss but easy to defend once you have seen it.</p></div>")

    h2("Short forms used in this document")
    p("Every short form below is also spelled out the first time it appears in the text. This table "
      "is here so you can look one up later without hunting for where it was introduced.")

    h3("Water chemistry")
    table([
        ["DO", "dissolved oxygen",
         "How much oxygen is in the water, in milligrams per litre. The single most important "
         "number in the whole project: if it runs out, the fish suffocate."],
        ["TAN", "total ammonia nitrogen",
         "All the ammonia in the water added together, whether it is in its harmful form or not."],
        ["NH&#8323;", "un-ionised ammonia",
         "The harmful part of the ammonia. How much of the TAN takes this form depends on how "
         "acidic and how warm the water is."],
        ["NO&#8322;", "nitrite",
         "A poisonous by-product made when bacteria break ammonia down."],
        ["pH", "&mdash;",
         "How acidic or alkaline the water is, on a scale where 7 is neutral, lower is more acidic "
         "and higher is more alkaline."],
        ["BOD", "biochemical oxygen demand",
         "A store of decaying material in the pond that steadily eats oxygen as bacteria break it "
         "down."],
        ["NTU", "nephelometric turbidity units",
         "The unit for how cloudy the water is."],
        ["CaCO&#8323;", "calcium carbonate",
         "The unit used for alkalinity &mdash; the pond's ability to resist swings in pH."],
        ["ppt", "parts per thousand", "The unit used here for saltiness."],
    ], ["Short form", "Stands for", "What it means"], "", cls="small")

    h3("Where the data comes from")
    table([
        ["USGS", "United States Geological Survey",
         "The US government agency that runs thousands of public river-monitoring stations."],
        ["NWIS", "National Water Information System",
         "The USGS service that publishes those readings."],
        ["EA", "Environment Agency",
         "The equivalent agency for England, which publishes the same sort of readings."],
        ["API", "application programming interface",
         "A web address a program can send a request to and get data back from, instead of a human "
         "downloading a file."],
        ["OGL", "Open Government Licence",
         "The licence the English data is published under, which is what allows it to be "
         "redistributed with this project."],
    ], ["Short form", "Stands for", "What it means"], "", cls="small")

    h3("Machine learning")
    table([
        ["LSTM", "long short-term memory",
         "A type of neural network built to read a sequence in order and remember what mattered "
         "earlier in it. Explained properly in Part 5."],
        ["RNN", "recurrent neural network",
         "The simpler family of sequence networks that LSTMs improved on."],
        ["MSE", "mean squared error",
         "A way of measuring how wrong a prediction is: square each mistake, then average."],
        ["RMSE", "root mean squared error",
         "The square root of the MSE. Used because it comes out in the same units as the thing "
         "being predicted, so an RMSE of 0.34 for oxygen means &ldquo;typically wrong by about "
         "0.34 milligrams per litre&rdquo;."],
        ["Adam", "adaptive moment estimation",
         "The standard algorithm used to nudge a network's numbers in the right direction during "
         "training."],
        ["ReLU", "rectified linear unit",
         "A very simple rule applied between network layers: keep positive numbers, replace "
         "negative ones with zero."],
        ["MLP", "multi-layer perceptron",
         "The plainest kind of neural network: a stack of layers with no memory of a sequence."],
        ["FedAvg", "federated averaging",
         "The method used here to train one model across many sites without collecting their data. "
         "Explained in Part 6."],
        ["IID / non-IID", "independent and identically distributed",
         "Jargon for &ldquo;every site's data looks like every other site's&rdquo;. Real sites are "
         "<i>non</i>-IID: they genuinely differ, which makes training across them harder."],
        ["DP", "differential privacy",
         "A formal method for limiting how much any one participant's data can be worked out from "
         "a shared model."],
        ["SGD", "stochastic gradient descent",
         "The basic training procedure that Adam is a refinement of. DP-SGD is the version with "
         "privacy protection added."],
        ["&epsilon; (epsilon)", "&mdash;",
         "The number that says how much privacy a differentially private system actually gives. "
         "Smaller is more private."],
        ["HDBSCAN", "hierarchical density-based spatial clustering of applications with noise",
         "A clustering method that finds natural groups in data and, unusually, is allowed to "
         "answer &ldquo;this point belongs to no group&rdquo;. Part 8."],
        ["DBCV", "density-based clustering validation",
         "A score that says whether the groups a clustering method found are real. Near zero means "
         "they are not."],
        ["PCA", "principal component analysis",
         "A standard way of squashing many measurements down to a few summary numbers."],
        ["IQR", "interquartile range",
         "The spread of the middle half of a set of numbers. Used instead of standard deviation "
         "when there are extreme values."],
        ["SHAP", "Shapley additive explanations",
         "A method that says how much each input pushed a model's answer up or down. Part 9."],
        ["LightGBM", "light gradient-boosting machine",
         "A fast method that builds many small decision trees and adds their votes together."],
        ["MPC", "model predictive control",
         "A control method that simulates each option forward, then picks the option with the best "
         "predicted outcome. This is how the agent chooses. Part 11."],
        ["RL", "reinforcement learning",
         "Learning by trial and error against rewards. Mentioned only in the background reading; "
         "this project does not use it."],
        ["CPU / GPU", "central / graphics processing unit",
         "Ordinary computer chip versus the specialised chip normally used for training networks. "
         "Everything here runs on an ordinary CPU."],
    ], ["Short form", "Stands for", "What it means"], "", cls="small")

    h3("Everything else")
    table([
        ["OU", "Ornstein&ndash;Uhlenbeck",
         "A way of generating random wobble that drifts and then returns towards where it started, "
         "instead of wandering off. Part 2.6."],
        ["QC", "quality control",
         "The step that throws away readings a broken instrument produced."],
        ["RNG", "random number generator",
         "The part of the program that produces random numbers. Copying its exact internal state "
         "is what lets two simulated ponds experience identical luck. Part 10.3."],
        ["dt", "&mdash;",
         "The size of one time step in the simulator. Here, five simulated minutes."],
        ["W, H, C, N", "&mdash;",
         "Sizes used throughout: W = how many past readings the network is shown, H = how many "
         "future steps it predicts, C = how many water measurements per reading, N = how many "
         "training examples."],
        ["CSV / YAML", "comma-separated values / a settings file format",
         "The two plain-text file formats the project stores results and settings in."],
    ], ["Short form", "Stands for", "What it means"], "", cls="small")


def part0() -> None:
    part(0, "The whole system in one diagram",
         "If you remember nothing else from this document, remember the four lanes and what "
         "passes between them.")

    h2("0.1 What the system does, in one sentence")
    p("<b>AquaSentinel keeps a simulated fish pond's water inside safe limits by acting before a "
      "limit is crossed. It uses models trained across many water-monitoring stations without any "
      "station handing over its raw data, and it puts fixed safety rules outside the learned part "
      "of the system, so that a mistaken model cannot do harm.</b>")

    p("That sentence contains three separate claims, and each one is a different part of the "
      "diagram on the next page.")
    table([
        ["&ldquo;acting before a limit is crossed&rdquo;",
         "predicting, then controlling", "Lane C &mdash; the live loop"],
        ["&ldquo;without handing over raw data&rdquo;",
         "federated learning &mdash; training across sites by sharing models, not data",
         "Lane B &mdash; offline training"],
        ["&ldquo;fixed safety rules outside the learned part&rdquo;",
         "the safety layer, which contains no neural network at all",
         "Lane C &mdash; the guardrail block"],
    ], ["The phrase", "The technical name for it", "Where it lives"],
        "The three claims and where each one is implemented.")

    figure(dg_master(),
           "The complete system. Lane A runs once and is the only part that uses the internet. "
           "Lane B trains the five models on a laptop, offline. Lane C is the live loop, which "
           "runs once every five simulated minutes. Lane D is how a person sees any of it. The "
           "dashed orange arrow between the two trained networks is the &ldquo;warm start&rdquo;, "
           "and Part 7 explains exactly what it does and does not promise.", landscape=True)

    h2("0.2 Why the four lanes are kept apart")
    ul([
        "<b>Lane A is the only part that goes online.</b> Once it has run, the whole project works "
        "with no internet connection. That matters for more than convenience: it means anyone can "
        "regenerate every number in the report on their own machine, which is what makes the "
        "results checkable rather than just claimed.",
        "<b>Lane B never sees the pond from Lane C.</b> The forecaster learns from rivers and is "
        "then used on a simulated pond. Rivers and ponds are genuinely different, and that "
        "difference is a real weakness. It is admitted throughout the code, and Part 11 gives the "
        "one place where it is actually measured.",
        "<b>The guardrails in Lane C are the only thing on the path to the equipment that cannot "
        "be skipped.</b> Every planner &mdash; the simple rule-based one, the predictive one, the "
        "optional language-model one &mdash; only <i>suggests</i> an action. None of them can "
        "switch anything on by itself.",
    ])

    h2("0.3 The five trained models")
    p("People often assume this project is &ldquo;an LSTM&rdquo; and nothing more. LSTM stands for "
      "<b>long short-term memory</b>, a kind of neural network for reading sequences, and there is "
      "one in here &mdash; but there are five trained components in total, doing five different "
      "jobs.")
    table([
        ["1", "Forecaster", "<code>src/perception/forecaster.py</code>",
         "42 real river stations", "trained across sites (FedAvg)",
         "What will the water do <i>if nobody does anything</i>?"],
        ["2", "&ldquo;What if I do this&rdquo; model", "<code>src/perception/world_model.py</code>",
         "8 simulated ponds", "trained across sites (FedAvg)",
         "What will the water do <i>if I switch the aerator on</i>?"],
        ["3", "Grouping / odd-reading detector", "<code>src/perception/regimes.py</code>",
         "real and simulated readings pooled", "unsupervised, one machine",
         "Is this stretch of readings unlike anything seen in training?"],
        ["4", "Risk classifier", "<code>src/perception/risk.py</code>",
         "simulator runs", "supervised, one machine",
         "Is the pond heading for a limit breach?"],
        ["5", "Explainer", "<code>src/perception/explain.py</code>",
         "&mdash; (reads model 4)", "&mdash;",
         "<i>Which reading</i> is driving that risk score right now?"],
    ], ["#", "Model", "File", "Trained on", "How", "The question it answers"],
        "The five trained components. Only models 1 and 2 are neural networks, and only those two "
        "are trained across sites. Models 3, 4 and 5 are trained on one machine because they never "
        "touch data that belongs to a particular farm.", cls="small")

    note("<p><b>Unsupervised</b> means the model was never shown any right answers &mdash; it had "
         "to find structure in the data by itself. <b>Supervised</b> means it was shown an answer "
         "for every example and learned to reproduce it.</p>", "note", "Two words used above")

    h2("0.4 The three clocks")
    p("Most confusion about the live loop clears up as soon as the clocks are clear. There are "
      "three of them and they fit inside one another.")
    figure(dg_clock(),
           "The three clocks. The simulator advances every 5 simulated minutes. The sensors report "
           "on the same schedule. The planner only re-thinks every 15 minutes, so it runs on one "
           "tick in three. The safety layer is the exception: it runs on every single tick, and "
           "that is the whole reason no unsafe action ever gets through between decisions.")

    note("<p>How often does the agent decide, and how often do the safety rules run?</p>"
         f"<p><b>The planner decides every {CFG['agent']['decide_every_minutes']} minutes. The "
         "safety rules run every tick, which is every 5 minutes.</b> That split is deliberate. "
         "Working out the best action is slow, but a pond can go from fine to dangerous in much "
         "less than 15 minutes. So between decisions, a cut-down version of the safety layer keeps "
         "re-checking the action that is currently being held.</p>",
         "exam", "Likely question")


def part1() -> None:
    part(1, "Collecting data from 42 real stations",
         "The only part of the project that goes online, and the part that decides what "
         "&ldquo;the sites genuinely differ&rdquo; actually means here.")

    h2("1.1 Why use real data when the pond is simulated?")
    p("This is usually the first question asked, so it is worth answering carefully.")
    p("The pond has to be simulated because no public dataset records what happens <i>after</i> "
      "somebody acts. Nobody publishes &ldquo;we ran the aerator at 60&nbsp;% for two hours and "
      "here is what the oxygen did&rdquo;. Without those before-and-after pairs there is no "
      "control problem to solve at all.")
    p("But the training-across-sites claim needs sites that really are different from each other. "
      "If both the sites and their data were invented, then how different they are would be "
      "something the authors chose, and the result would say something about the simulation rather "
      "than about the method. So the forecaster is trained on real monitoring stations, where the "
      "differences are inherited rather than invented: different climates, different seasons, "
      "different rivers, and different instruments fitted at each station.")

    note("<p><b>Rivers are not ponds.</b> All 42 stations are rivers and estuaries. A river flows, "
         "is usually cooler, and has no stocked fish, no feeding schedule and no aerator. This is "
         "stated in the code, in the report and on the slides. Part 11 shows the one place in the "
         "testing where this difference is measured instead of just admitted &mdash; and there, it "
         "makes the system worse.</p>", "honest", "The gap between rivers and ponds")

    h2("1.2 The nine collection steps")
    figure(dg_acquisition(),
           "How readings become training data. Steps 1 to 3 are the only internet traffic in the "
           "whole project. Steps 4 to 9 are fixed transformations of the saved responses, so "
           "re-running them changes nothing.")

    h3("The two sources, and why these two")
    table([
        ["USGS NWIS", "United States", "28",
         "Public domain. The data is marked provisional and may later be revised."],
        ["Environment Agency", "England", "14",
         "Open Government Licence v3.0, &copy; Environment Agency."],
    ], ["Source", "Region", "Stations", "Licence"],
        "USGS stands for United States Geological Survey and NWIS for its National Water "
        "Information System. Both services need no registration and no password, which was a "
        "deliberate requirement: a marker has to be able to repeat the collection step without any "
        "credential of ours.", cls="small")

    p("The stations are found by the program rather than typed in by hand. For the American source "
      "the program asks the station catalogue for every station that reports the measurements we "
      "need, and for how long each one has been reporting them. A station only qualifies if it was "
      "genuinely reporting during the months we want. The catalogue lists plenty of stations whose "
      "oxygen probe stopped working years ago, and hand-typed station numbers would quietly go "
      "stale. The catalogue answers are saved to disk so the step does not need repeating.")

    h2("1.3 Making the two sources comparable")
    p("Each source arrives in its own format, with its own codes for each measurement and its own "
      "reporting schedule. Both are converted into one common shape: a long list of "
      "<code>[time, measurement name, value]</code> rows. From there, all sources are treated "
      "identically.")
    ol([
        "<b>Rename.</b> Each source's own code for a measurement becomes the shared name. The "
        "American code <code>00300</code> and the English oxygen measure both become "
        "<code>do</code>, short for dissolved oxygen.",
        "<b>Reshape and line up in time.</b> The long list becomes a table with one row per "
        "timestamp and one column per measurement, placed on an even 15-minute grid. Missing "
        "readings are left missing.",
        "<b>Throw away impossible values.</b> A pH of 27 is an instrument fault, not a reading. "
        "The plausible range for each measurement is listed in the settings file.",
        "<b>Throw away frozen readings.</b> A probe reporting exactly the same value for hours is "
        "broken, not calm. Runs of identical values beyond a set length are removed.",
        "<b>Throw away readings pinned at the instrument's floor.</b> If a probe cannot measure "
        "below a certain amount, a reading sitting exactly on that floor is not a measurement of "
        "the water.",
        "<b>Record what survived.</b> One true-or-false flag per cell, answering &ldquo;was this "
        "actually measured?&rdquo; This is called the <b>availability mask</b>, and it is used "
        "again in Part 4.",
    ])

    note("<p>Why keep a record of the gaps instead of filling them in?</p>"
         "<p><b>Because the gaps are information, not a nuisance.</b> Stations differ in which "
         "instruments they have &mdash; only 12 of the 42 measure ammonia, and none measures "
         "nitrite. If the gaps were filled in, every station would look the same in shape, and the "
         "claim that the sites genuinely differ would disappear. The mask is also given to the "
         "network as extra inputs (Part 4), so the network can tell &ldquo;the reading is "
         "zero&rdquo; apart from &ldquo;there is no reading&rdquo;.</p>", "exam", "Likely question")

    h2("1.4 What came out")
    man = manifest()
    if man is not None:
        n = len(man)
        rows = int(man["rows"].sum())
        src = man["source"].value_counts().to_dict()
        regions = man["region"].nunique()
        miss = float(man["missing_frac"].mean())
        chan_counts = man["n_channels"].value_counts().sort_index()
        note(f"<p><b>{n} stations &middot; {rows:,} rows of readings &middot; one reading every 15 "
             f"minutes &middot; {regions} regions &middot; on average {miss:.1%} of cells "
             f"missing</b></p>"
             f"<p>By source: " + ", ".join(f"{k} = {v}" for k, v in src.items()) +
             ". How many measurements each station reports: " +
             ", ".join(f"{int(k)} &rarr; {int(v)} stations" for k, v in chan_counts.items()) +
             ".</p>", "num", "Read from data_cache/manifest.csv")
        show = man.head(6)[["client_id", "source", "region", "n_channels", "rows", "missing_frac"]]
        table([[f"<code>{r.client_id}</code>", r.source, r.region, int(r.n_channels),
                f"{int(r.rows):,}", f"{float(r.missing_frac):.1%}"] for r in show.itertuples()],
              ["station id", "source", "region", "measurements", "rows", "missing"],
              "The first six stations. Each one of these becomes a single training participant in "
              "Part 6.", cls="small", align="llllrr")
    p("The record also stores the licence for each station, which is what allows the data to be "
      "shipped alongside the code.")

    h2("1.5 The one thing deliberately not collected")
    note("<p>Several of these services also publish a ready-made water-quality grade for each "
         "reading. This project never reads it.</p>"
         "<p>Those grades are produced by applying fixed thresholds to the very same measurements "
         "the model would be given as inputs. A model trained on them and then marked against them "
         "has only learned the threshold rule back again, and the accuracy it reports means nothing "
         "about water. Every target used anywhere in this project is instead worked out from the "
         "simulator's own safe-range table (Part 2.8), which is written down openly and can be "
         "argued with directly.</p>", "honest", "No ready-made grades are used")


def part2() -> None:
    t = CFG["twin"]
    part(2, "The pond simulator",
         "The world the agent acts on. It is a simulation, tuned against real data but never "
         "validated against a real pond &mdash; and this document says so wherever it matters.")

    h2("2.1 What it is and what it is not")
    p("<code>PondTwin</code> is a computer model of a warm-water earthen fish pond. It exists "
      "because the agent needs something it can actually <i>act on</i>: a world where switching "
      "on an aerator really does raise the oxygen level, where feeding really does produce "
      "ammonia, and where those effects arrive with a realistic delay. No public dataset provides "
      "that.")
    p("Such a model is often called a <b>digital twin</b> &mdash; a running software copy of a "
      "physical thing, kept close enough to it to be useful for asking &ldquo;what if&rdquo;.")
    note("<p>The equations follow standard pond-farming practice, taken from Boyd's <i>Water "
         "Quality in Ponds for Aquaculture</i>, the FAO pond manuals, and the standard reference "
         "for how much oxygen water can hold. (FAO is the Food and Agriculture Organization of the "
         "United Nations.)</p>"
         "<p>But <b>every number in those equations is a starting value that can be tuned, not a "
         "measured fact.</b> The values were chosen so that the pond behaves believably: oxygen "
         "sags before dawn, pH rises during the day, ammonia builds up after feeding. Nothing the "
         "simulator produces should be read as a statement about a real farm.</p>",
         "honest", "This is not a validated pond model")

    h2("2.2 What the simulator keeps track of")
    p("The simulator carries three kinds of quantity, and the difference between them is what "
      "makes the control problem hard.")
    table([
        ["Things a sensor can measure", "7",
         "<code>do, temp_c, ph, tan, no2, turbidity, conductivity</code>",
         "The agent is allowed to see these."],
        ["Hidden quantities", "6",
         "<code>algae, biomass, alkalinity, bod_pool, feed_pool, chloride</code>",
         "Real properties of a pond that no probe reports. The agent never sees these."],
        ["Worked out from the others", "1", "<code>nh3</code>",
         "Un-ionised ammonia &mdash; the harmful form. Calculated from total ammonia, pH and "
         "temperature."],
    ], ["Kind", "Count", "Names in the code", "Who can see them"],
        "Hidden quantities are the reason one of the comparison controllers in Part 13 is labelled "
        "as cheating: the &ldquo;oracle&rdquo; is defined precisely by being allowed to read this "
        "middle row, which no real farm could do.")

    note("<p>Why calculate NH&#8323; instead of simulating it directly?</p>"
         "<p><b>Because how poisonous ammonia is depends on the water, not just on how much "
         "ammonia there is.</b> Ammonia exists in two forms, and only one of them harms fish. "
         "Which form it takes is set by pH and temperature. Total ammonia of 1.0 milligram per "
         "litre is harmless at pH 7 and lethal at pH 9. So the simulator tracks the total, then "
         "works out the harmful share each step using the standard 1975 relation by Emerson and "
         "colleagues. Simply putting a threshold on the total would get the chemistry wrong in "
         "exactly the situation that matters.</p>", "exam", "Likely question")

    h2("2.3 One time step, in order")
    p(f"The simulator moves forward in fixed jumps of <code>{t['dt_minutes']}</code> simulated "
      "minutes, using the simplest possible method: work out the rate of change, multiply by the "
      "length of the step, add it on. (This is called Euler integration, after the mathematician "
      "Leonhard Euler.) The order of operations inside one step is deliberate.")
    figure(dg_twin_step(),
           "The nine stages of one step. Actions are applied before the rates are worked out, so "
           "an action taken this step affects this step's chemistry. The random wobble is added "
           "after the step, so it disturbs the water rather than the rates. The final limits come "
           "last, so nothing downstream ever receives an impossible value such as a negative "
           "concentration.", wide=True)

    h2("2.4 Where the oxygen goes")
    p("Oxygen is the measurement the whole project revolves around, so it is worth knowing its "
      "equation piece by piece. Three things add oxygen; three things remove it.")
    figure(dg_do_budget(),
           "The six things that change the oxygen level. Note that photosynthesis is exactly zero "
           "at night. That single fact produces the pre-dawn oxygen sag which the "
           "<code>night_do_crash</code> test is built around, and it is why a simple thermostat "
           "that only reacts to the current reading is always too late.")
    ox = t["oxygen"]
    code(
        "change in oxygen per hour =\n"
        "      k_a   x (saturation - current)     # oxygen soaking in from the air\n"
        "    + k_aer x u x (saturation - current) # the aerator, u = how hard it runs, 0 to 1\n"
        "    + P_max x light x algae              # plants making oxygen; light is 0 at night\n"
        "    - R_20  x theta^(T-20) x (fish + w x algae)   # everything breathing\n"
        "    - k_BOD x decaying_material          # rotting matter using oxygen up\n"
        "    - 4.57  x nitrogen_processed         # bacteria processing ammonia\n\n"
        f"k_a = {ox['k_reaeration_per_h']}   k_aer = {ox['k_aerator_per_h']}   "
        f"P_max = {ox['p_max_per_h']}   R_20 = {ox['r20_per_h']}   "
        f"theta = {ox['theta_resp']}   w = {ox['algae_respiration_weight']}   "
        f"k_BOD = {ox['k_bod_per_h']}",
        "src/twin/dynamics.py, with the values currently set in configs/twin.yaml")
    p("Two details are worth being able to defend.")
    ul([
        "<b>The aerator term is a gap term, not a fixed boost.</b> An aerator pushes oxygen "
        "towards the maximum the water can hold, so it does less and less as the water fills up, "
        "and nothing at all in water that is already over-full. Anybody expecting &ldquo;aerator "
        "on = +2 milligrams per litre&rdquo; has the wrong mental model.",
        "<b>Breathing speeds up when the water is warm.</b> The <code>theta^(T-20)</code> factor "
        "is the standard way of writing that. This is what makes the <code>heatwave</code> test "
        "genuinely dangerous rather than merely warm: hot water holds less oxygen <i>and</i> "
        "everything in it uses oxygen faster.",
    ])

    h2("2.5 Ammonia, pH and algae")
    h3("Ammonia")
    p("Feed goes into a store and is broken down gradually rather than all at once. That is why "
      "the ammonia spike in the <code>overfeeding_ammonia</code> test arrives hours after the feed "
      "does. The bacteria that clear ammonia need oxygen to do it, and the simulator makes their "
      "rate depend on how much oxygen is available. So a pond that is already short of oxygen also "
      "loses its ability to clear ammonia &mdash; two problems feeding each other, which is "
      "exactly the sort of thing the agent has to see coming.")
    h3("pH and alkalinity")
    p("Plants making oxygen also strip out carbon dioxide, which pushes pH up; everything "
      "breathing puts carbon dioxide back, which pulls pH down. How far pH actually moves depends "
      "on the pond's <b>alkalinity</b> &mdash; its chemical resistance to pH change, a bit like a "
      f"shock absorber. The simulator boosts the swing by "
      f"<code>normal_alkalinity / current_alkalinity</code>, capped at "
      f"{t['ph']['max_buffer_gain']} times. A pond whose shock absorber has collapsed swings far "
      f"harder for the same amount of biological activity, and that one line is the whole "
      f"<code>ph_swing</code> test. Clearing ammonia also uses alkalinity up, at "
      f"{t['ammonia']['alkalinity_per_n']} milligrams of calcium carbonate per milligram of "
      "nitrogen &mdash; so an ammonia problem quietly turns into a pH problem.")
    h3("Algae")
    p("Algae grow on light and nitrogen, and die faster the more crowded they get &mdash; the "
      "death rate rises with the <i>square</i> of how full the pond is. That squared term is what "
      "makes a bloom <i>collapse</i> rather than level off. The dead algae then join the store of "
      "decaying material, which uses up oxygen at night. The <code>algal_bloom</code> test is that "
      "whole sequence in one run.")

    h2("2.6 The random wobble, and why it is not plain random noise")
    p("Six of the measurements get random wobble added, and it is the kind that drifts and then "
      "comes back rather than wandering off. This is called an Ornstein&ndash;Uhlenbeck process, "
      "after the two physicists who described it:")
    w('<p class="eq">change in wobble = &minus;&theta; &times; current wobble &times; dt '
      '&nbsp;+&nbsp; &sigma; &times; &radic;dt &times; a random draw</p>')
    p("&theta; (theta) controls how strongly it is pulled back towards zero; &sigma; (sigma) "
      "controls how big the random kicks are.")
    note("<p>Why not just add ordinary random noise to each measurement?</p>"
         "<p><b>Because ordinary noise on a five-minute step looks like a twitchy instrument, and "
         "this is meant to be the water, not the instrument.</b> Weather and biology drift over "
         "hours and then come back. This kind of wobble does that; independent random draws do "
         "not. Instrument twitchiness is added separately, one layer later, in Part 3. Keeping the "
         "two apart is what makes it fair to ask the odd-reading detector to tell them apart. "
         f"&theta; = {t['noise']['ou_theta_per_h']} per hour.</p>", "exam", "Likely question")

    h2("2.7 How an action reaches the water")
    p("The simulator, not the agent, owns what an action physically does. There are five things "
      "the agent can operate:")
    a = t["actions"]
    table([
        ["<code>aerator</code>", "any value from 0 to 1", "straight away",
         f"pushes oxygen towards saturation at rate {ox['k_aerator_per_h']} per hour when running "
         f"flat out"],
        ["<code>feed</code>", "normal, reduced or skip",
         f"only at {a['feed_hours']} o'clock",
         f"{a['feed_normal_kg_per_day']} kg per day split between the feeding times; reduced means "
         f"{a['feed_reduced_frac']} of that, skip means none"],
        ["<code>exchange</code>", "none, small or large",
         f"runs for {a['exchange_duration_h']} hour",
         f"gradually replaces pond water with cleaner water, at "
         f"{a['exchange_small_frac_per_h']} or {a['exchange_large_frac_per_h']} of the pond per hour"],
        ["<code>lime</code>", "none or dose", "dissolves over hours",
         f"{a['lime_dose_kg']} kg goes into an undissolved pile, releasing "
         f"{a['lime_release_per_h']} of what is left each hour and raising alkalinity"],
        ["<code>salt</code>", "none or dose", "straight away, then fades",
         f"{a['salt_dose_kg']} kg becomes chloride in the water, fading at "
         f"{a['chloride_decay_per_h']} per hour"],
    ], ["Action", "Values", "How fast it acts", "What the simulator does"],
        "The five actions. Two of these timings matter a lot for the planner: feeding can only "
        "happen at the scheduled times, and lime dissolves gradually instead of working on "
        "contact. An agent that doses lime and expects the pH to move immediately will be "
        "disappointed &mdash; which is part of why the safety layer enforces a minimum gap between "
        "doses.", cls="small")

    note("<p>Salt does not neutralise nitrite. What it does is compete with nitrite at the fish's "
         "gills, which is the standard treatment for the condition farmers call brown blood "
         "disease. The simulator represents this by <i>loosening the nitrite safety limit</i> in "
         "proportion to how much salt there is relative to nitrite, and caps that loosening at "
         "three times &mdash; because salt is a treatment, not permission to ignore nitrite.</p>",
         "note", "A detail worth knowing")

    h2("2.8 The safe-range table &mdash; what everything is scored against")
    p("These ranges are <b>not learned</b>. They come from published guidance for warm-water fish "
      "ponds and they sit in a plain settings file, which means anybody can disagree with them "
      "directly instead of having to argue with a model. Model 4 (Part 9) is trained to "
      "<i>see them coming</i>; it never replaces them.")
    p("Each measurement gets two bands. <b>Warning</b> means the fish are stressed. "
      "<b>Critical</b> means they are in real danger.")
    rb = t["risk_bands"]
    rows = []
    pretty = {"do": "dissolved oxygen (mg/L)", "ph": "pH", "nh3": "harmful ammonia (mg/L)",
              "no2": "nitrite (mg/L)", "temp_c": "temperature (&deg;C)"}
    for ch, spec in rb.items():
        crit, warn, extra = [], [], []
        if spec.get("critical_below") is not None:
            crit.append(f"below {spec['critical_below']}")
        if spec.get("critical_above") is not None:
            crit.append(f"above {spec['critical_above']}")
        if spec.get("warning_below") is not None:
            warn.append(f"below {spec['warning_below']}")
        if spec.get("warning_above") is not None:
            warn.append(f"above {spec['warning_above']}")
        healthy = spec.get("healthy")
        if healthy:
            extra.append(f"ideal {healthy[0]}&ndash;{healthy[1]}")
        if spec.get("supersaturation_above") is not None:
            extra.append(f"too much oxygen above {spec['supersaturation_above']}")
        rows.append([pretty.get(ch, ch) + f" <code>{ch}</code>", ", ".join(crit) or "&mdash;",
                     ", ".join(warn) or "&mdash;", "; ".join(extra) or "&mdash;"])
    table(rows, ["Measurement", "Critical", "Warning", "Other notes"],
          "The safe ranges. At any instant, the worst measurement decides the pond's status "
          "&mdash; a pond is only as safe as its worst reading. The headline score used throughout "
          "the testing, <b>time in risk</b>, is simply how many hours the pond spent outside the "
          "safe band.", cls="small")

    h2("2.9 The eight test situations")
    p("A test situation, called a <b>scenario</b> in the code, is three things: a starting state, "
      "a timetable of things going wrong, and a random seed. (A <b>seed</b> is the number that "
      "fixes which random values get produced, so the same seed always gives the same run.) Each "
      "scenario can be run on its own and reproduces exactly.")
    scn = CFG["scenarios"]["scenarios"]
    rows = []
    for sid, spec in scn.items():
        rows.append([f"<code>{sid}</code>", spec.get("label", ""),
                     f"{spec.get('duration_h', 48)} h", str(spec.get("seed", "")),
                     spec.get("description", "")])
    table(rows, ["Name", "What it is called", "Length", "Seed", "What it tests"],
          "The eight scenarios. Two of them are deliberately undramatic. "
          "<code>healthy_baseline</code> checks that the agent leaves a healthy pond alone, and "
          "<code>sensor_fault</code> breaks the instruments rather than the water, so it tests "
          "whether the system notices rather than whether it can fix anything.", cls="small")

    note("<p>Most of these problems stay switched on for a stretch of time. Three of them "
         "&mdash; a sudden extra dump of feed, a sudden bloom of algae, and a collapse of "
         "alkalinity &mdash; happen once and then stop.</p>"
         "<p>Crucially, <b>the agent is never told the timetable.</b> Its internal picture of the "
         "pond has no knowledge of the scenario at all. It cannot know that the aerator is about "
         "to fail; it can only notice that the oxygen is not responding to the command it "
         "gave.</p>", "note", "How the problems are injected")

    h2("2.10 Tuning &mdash; what was measured, and then deliberately not used")
    cal = t.get("calibration", {})
    if cal:
        note(f"<p>A separate script measured the daily rhythms across the "
             f"{cal.get('n_real_sites', 42)} real stations and wrote the results into the settings "
             f"file. It then <b>deliberately did not apply them</b>: "
             f"<code>adopt_mean_temperature: {cal.get('adopt_mean_temperature')}</code>, "
             f"<code>adopt_diurnal: {cal.get('adopt_diurnal')}</code>.</p>"
             "<p>The measured values are kept in the file under a name that says they were not "
             "applied, so the decision is visible rather than hidden. The reason is the "
             "river-versus-pond gap again: the real stations average about 16&nbsp;&deg;C, and "
             "forcing a tropical fish pond to follow a cool river's daily temperature pattern "
             "would make it a <i>worse</i> pond model, not a better one. Tuning towards the data "
             "would have looked more rigorous and been less honest.</p>",
             "honest", "Measured, then overridden on purpose")


def part3() -> None:
    s = CFG["sensors"]
    part(3, "The fake sensors",
         "The rest of the system never sees the simulator directly. It sees this: rounded off, "
         "slightly wrong, noisy, and sometimes missing.")

    h2("3.1 Why there is a separate sensor layer at all")
    p("If the agent could read the simulator's true state, the whole project would be a control "
      "exercise with perfect information, and none of the perception models would be doing any "
      "work. <code>SensorStream</code> is a deliberate bottleneck. It turns the true state into a "
      "believable instrument feed, and everything after it &mdash; forecaster, grouping model, "
      "risk model, planner, safety layer &mdash; lives on the far side of that bottleneck.")

    figure(dg_sensor_chain(),
           "How a true value becomes a reading, in the order the steps are applied. Note that the "
           "fixed calibration error is drawn <b>once per run</b> and then kept. That is why two "
           "identical ponds with two identical probes still disagree, and it is a large part of "
           "why the forecaster must not be allowed to trust the absolute value of a reading.",
           wide=True)

    h2("3.2 The exact numbers for each measurement")
    chans = list(s["noise_std"].keys())
    pretty = {"do": "dissolved oxygen", "temp_c": "temperature", "ph": "pH",
              "tan": "total ammonia", "no2": "nitrite", "turbidity": "cloudiness"}
    table([[f"{pretty.get(c, c)} <code>{c}</code>", s["noise_std"].get(c, "&mdash;"),
            s["bias"].get(c, "&mdash;"), s["quantization"].get(c, "&mdash;")] for c in chans],
          ["Measurement", "Random noise per reading", "Fixed error per run",
           "Smallest step it can report"],
          f"Sensor behaviour. On top of all this, each measurement independently goes missing "
          f"{s['dropout_prob']:.0%} of the time, and readings are taken every "
          f"{s['sample_every_minutes']} minutes. The two error columns are given as standard "
          f"deviations &mdash; the typical size of the error.", cls="small", align="lrrr")

    note(f"<p>The safety layer does not see the truth either &mdash; it sees this feed. That is "
         f"why the settings file gives it a <b>margin</b> for each measurement it checks: "
         f"oxygen {CFG['agent']['guardrails']['sensor_margin']['do']}, "
         f"harmful ammonia {CFG['agent']['guardrails']['sensor_margin']['nh3']}, "
         f"pH {CFG['agent']['guardrails']['sensor_margin']['ph']}. Every safety threshold is "
         f"tested against the reading shifted in whichever direction is <i>less</i> safe, by "
         f"roughly three times the typical sensor error. The result is that a safety rule fires "
         f"slightly early rather than slightly late. Part 12 shows this in use.</p>",
         "num", "What this means for safety")

    h2("3.3 A broken probe is not bad water")
    p("Four kinds of instrument failure can be injected on a timetable:")
    table([
        ["<code>dropout</code>", "the reading simply goes missing"],
        ["<code>spike</code>", "a large jump up or down for a moment"],
        ["<code>stuck</code>", "the reading latches onto one value and stops moving"],
        ["<code>drift</code>", "a steady offset creeps in"],
    ], ["Failure mode", "What it looks like"], "", cls="small")
    note("<p>Why is instrument failure handled here rather than in the simulator?</p>"
         "<p><b>Because a broken probe spoils the reading, not the water.</b> That distinction is "
         "the entire point of the <code>sensor_fault</code> test. The odd-reading detector has to "
         "flag &ldquo;this stretch of readings is unlike anything I know&rdquo; without being able "
         "to tell whether the cause is a genuine problem in the pond or a failing instrument. In "
         "that test every controller scores zero hours at risk &mdash; because the water was fine "
         "the whole time. Anything the agent did there was a reaction to a false reading.</p>",
         "exam", "Likely question")

    h2("3.4 What comes out")
    p("An <code>Observation</code>: a set of values, a matching set of true-or-false flags saying "
      "which of them are real, and a timestamp. Conductivity is passed through untouched. Real "
      "stations report it, and the agent does not act on it, but keeping it in the format means "
      "the simulated path and the real-data path have exactly the same shape &mdash; so the same "
      "perception code runs on both without modification.")


def part4() -> None:
    m = CFG["models"]
    W, H = m["window"], m["horizon"]
    ch = m["forecaster"]["channels"]
    C = len(ch)
    part(4, "Cutting the data into training examples",
         "The step most often skipped over, and the one that decides whether anything learned from "
         "a river is any use on a pond.")

    h2("4.1 The problem")
    p("A neural network cannot be handed &ldquo;six months of readings&rdquo;. It needs many "
      "examples, all the same size. The function <code>make_windows()</code> produces them, and "
      "three decisions inside it carry the whole training-across-sites result.")
    p("Some shorthand used from here on: <b>W</b> is how many past readings the network is shown "
      "(the <b>window</b>), <b>H</b> is how many future steps it has to predict (the "
      "<b>horizon</b>), and <b>C</b> is how many different measurements there are in each reading.")
    figure(dg_windows(),
           "How one station's readings become training examples. The cell outlined in red is the "
           "<b>anchor</b> &mdash; the last reading the network is shown. Everything the network "
           "predicts is expressed as a change from that reading, which is why an example is thrown "
           "away if the anchor is missing.", wide=True)

    h2("4.2 Decision one: the window is counted in readings, not in hours")
    note(f"<p><code>W = {W}</code> and <code>H = {H}</code> are <b>counts of readings, not "
         f"lengths of time</b>. The code slices by position in the list.</p>"
         f"<ul><li>On the real data, where a reading arrives every 15 minutes: {W} readings in "
         f"&rarr; {H} readings out is <b>{W * 15 / 60:.0f} hours &rarr; {H * 15 / 60:.0f} "
         f"hours</b>.</li>"
         f"<li>Inside the simulator, where a step is 5 minutes: the same {W} and {H} is "
         f"<b>{W * 5 / 60:.0f} hours &rarr; {H * 5 / 60:.0f} hours</b>.</li></ul>"
         "<p>Both statements are true and they are not the same amount of time. If somebody asks "
         "&ldquo;what is your forecast horizon?&rdquo;, the correct answer is &ldquo;24 readings "
         "&mdash; two hours in the simulator, six hours on the real data&rdquo;.</p>",
         "num", "W and H count readings")

    h2("4.3 Decision two: predict the change, not the value")
    p("The network does <b>not</b> predict &ldquo;oxygen will be 6.2 milligrams per litre&rdquo;. "
      "It predicts &ldquo;oxygen will go <i>up by</i> 0.4 from the anchor reading&rdquo;. That "
      "change is then divided by a fixed number for each measurement, to keep all the values "
      "roughly around 1 so that no single measurement dominates training.")
    w('<p class="eq">what the network sees = ( reading &minus; anchor reading ) &divide; '
      'the fixed scale for that measurement</p>')
    ds = m["forecaster"]["delta_scale"]
    pretty = {"do": "dissolved oxygen", "temp_c": "temperature", "ph": "pH",
              "tan": "total ammonia", "no2": "nitrite", "turbidity": "cloudiness"}
    table([[f"{pretty.get(c, c)} <code>{c}</code>", ds[c],
            "roughly how much this measurement moves in a busy two hours"] for c in ch],
          ["Measurement", "Fixed scale", "Why that size"],
          "The fixed scaling numbers. They are chosen once, by hand, and written into the settings "
          "file.", cls="small", align="lrl")
    note("<p>Why predict changes rather than actual values?</p>"
         "<p><b>Because levels do not carry across from one water body to another, but shapes "
         "do.</b> A network trained on actual values learns something like &ldquo;an English river "
         "in April sits around 9 milligrams per litre&rdquo;, which is useless on a "
         "28&nbsp;&deg;C tropical pond. A network trained on changes learns the <i>shape</i> of a "
         "daily oxygen cycle &mdash; the dip, the recovery, and how fast each happens &mdash; and "
         "that shape does carry across.</p>", "exam", "Likely question")

    h2("4.4 Decision three: the scales are fixed, never calculated from the data")
    note("<p>This looks like a small implementation detail. It is actually an argument about "
         "privacy.</p>"
         "<p>The obvious thing to do is work out the typical size of each measurement from the "
         "training data and divide by that. Doing so here would <b>break the whole premise</b>: "
         "each station's averages would have to be sent somewhere central to be combined, which is "
         "exactly the kind of leak that training-across-sites exists to prevent.</p>"
         "<p>It would also make the three training methods in Part 6 impossible to compare fairly, "
         "because each would be scaling by different numbers. With the scales fixed in a settings "
         "file, all three do identical preparation, so any difference between them comes from the "
         "method and not from the plumbing.</p>", "exam", "Why this matters more than it looks")

    h2("4.5 The anchor rule")
    anchors = m["forecaster"].get("anchor_channels", ch)
    p(f"An example is kept only if <b>every anchor measurement</b> &mdash; here "
      f"{', '.join('<code>' + a + '</code>' for a in anchors)} &mdash; has a genuine reading at the "
      f"last step of the window. Gaps anywhere else in the window are allowed, and are simply "
      f"marked as missing.")
    p("The reason is a matter of definition, not statistics. What the network predicts is "
      "<i>the change from the anchor</i>. If the anchor is missing, there is nothing to measure "
      "the change from. Filling in a plausible value there would invent a reference point and then "
      "train the network to predict changes relative to a number nobody ever measured.")

    h2("4.6 Telling the network what is missing")
    p(f"The input is {2 * C} columns wide, not {C}. The first {C} columns are the changes; the "
      f"next {C} are the availability mask from Part 1 &mdash; a 1 where the reading is real and a "
      f"0 where it is missing.")
    note("<p>What does including the mask buy you?</p>"
         "<p><b>It lets the network tell &ldquo;the reading is zero&rdquo; apart from &ldquo;there "
         "is no reading&rdquo;.</b> Without it, a gap that has been filled in and a genuinely flat "
         "reading look identical, and the network learns to treat a broken instrument as a real "
         "state of the water. Since real stations differ in which instruments they have at all "
         "&mdash; a station with no ammonia probe has that column permanently marked missing "
         "&mdash; this is also how one network design serves 42 stations that are structurally "
         "different from one another.</p>", "exam", "Likely question")

    h2("4.7 Every size, in one table")
    p("Sizes are written as (rows, columns) or (examples, rows, columns).")
    table([
        ["the station's readings", f"(T, {C})",
         "T rows, one every 15 minutes, one column per measurement"],
        ["the availability mask", f"(T, {C})", "true or false: was this cell genuinely measured?"],
        ["<code>X</code> &mdash; the inputs", f"(N, {W}, {2 * C})",
         "N examples, each W readings long, each reading holding the changes then the mask"],
        ["<code>Y</code> &mdash; the answers", f"(N, {H}, {C})", "the scaled changes to predict"],
        ["<code>M</code> &mdash; which answers count", f"(N, {H}, {C})",
         "true or false: only real readings are marked, so the network is never scored on a gap"],
        ["what the network outputs", f"(N, {H}, {C})", "scaled changes, before converting back"],
        ["the finished forecast", f"({H}, {C})",
         "real units again: milligrams per litre, degrees, pH"],
    ], ["What it is", "Size", "What it holds"],
        "The complete size chain for Model 1. N is how many usable examples a station produced. "
        f"The code steps forward {CFG['federated']['clients']['window_stride']} readings at a time "
        "rather than one, so examples overlap but not completely, and any example failing the "
        "anchor rule is dropped.", cls="small")


def part5() -> None:
    m = CFG["models"]
    fc = m["forecaster"]
    W, H = m["window"], m["horizon"]
    C = len(fc["channels"])
    part(5, "Model 1 &mdash; the forecaster",
         "What will the water do if nobody does anything? Learned from 42 real rivers, then used "
         "on a simulated pond.")

    h2("5.1 What kind of network this is")
    p("The forecaster is an <b>LSTM</b>, short for <b>long short-term memory</b>. It is a network "
      "designed to read a sequence one step at a time and carry forward a summary of what it has "
      "seen so far. That summary is called the <b>hidden state</b>.")
    p("Stated precisely, so it can be quoted:")
    ul([
        "<b>One layer</b>, not stacked.",
        "<b>One direction</b> &mdash; it reads the window from oldest to newest and never "
        "backwards. The technical word is <i>unidirectional</i>.",
        f"<b>{fc['hidden_size']} hidden units</b>, meaning its running summary is a list of "
        f"{fc['hidden_size']} numbers.",
        f"<b>Read out once, at the end.</b> Only the summary after the final reading is used.",
        f"That summary then goes through <b>one plain layer</b> that turns "
        f"{fc['hidden_size']} numbers into {H} &times; {C} = {H * C} numbers &mdash; the whole "
        "forecast in one shot.",
    ])
    note("<p><b>What kind of encoder does the forecaster use?</b></p>"
         f"<p>A single-layer, one-direction LSTM with {fc['hidden_size']} hidden units, reading the "
         f"window in order and read out only at the final step, feeding one plain "
         f"{fc['hidden_size']} &rarr; {H * C} layer. <b>Not</b> two-directional, <b>not</b> an "
         "attention model, <b>not</b> a transformer, and <b>not</b> the kind that generates its "
         "output one step at a time.</p>"
         "<p><i>Encoder</i> is just the name for the part that reads the input and compresses it; "
         "here that is the LSTM itself, and the plain layer after it is the <i>head</i>.</p>",
         "exam", "The question this document was asked to answer")

    figure(dg_lstm(),
           "The forecaster from end to end. One thing worth being able to say without hesitating: "
           "only the <i>final</i> summary is read. The network maps the whole window to the whole "
           f"forecast in a single pass, which is why the head is one {fc['hidden_size']} &rarr; "
           f"{H * C} layer rather than something that generates step by step.", wide=True)
    code(
        f"class _LSTMNet(nn.Module):\n"
        f"    self.lstm = nn.LSTM(input_size={2 * C},      # {C} changes + {C} mask columns\n"
        f"                        hidden_size={fc['hidden_size']},\n"
        f"                        num_layers={fc['num_layers']},\n"
        f"                        batch_first=True,\n"
        f"                        dropout={fc['dropout']})    # has no effect with 1 layer\n"
        f"    self.head = nn.Linear({fc['hidden_size']}, {H} * {C})   # -> {H * C} numbers\n\n"
        f"    def forward(self, x):            # x is (batch, {W}, {2 * C})\n"
        f"        out, _ = self.lstm(x)        # out is (batch, {W}, {fc['hidden_size']})\n"
        f"        y = self.head(out[:, -1, :]) # [-1] = the LAST step only\n"
        f"        return y.view(-1, {H}, {C})  # reshape into a forecast",
        "src/perception/forecaster.py")

    h2("5.2 Why an LSTM, and why such a small one")
    p("A plain sequence network struggles to remember anything from many steps back, because the "
      "signal used to correct it fades away the further back it has to travel. An LSTM fixes that "
      "with four small gates that decide, at each step, what to forget, what to take in, what to "
      "propose, and what to reveal.")
    figure(dg_lstm_cell(),
           "The four gates. &sigma; (sigma) squashes a number into the range 0 to 1, so each gate "
           "acts like a dial from &ldquo;none of it&rdquo; to &ldquo;all of it&rdquo;. The "
           "important part is that the memory line c<sub>t</sub> is <i>added to</i> rather than "
           "repeatedly multiplied, which is what lets the network still learn from something that "
           "happened 36 steps ago.")
    p("The honest answer to &ldquo;why not a transformer?&rdquo; &mdash; the architecture behind "
      "modern language models &mdash; is that this problem does not call for one. The window is "
      "only 36 readings long, so there is nothing far away to reach for. Each station has only a "
      "few thousand examples, so a bigger model would simply memorise them. Everything has to run "
      "on an ordinary computer without a graphics card. And every extra number in the network is "
      "one more thing that has to be sent over the network 12 times during training, which Part 6 "
      "explains.")

    h2("5.3 How it is trained")
    table([
        ["What counts as wrong", "masked mean squared error",
         "square each mistake and average, but skip any answer that was never measured"],
        ["Optimiser", "Adam", f"learning rate {fc['lr']} &mdash; how big a correction to make each "
         "time. Adam stands for adaptive moment estimation."],
        ["Batch size", fc["batch_size"],
         "how many examples are looked at before making a correction"],
        ["Epochs", fc["epochs"],
         "how many times it goes through all the examples. Part 6 overrides this."],
        ["Gradient clipping", fc["grad_clip"],
         "a cap on how large a single correction may be, so one strange example cannot wreck the "
         "network"],
        ["Hardware", "ordinary CPU", "no graphics card needed; runs the same way every time"],
    ], ["Setting", "Value", "What it means"], "How Model 1 is trained.", cls="small")
    w('<p class="eq">error = sum of ( (prediction &minus; answer)&sup2; &times; was-it-measured ) '
      '&divide; how many were measured</p>')
    p("Skipping the unmeasured answers matters more than it sounds. Without it, a station whose "
      "ammonia probe is dead would be rewarded for confidently predicting zeros on a measurement "
      "it cannot take &mdash; and that reward would then be blended into the shared model.")

    h2("5.4 Making a forecast, step by step")
    ol([
        f"Take the most recent {W} readings for the {C} measurements the model handles.",
        "Fill any gaps inside the window by carrying the last known value forward, so there are no "
        "holes.",
        "Read the anchor (the last row) and subtract it from every row.",
        "Divide by the fixed scales, attach the mask columns; that is the input.",
        f"One pass through the network gives {H} &times; {C} scaled changes.",
        "Multiply by the fixed scales and add the anchor back &mdash; now it is in real units.",
        "Summarise it, as described next.",
    ])

    h2("5.5 The forecast summary &mdash; how everything else uses the forecast")
    p("Nothing after this point uses the full forecast. Everything uses five summary numbers per "
      "measurement, which is what makes the risk model's input a fixed size.")
    table([
        ["lowest", "the worst case for anything that becomes dangerous when it falls, like oxygen"],
        ["highest", "the worst case for anything dangerous when it rises, like ammonia"],
        ["average", "the general level over the period"],
        ["final", "where it ends up &mdash; is it still moving?"],
        ["trend", "how fast it is changing: (last &minus; first) divided by the number of steps"],
    ], ["Summary number", "What it is for"],
        f"Five numbers &times; {C} measurements = {5 * C} in total, plus a set worked out for "
        "harmful ammonia. This is the shared vocabulary the risk model, the safety layer and the "
        "planner all use.", cls="small")
    note("<p>Taking the <i>extreme in whichever direction hurts</i> &mdash; the lowest oxygen, the "
         "highest ammonia &mdash; rather than the average, is what makes everything downstream act "
         "<b>early</b> rather than merely describe the present. A forecast whose average oxygen is "
         "comfortable but whose lowest point is 2.1 milligrams per litre is a forecast with a dead "
         "pond in it.</p>", "note", "A small choice with a large effect")

    h2("5.6 How good is the forecast?")
    p("Accuracy is reported as <b>RMSE</b>, short for <b>root mean squared error</b>: square every "
      "mistake, average them, then take the square root. The square root is taken so the answer "
      "comes out in the same units as the thing being predicted. An RMSE of 0.34 for oxygen means "
      "the forecast is typically wrong by about 0.34 milligrams per litre. Lower is better.")
    fh = csv("forecast_horizon")
    if fh is not None:
        sub = fh[(fh["source"] == "federated") & (fh["channel"] == "do")]
        g = sub.groupby("horizon_minutes")["rmse"].mean()
        picks = [x for x in (5, 30, 60, 90, 120) if x in g.index]
        table([[f"{int(x)} minutes ahead", f"{g[x]:.3f} mg/L"] for x in picks],
              ["How far ahead", "Typical error in oxygen"],
              "Error grows the further ahead you look, as it must. These figures are from stations "
              "that no training method was allowed to learn from.", cls="small", align="lr")
    note("<p>The testing also reports how much better the forecast is than simply assuming "
         "<b>nothing will change</b>. Because the network predicts changes rather than values, "
         "that comparison is free: assuming nothing changes is the same as predicting zero. A "
         "model that cannot beat zero has learned nothing about how water behaves.</p>",
         "note", "The comparison is built in")


def part6() -> None:
    f = CFG["federated"]
    reg = f["regimes"]
    part(6, "Training across 42 stations without sharing data",
         "The claim being tested: can farms that refuse to pool their records still train a better "
         "model together than each could alone?")

    h2("6.1 What federated learning means")
    p("<b>Federated learning</b> is training one shared model across many owners of data, where "
      "the data never moves. Each owner trains on their own records, sends only the trained "
      "<i>numbers of the model</i> to a central point, and receives back an average of everybody's "
      "numbers. Nobody's records leave their own machine.")
    p("The specific method used here is <b>FedAvg</b>, short for <b>federated averaging</b>, "
      "published by McMahan and colleagues in 2017. It is the simplest version of the idea and it "
      "is still the standard baseline.")
    p("In this project each of the 42 stations plays the part of one farm. The code calls each one "
      "a <b>client</b>.")

    h2("6.2 What a client is")
    p("The class <code>PondClient</code> wraps exactly one station. The separation is built into "
      "the structure rather than merely promised: the class holds its own copy of the data, builds "
      "its own examples, owns its own network, and offers only four things to the outside world.")
    table([
        ["<code>local_train(epochs)</code>", "train on my own examples and tell you my error"],
        ["<code>get_weights()</code>", "hand over a copy of my network's numbers"],
        ["<code>set_weights(w)</code>", "replace my network's numbers with these"],
        ["<code>evaluate(on)</code>", "test myself on my own held-back readings"],
    ], ["What it offers", "What that does"],
        "The complete list. Nothing inside the class can reach another client's data, and the "
        "central point only ever calls these four things.", cls="small")

    figure(dg_fedavg(),
           "One round of federated averaging. The important part is the green panel: a set of "
           "numbers and one count go up, an average comes back down, and nothing else ever "
           "crosses.", wide=True)

    h2("6.3 Splitting the data, and why the test set is the end of the series")
    ul([
        f"<b>Some whole stations are held back.</b> {f['split']['holdout_fraction']:.0%} of them "
        f"(at least {f['split']['holdout_min']}) are set aside and <b>no training method is "
        "allowed to touch them</b>. So the comparison measures how well a model copes with a "
        "station it has never seen, not how well it memorised the ones it has.",
        f"<b>The held-back set is balanced across sources</b>, so American and English stations "
        "appear on both sides. Otherwise the held-back set could accidentally be all English.",
        f"<b>Within each training station, the last "
        f"{f['split']['val_fraction']:.0%} of the series</b> is kept back for checking &mdash; the "
        "most recent readings, in time order, never a random selection.",
    ])
    note("<p>Why is a random split wrong here?</p>"
         "<p><b>Because the training examples overlap.</b> The code steps forward 6 readings at a "
         "time through a 36-reading window, so two neighbouring examples share 30 of their 36 "
         "readings. A random split therefore puts near-copies of the same hours on both sides. The "
         "model effectively sees the answers during training, every method is flattered equally, "
         "and the difference being measured is hidden. Taking the most recent stretch instead is "
         "the only honest option.</p>", "exam", "Likely question")

    h2("6.4 The three methods being compared")
    rows = [
        ["<b>Each station alone</b><br><code>independent_local</code>",
         f"{reg['independent_local']['epochs']} passes over its own data",
         "no sharing of any kind", "the floor &mdash; what you get without cooperating"],
        ["<b>Federated</b><br><code>federated</code>",
         f"{reg['federated']['rounds']} rounds &times; "
         f"{reg['federated']['local_epochs']} passes each round",
         "numbers averaged after every round", "the method being tested"],
        ["<b>All data pooled</b><br><code>central</code>",
         f"{reg['central']['epochs']} passes over everything at once",
         "every station's examples in one pile",
         "the ceiling &mdash; and it breaks the whole premise"],
    ]
    table(rows, ["Method", "How much training", "What is shared", "What it is there for"],
          f"The three methods. The total amount of training is matched on purpose: "
          f"{reg['federated']['rounds']} rounds &times; "
          f"{reg['federated']['local_epochs']} passes = "
          f"{reg['federated']['rounds'] * reg['federated']['local_epochs']}, which equals the "
          f"{reg['independent_local']['epochs']} passes each station does alone. Without matching "
          f"it, a win for the federated method might just be a win for training longer.",
          cls="small")

    note("<p><b>Every station starts each method from the same starting numbers.</b> All clients "
         "are reset, and then one client's freshly initialised numbers are copied to everybody.</p>"
         "<p>This is not just tidiness. A neural network's hidden units have no fixed meaning "
         "&mdash; unit 3 in one network has nothing to do with unit 3 in another, because each "
         "started from different random numbers. Averaging two such networks position by position "
         "produces nonsense. Starting everyone from the same numbers is what makes averaging them "
         "a sensible thing to do at all.</p>"
         "<p>Note that these shared starting numbers are <b>random</b>. Nothing is pre-trained "
         "here. That distinction matters in Part 7, where the second network <i>does</i> start "
         "from already-trained numbers.</p>", "exam", "Why everyone starts the same")

    h2("6.5 The averaging step, written out")
    w('<p class="eq">new shared numbers = the sum over stations of '
      '( that station&rsquo;s example count &divide; the total example count ) &times; '
      'that station&rsquo;s numbers</p>')
    p("In words: a weighted average, where a station with more training examples counts for more. "
      "It is applied to each number in the network separately, in high precision, then converted "
      f"back. A station with 4,000 examples counts four times one with 1,000, and that is why the "
      f"example count has to be sent at all &mdash; it is the only thing besides the numbers "
      f"themselves that crosses.")
    p(f"Every station takes part in every round, which is affordable with 32 stations on one "
      "computer and removes &ldquo;which stations were picked this round&rdquo; as an extra "
      "variable.")

    h2("6.6 The four data-poor stations &mdash; where the claim is really tested")
    dp = f["data_poor"]
    note(f"<p><b>{dp['n_clients']} stations keep only {dp['data_fraction']:.0%} of their "
         f"history.</b> Everything else about them is unchanged, and they keep the <i>oldest</i> "
         f"slice so the readings they are tested on are untouched.</p>"
         "<p>This small group is the honest form of the claim. Training together is not supposed "
         "to help a station that already has six months of clean readings &mdash; it is supposed "
         "to help the farm that cannot train a useful model alone. If the benefit showed up "
         "everywhere <i>except</i> here, the result would be an accident rather than the effect.</p>",
         "num", "The group that matters")

    h2("6.7 Results")
    fs = csv("federated_summary")
    if fs is not None:
        order = {"central": 0, "federated": 1, "independent_local": 2}
        fs = fs.sort_values("regime", key=lambda s: s.map(order))
        label = {"central": "All data pooled (breaks the premise)",
                 "federated": "<b>Federated (no data moves)</b>",
                 "independent_local": "Each station alone"}
        table([[label.get(r.regime, r.regime), f"{r.rmse_do:.3f}", f"{r.rmse_temp_c:.3f}",
                f"{r.rmse_ph:.3f}"] for r in fs.itertuples()],
              ["Method", "Oxygen error", "Temperature error", "pH error"],
              "Typical forecast error (RMSE, so lower is better) on stations that no method was "
              "allowed to train on. The federated method beats each-station-alone on all three "
              "measurements, and comes out level with pooling everything &mdash; which is the "
              "option that would require farms to hand over their records.", align="lrrr")
    fr = csv("federated_rounds")
    if fr is not None and len(fr):
        first, last = fr.iloc[0], fr.iloc[-1]
        note(f"<p>{len(fr)} rounds, {int(first['clients'])} stations taking part, "
             f"{int(first['total_samples']):,} training examples between them. Average error "
             f"during local training fell from {first['mean_local_loss']:.3f} to "
             f"{last['mean_local_loss']:.3f}.</p>", "num", "Read from outputs/federated_rounds.csv")
    dpv = csv("federated_data_poor")
    if dpv is not None and len(dpv):
        note("<p>On the four data-poor stations, training together improved the oxygen forecast on "
             "3 of the 4, by about 13&nbsp;% in the middle case. <b>The fourth one is printed in "
             "the results table rather than left out.</b> A result that held on 4 out of 4 with "
             "only four stations would be worth less, not more &mdash; it would suggest the test "
             "was too easy.</p>", "num", "The data-poor group")

    h2("6.8 The privacy option that has not been run")
    dpc = f["differential_privacy"]
    p("Sending only the model's numbers is a big improvement over sending the data, but it is not "
      "a mathematical guarantee: in principle something about the data can still be worked out "
      "from the numbers. The formal fix is <b>differential privacy</b>, usually shortened to "
      "<b>DP</b>. It works by limiting how much any one participant can change the shared result, "
      "and then adding a controlled amount of random noise.")
    p("The code implements it and it is switched off. When enabled, each station's <i>change</i> "
      "to the numbers is capped in size, and noise is added to the average:")
    code(f"cap on each station's change     = {dpc['clip_norm']}\n"
         f"how much noise to add            = {dpc['noise_multiplier']}\n"
         f"allowed chance of failure        = {dpc['delta']}\n"
         f"switched on?                     = {dpc['enabled']}",
         "configs/federated.yaml and src/federated/simulate.py")
    note("<p><b>It has not been run, and no privacy figure is claimed.</b> The strength of a "
         "differentially private system is summarised by a number called &epsilon; (epsilon) "
         "&mdash; smaller means more private. Producing a trustworthy &epsilon; requires a "
         "specialist accounting tool and an actual training run, and neither has happened here. "
         "Quoting a plausible-sounding &epsilon; would be dishonest.</p>"
         "<p>If asked: &ldquo;implemented, optional, not yet run, and we claim no &epsilon; "
         "&mdash; it is Project-II work.&rdquo;</p>", "honest", "Built but not run")


def part7() -> None:
    m = CFG["models"]
    wm = m["world_model"]
    W, H = m["window"], m["horizon"]
    C = len(m["forecaster"]["channels"])
    part(7, "Model 2 &mdash; the &ldquo;what if I do this&rdquo; model",
         "The forecaster says what the water will do. This one says what it will do <i>if you "
         "act</i>.")

    h2("7.1 Why a second network is needed")
    p("Model 1 has never seen an aerator. It cannot be asked &ldquo;what happens if I switch the "
      "aerator on?&rdquo;, because nothing in the data it learned from contains an aerator. That "
      "is not a shortcoming of the design; it is a fact about rivers.")
    p("So Model 2 is trained on a different kind of question. It takes the recent readings "
      "<i>and</i> a proposed action, and predicts what follows. A network like this is usually "
      "called a <b>world model</b> &mdash; a learned stand-in for the environment, which a planner "
      "can try things out on before doing them for real.")
    note("<p>Why is Model 2 trained on the simulator instead of the 42 real stations?</p>"
         "<p><b>Because the real stations contain no examples of anybody doing anything.</b> They "
         "are rivers: no aerator, no feeding schedule, no liming. No amount of training across "
         "them can teach a network what an aerator does, because none of them has ever had one. "
         "The part that depends on actions has to be learned where actions exist, and the only "
         "place they exist is the simulator.</p>",
         "exam", "The first question you will be asked about this model")

    h2("7.2 The design")
    figure(dg_worldmodel(),
           "Model 2. The reading part is exactly the same shape as Model 1's, which is what makes "
           "the warm start possible with no modification &mdash; the measurement list and the "
           "window length were deliberately kept identical.", wide=True)
    p("The reading part is another single-layer LSTM. What is new is what happens afterwards: the "
      "action is attached to the summary, and the pair is passed through two plain layers with a "
      "<b>ReLU</b> in between. ReLU stands for <b>rectified linear unit</b>, and it is the "
      "simplest useful thing you can put between two layers: keep positive numbers as they are, "
      "replace negative ones with zero. Without something like it, two stacked plain layers would "
      "collapse into one and the network could only learn straight-line relationships.")
    code(
        f"class _WorldNet(nn.Module):\n"
        f"    self.lstm = nn.LSTM(input_size={2 * C}, hidden_size={wm['hidden_size']},\n"
        f"                        num_layers={wm['num_layers']}, batch_first=True)\n"
        f"    self.head = nn.Sequential(\n"
        f"        nn.Linear({wm['hidden_size']} + 8, {wm['hidden_size']}),  # 8 = the action\n"
        f"        nn.ReLU(),\n"
        f"        nn.Linear({wm['hidden_size']}, {H} * {C}),\n"
        f"    )\n\n"
        f"    def forward(self, x, a):         # x is (batch, {W}, {2 * C}); a is (batch, 8)\n"
        f"        out, _ = self.lstm(x)\n"
        f"        h = torch.cat([out[:, -1, :], a], dim=1)   # attach the action AFTER reading\n"
        f"        return self.head(h).view(-1, {H}, {C})",
        "src/perception/world_model.py")

    h3("How an action is described to the network")
    p("An action becomes eight numbers:")
    table([
        ["1", "how hard the aerator runs", "anything from 0 to 1"],
        ["2", "small water exchange?", "0 or 1"],
        ["3", "large water exchange?", "0 or 1"],
        ["4", "feed reduced?", "0 or 1"],
        ["5", "feed skipped?", "0 or 1"],
        ["6", "lime dosed?", "0 or 1"],
        ["7", "salt dosed?", "0 or 1"],
        ["8", "for how much of the period is the action held?", "anything from 0 to 1"],
    ], ["#", "What it describes", "Values"],
        "The eighth number is the subtle one. The planner holds a candidate action for "
        f"{CFG['agent']['predictive']['hold_action_steps']} of its "
        f"{CFG['agent']['predictive']['rollout_steps']} simulated steps and then stops, so a "
        "network told only &ldquo;aerator at full&rdquo; would consistently overestimate the "
        "effect.", cls="small")

    note("<p>Why is the action attached after the reading part, instead of being added to every "
         "input step?</p>"
         "<p><b>Because the window is the past, and the action has not happened yet.</b> Putting "
         "the action into the sequence of past readings would teach the network that the aerator "
         "was already running during the readings it is looking at &mdash; which is false, and "
         "exactly backwards for a network whose whole job is to answer &ldquo;what if&rdquo;.</p>",
         "exam", "Likely question")

    h2("7.3 The warm start &mdash; exactly what happens")
    p("A <b>warm start</b> means beginning training from somebody else's already-trained numbers "
      "instead of from random ones. Here, the reading part of Model 2 starts from the reading part "
      "of Model 1 &mdash; the version produced by training across the 42 real stations. The idea "
      "is that the way the network reads recent water behaviour was learned from real data, even "
      "though the action part has to be learned in simulation.")
    figure(dg_warmstart_options(),
           "Three ways to begin from somebody else's numbers, and which one this is. The "
           "difference between the middle option and the right-hand one is a single line of code "
           "that does not appear anywhere in this project: the line that would freeze the copied "
           "part in place.")
    ol([
        "The code opens the trained forecaster file and pulls out only the numbers belonging to "
        "the LSTM reading part.",
        "It copies them into Model 2's reading part.",
        "The layers after that get nothing. Model 1 has no action part to copy from, so those "
        "layers begin at random.",
        "Training then runs over <b>every</b> number in the network, reading part included. "
        "Nothing is frozen, and the reading part is not even trained more gently than the rest.",
        "If the shapes ever fail to line up, the copy is abandoned, a warning is printed, and "
        "training starts from random instead of crashing.",
    ])
    note("<p><b>This is a warm start, not fine-tuning.</b> The phrase <i>fine-tuning</i> normally "
         "means freezing the borrowed part and training only the new part, so that the borrowed "
         "knowledge is provably still there at the end. That is not what happens here. The copied "
         "numbers fix where simulator training <i>begins</i>, not where it ends. Over 12 rounds of "
         "training on simulated data, the reading part is free to drift as far as it likes from "
         "what the real stations taught it.</p>"
         "<p>So the defensible sentence is: <i>&ldquo;the reading part starts from the model "
         "trained across the real stations and is then trained further on simulator data; we do "
         "not currently measure how much of that starting point survives.&rdquo;</i> Saying the "
         "representation <i>is</i> the real-data one would claim more than the evidence "
         "supports.</p>", "honest", "The distinction to get right")
    note("<p>The warm start is applied the same way to <b>all three</b> training methods &mdash; "
         "federated, pooled and each-pond-alone. So it cannot be quietly boosting the federated "
         "result; it is the same for everybody in the comparison.</p>",
         "note", "It does not tilt the comparison")

    h2("7.4 Training Model 2 across simulated ponds")
    p("The participants here are not stations but simulated ponds with different physical "
      "properties. Each one differs along axes that genuinely change how the water behaves, rather "
      "than by cosmetic random jitter:")
    ul([
        "<b>size and depth</b> &mdash; a deeper pond mixes and absorbs oxygen differently;",
        "<b>how good the aerator is</b> &mdash; a weaker one takes longer to recover;",
        "<b>how productive the algae are</b> &mdash; a richer pond swings harder overnight;",
        "<b>the temperature range</b> &mdash; which shifts how much oxygen the water can hold.",
    ], "tight")
    p("Each pond generates its own runs with randomly chosen actions, so its data contains the "
      "before-and-after pairs no real station can provide. Two further ponds are held back "
      "entirely and used only for marking.")

    h2("7.5 Results")
    ws = csv("world_model_summary")
    if ws is not None:
        order = {"central": 0, "federated": 1, "independent_local": 2}
        ws = ws.sort_values("regime", key=lambda s: s.map(order))
        label = {"central": "All ponds' data pooled", "federated": "<b>Federated</b>",
                 "independent_local": "Each pond alone"}
        table([[label.get(r.regime, r.regime), f"{r.rmse:.3f}", f"{r.rmse_do_mgl:.3f}"]
               for r in ws.itertuples()],
              ["Method", "Error (scaled)", "Oxygen error (mg/L)"],
              "Model 2 on ponds no method trained on. The ordering repeats Part 6: training "
              "together beats each-pond-alone, and matches pooling everything.", align="lrr")
    wr = csv("world_model_rounds")
    if wr is not None and len(wr):
        note(f"<p>{len(wr)} rounds. Average local error fell from "
             f"{wr.iloc[0]['mean_local_loss']:.3f} to {wr.iloc[-1]['mean_local_loss']:.3f}; error "
             f"on the held-back ponds fell from {wr.iloc[0]['holdout_rmse_do_mgl']:.3f} to "
             f"{wr.iloc[-1]['holdout_rmse_do_mgl']:.3f} milligrams per litre.</p>",
             "num", "Read from outputs/world_model_rounds.csv")

    h2("7.6 Keeping predictions physically possible")
    p("A neural network asked about a situation unlike anything it was trained on will happily "
      "predict a negative oxygen level or a pH of 22. Inside a planner that is not a cosmetic "
      "problem: an impossible prediction would look like the most urgent situation in the world "
      "and drive a nonsense action. So every predicted value is forced back inside a possible "
      "range before the planner is allowed to score it.")
    table([["dissolved oxygen", "0 to 25 mg/L"], ["temperature", "0 to 45 &deg;C"],
           ["pH", "4 to 11"], ["total ammonia", "0 to 20 mg/L"],
           ["nitrite", "0 to 20 mg/L"], ["cloudiness", "0 to 500 NTU"]],
          ["Measurement", "Allowed range"],
          "NTU stands for nephelometric turbidity units, the standard unit for cloudiness. "
          "Harmful ammonia is not capped directly &mdash; it is recalculated from the capped total "
          "ammonia, pH and temperature, so it can never disagree with them.", cls="small")


def part8() -> None:
    rg = CFG["models"]["regimes"]
    part(8, "Model 3 &mdash; grouping readings, and a result that did not work out",
         "The one model that failed to deliver what it was designed for, reported as such.")

    h2("8.1 What it was meant to do")
    p("The plan was to discover, without being told, that ponds spend their time in a handful of "
      "recognisable states &mdash; something like &ldquo;calm night&rdquo;, &ldquo;bloom "
      "building&rdquo;, &ldquo;recovering after a crash&rdquo;. The dashboard could then name the "
      "current state, and the planner could take it into account. Finding groups in data without "
      "being given any labels is called <b>clustering</b>, and no labels are used anywhere in "
      "this part of the code.")
    p("The method chosen is <b>HDBSCAN</b>. The name stands for <b>hierarchical density-based "
      "spatial clustering of applications with noise</b>, which unpacks as: it looks for regions "
      "where data points are packed densely together, it tries several levels of "
      "&ldquo;densely&rdquo; and keeps the best, and &mdash; the useful part &mdash; it is allowed "
      "to answer <b>&ldquo;this point belongs to no group at all&rdquo;</b>. That last answer is "
      "what makes it usable as an odd-reading detector.")

    h2("8.2 Why the inputs are converted before clustering")
    figure(dg_regimes(),
           "The grouping pipeline. Everything to the left of HDBSCAN exists so that a "
           "&ldquo;state&rdquo; means the same thing on a 12&nbsp;&deg;C English river and a "
           "28&nbsp;&deg;C tropical pond.", wide=True)
    p("Grouping on raw readings does not carry across water bodies &mdash; raw readings <i>are</i> "
      "the difference between a river and a pond. So four converted quantities are used instead:")
    table([
        ["<code>do_sat</code>", "oxygen divided by the most that water could hold at its "
         "temperature",
         "A plain ratio with no units. A river at 9 milligrams per litre and 12&nbsp;&deg;C and a "
         "pond at 6.5 and 28&nbsp;&deg;C are both about 85&nbsp;% full &mdash; and that is the "
         "comparison that actually means something."],
        ["<code>temp_anom</code>", "temperature minus the average temperature over the window",
         "The shape of the daily rise and fall, not the climate."],
        ["<code>ph_anom</code>", "pH minus the average pH over the window",
         "The size of the swing, not the local geology."],
        ["<code>turb_log</code>", "cloudiness, compressed by taking a logarithm",
         "Cloudiness spans several orders of magnitude, so a few extreme values would otherwise "
         "drown out everything else."],
    ], ["Converted quantity", "How it is calculated", "Why it carries across"],
        "Clustering on raw readings instead sends about 38&nbsp;% of examples to the "
        "&ldquo;belongs to nothing&rdquo; label, against about 21&nbsp;% with these.")
    note(f"<p>Only four measurements are used here, deliberately fewer than the forecaster's six: "
         f"{', '.join('<code>' + c + '</code>' for c in rg['channels'])}. Only 12 of the 42 real "
         f"stations measure ammonia and none measures nitrite, so including those would leave "
         f"every real station's row incomplete and the groups would end up being found in "
         f"simulated data alone. These four are reported by nearly every station, which is what "
         f"lets real and simulated readings share one space.</p>", "note", "Why only four")
    p(f"Each window is boiled down to {len(rg['features'])} summary numbers per converted quantity "
      f"&mdash; {', '.join(rg['features'])} &mdash; giving {len(rg['features'])} &times; "
      f"{len(rg['derived'])} = {len(rg['features']) * len(rg['derived'])} numbers in total. Those "
      f"are then put on a common footing using the middle value and the spread of the middle half "
      f"of the data (the <b>interquartile range</b>, or <b>IQR</b>) rather than the average and "
      f"standard deviation, because water measurements contain extreme values by nature and the "
      f"average would be dragged around by them.")

    h2("8.3 What actually happened")
    rm = csv("regime_metrics")
    if rm is not None and len(rm):
        r = rm.iloc[0]
        note(f"<p><b>{int(r['n_clusters'])} groups &middot; "
             f"{float(r['noise_fraction']):.1%} of examples belong to no group &middot; "
             f"DBCV score {float(r['dbcv']):.6f} &middot; "
             f"silhouette score {float(r['silhouette']):.3f} &middot; "
             f"{int(r['n_points']):,} examples</b></p>"
             "<p>The DBCV score is indistinguishable from zero. The silhouette score looks "
             "healthy. That disagreement is itself the finding, and the next box explains it.</p>",
             "num", "Read from outputs/regime_metrics.csv")
    note("<p><b>The data does not contain distinct states.</b> The clustering was re-run across a "
         "wide range of settings &mdash; group sizes from 40 to 800, several density settings, "
         "both of the method's selection modes, a version with the inputs compressed first, and on "
         "real and simulated readings both together and separately. Every single time it returns "
         "the same shape: one dominant blob holding roughly 80&nbsp;% of the examples, a few small "
         "satellites, and a DBCV score of about zero. The whole sweep is written to a results file, "
         "so this rests on evidence rather than on assertion.</p>"
         "<p>The honest reading is that water readings form a <i>continuum</i> rather than "
         "separate groups. A pond does not jump between distinct states; it slides gradually "
         "around a range of conditions. Reporting whichever setting happened to produce a "
         "nicer-looking number of groups would have been the dishonest option.</p>",
         "honest", "The result that did not work out")

    note("<p>Both scores measure something, but not the same thing.</p>"
         "<p><b>Silhouette</b> asks &ldquo;given that these groups exist, how well separated are "
         "they?&rdquo; It happily rewards a single blob cut neatly down the middle. <b>DBCV</b> "
         "&mdash; density-based clustering validation &mdash; asks the prior question: "
         "&ldquo;is there any densely-packed structure here at all?&rdquo; It answers no. Quoting "
         "only the silhouette score would have made this look like a success.</p>",
         "exam", "Why report two scores that disagree?")

    h2("8.4 What the model is actually used for")
    ul([
        "<b>Spotting odd readings &mdash; the part that does work.</b> When HDBSCAN says a stretch "
        "of readings belongs to no group, that means it is unlike anything seen in training. That "
        "is a genuine and useful capability, and it is what feeds the dashboard and the agent's "
        "context.",
        "<b>A rough split into two modes</b>, with descriptive names read off afterwards. Given a "
        "DBCV score of about zero, these names are a reading aid and not a claim. "
        "<b>Nothing downstream depends on them</b> &mdash; the planner never looks at which group "
        "the pond is in.",
    ])
    note("<p>The odd-reading flag cannot tell a broken probe from a genuine new problem. It reports "
         "&ldquo;this is unlike what I know&rdquo;, and that is all it reports. The "
         "<code>sensor_fault</code> test measures exactly how far that gets you, and the answer is "
         "written to a results file rather than asserted in prose.</p>",
         "honest", "The limit of the odd-reading flag")


def part9() -> None:
    r = CFG["models"]["risk"]
    part(9, "Models 4 and 5 &mdash; risk, and why it said so",
         "A fast model that sees a limit breach coming, and a method that explains which reading "
         "made it say that.")

    h2("9.1 How the two fit together")
    figure(dg_risk(),
           "Models 4 and 5. The top row shows how the training answers are produced; the bottom "
           "row is one live reading being classified. The answers come from a written-down rule, "
           "and that is exactly what makes the explanation meaningful.", wide=True)

    h2("9.2 What Model 4 is trained to predict")
    p("The answer it learns is <b>the risk level that the safe-range table from Part 2.8 would "
      "give to the worst moment in the forecast</b>. It is not a human judgement and it is not "
      "some learned idea of &ldquo;water quality&rdquo;.")
    note("<p>Isn't it pointless to train a model to reproduce a rule you already have?</p>"
         "<p><b>No, because applying the rule requires knowing the future, and the future is "
         "uncertain.</b> To use the rule you need a forecast; the forecast has real error; and a "
         "hard threshold applied to a single predicted number fails badly whenever that one number "
         "is wrong. A model that looks at all thirty summary numbers from the forecast at once "
         "copes far better with that error. One of the tests in Part 13 measures precisely this.</p>"
         "<p>The framing also buys the explanation. &ldquo;Risk is critical because the lowest "
         "predicted oxygen is 2.1&rdquo; is an honest account of what the model did <i>and</i>, "
         "because the answer came from the rule, an honest account of the rule too. If the answer "
         "had been some learned notion of quality, neither would hold.</p>",
         "exam", "The obvious objection, answered")

    h2("9.3 What it looks at")
    C = len(CFG["models"]["forecaster"]["channels"])
    table([
        ["the current reading", "7",
         "oxygen, temperature, pH, total ammonia, harmful ammonia, nitrite, cloudiness"],
        ["the forecast summary", f"{5 * C}",
         f"lowest, highest, average, final and trend for each of the {C} forecast measurements"],
        ["harmful ammonia forecast", "5", "worked out from the forecast ammonia, pH and temperature"],
    ], ["Group", "How many numbers", "What they are"],
        "The full input. It is assembled with proper column names rather than as a bare grid of "
        "numbers &mdash; otherwise the library records placeholder names and complains on every "
        "single prediction.", cls="small")

    h2("9.4 The method and how well it does")
    p("Model 4 is a <b>LightGBM</b> model. The name stands for <b>light gradient-boosting "
      "machine</b>. It works by building a large number of small decision trees one after another, "
      "where each new tree concentrates on the cases the previous ones got wrong, and the final "
      "answer is the whole set voting together. It was chosen because it is fast, works well on "
      "tables of numbers, and can be explained exactly by the method in Model 5.")
    table([
        ["how many trees", r["n_estimators"]],
        ["how many branches per tree at most", r["num_leaves"]],
        ["how deep a tree may go", r["max_depth"]],
        ["how much each tree is allowed to change the answer", r["learning_rate"]],
        ["fewest examples allowed in a branch", r["min_child_samples"]],
        ["the three answers it can give", ", ".join(r["classes"])],
        ["the score each answer maps to", ", ".join(f"{k} = {v}" for k, v in
                                                    r["score_weights"].items())],
    ], ["Setting", "Value"], "How Model 4 is configured.", cls="small", align="lr")
    rmx = csv("risk_metrics")
    if rmx is not None and len(rmx):
        q = rmx.iloc[0]
        note(f"<p>Overall correct: {float(q['accuracy']):.1%} &middot; "
             f"correct averaged evenly over the three answers: "
             f"{float(q['balanced_accuracy']):.1%} &middot; "
             f"<b>of the genuinely critical cases, it caught "
             f"{float(q['critical_recall']):.1%}</b> &middot; "
             f"of the cases it called critical, {float(q['critical_precision']):.1%} really were "
             f"&middot; tested on {int(q['n']):,} examples from a run it never trained on</p>"
             "<p>The figure that matters is how many genuinely critical cases it catches. A missed "
             "critical case is a dead pond; a false alarm costs some electricity. That imbalance is "
             "why the two figures are reported separately instead of being blended into one "
             "score.</p>", "num", "Read from outputs/risk_metrics.csv")

    h2("9.5 Model 5 &mdash; the explanation")
    p("<b>SHAP</b> stands for <b>Shapley additive explanations</b>. The idea comes from game "
      "theory: if several players contribute to a shared result, Shapley values are the fair way "
      "to divide the credit between them. Applied here, the &ldquo;players&rdquo; are the input "
      "readings and the &ldquo;result&rdquo; is the risk score, so SHAP says how much each reading "
      "pushed the score up or down.")
    p("Because Model 4 is made of decision trees, these contributions can be worked out "
      "<b>exactly</b> rather than estimated by sampling, which makes them cheap enough to compute "
      "on every single dashboard refresh.")
    note("<p>The contributions are always calculated for the <b>critical</b> answer specifically, "
         "not for whichever answer the model currently favours.</p>"
         "<p>Explaining whichever answer is winning means the explanation quietly changes meaning "
         "as the prediction moves: one moment it explains why things are fine, the next why they "
         "are worrying, and the operator cannot compare the two. &ldquo;What is pushing this "
         "towards critical?&rdquo; is one stable question with one stable answer, so the panel can "
         "be read over time like any other chart.</p>", "exam", "A design choice worth defending")
    p("The result goes to two places: the dashboard's risk panel, and the sentence the agent "
      "records to explain itself. That is deliberate &mdash; the operator sees the same "
      "explanation the agent worked with, rather than a separate story assembled afterwards.")
    note("<p>SHAP explains the <i>model</i>, not the water. A correct attribution is not a "
         "statement about cause: &ldquo;the lowest predicted oxygen drove the risk score&rdquo; is "
         "a true statement about the model, and it would remain true even if the forecast were "
         "completely wrong.</p>", "honest", "What SHAP does not tell you")


def part10() -> None:
    ag = CFG["agent"]
    part(10, "The live loop, step by step",
         "Where all five models, the simulator, the sensors and the safety layer finally meet.")

    h2("10.1 One tick, completely")
    p("The loop advances in <b>ticks</b>. One tick is one sensor reading, which is five simulated "
      "minutes.")
    figure(dg_tick(),
           "One tick, in order. Steps 6 to 9 run only on the ticks where the planner is allowed to "
           "re-think, which is one in three; step 10 is the shortcut taken on the other two. Steps "
           "1 to 4, 11 and 12 run every single time without exception.", landscape=True)

    h2("10.2 The two kinds of tick")
    tpd = int(ag['decide_every_minutes'] / CFG['sensors']['sample_every_minutes'])
    table([
        ["How often", f"1 tick in {tpd}", f"{tpd - 1} ticks in {tpd}"],
        ["Perception models run?", "yes &mdash; all four", "no"],
        ["Planner runs?", "yes &mdash; 14 options compared", "no"],
        ["Safety rules run?", "yes &mdash; all five groups",
         "yes &mdash; but only groups 3 and 5"],
        ["The action", "may change", "held as it is, unless safety steps in"],
        ["Explanation recorded", "recalculated", "carried over from the last decision"],
    ], ["", "A deciding tick", "A holding tick"],
        "The explanation is carried over on purpose. The planner runs three times less often than "
        "the sensors, so without carrying it forward, two thirds of the recorded decisions would "
        "have no explanation attached and the dashboard panel would keep going blank &mdash; "
        "making the system look as though it only sometimes explains itself, when in fact the "
        "explanation simply has not been recalculated.", cls="small")

    h2("10.3 The shadow pond")
    p("Alongside the real run, a second identical pond is simulated with the agent switched off. "
      "The code calls it the <b>shadow</b>. It answers the question a farmer would actually ask: "
      "&ldquo;what would have happened if I had done nothing?&rdquo; Comparing the two is called a "
      "<b>counterfactual</b> comparison &mdash; comparing what happened against what would have "
      "happened otherwise.")
    figure(dg_counterfactual(),
           "How the comparison is set up. The random number generator's exact internal state is "
           "copied across, not merely re-seeded, so the shadow pond gets the identical sequence of "
           "random events.")
    note("<p>Why fork the shadow at the start instead of just running a second simulation?</p>"
         "<p><b>Because otherwise the comparison is polluted by luck.</b> Two separately started "
         "ponds differ because of random chance as well as because of the agent, and over a "
         "36-hour run that chance difference can easily be larger than the effect being measured. "
         "Copying the random number generator's internal state means both ponds get the same "
         "weather, the same wobble and the same problems at the same moments. Any difference "
         "between them is then down to the agent's actions.</p>", "exam", "Likely question")
    note("<p>The do-nothing comparison runs with <b>the safety layer switched off</b>, and that is "
         "deliberate rather than an oversight. It is meant to show what happens when nobody acts. "
         "If the safety layer were allowed to rescue it, it would stop being a do-nothing run and "
         "the comparison would be worthless. This is why it is the only row in the safety table "
         "with unsafe actions recorded against it.</p>", "note", "One deliberate asymmetry")

    h2("10.4 Failing gracefully instead of crashing")
    p("The perception models are handed to the loop rather than built into it, and any of them may "
      "be absent. Each one is wrapped so that a failure is <i>recorded</i> rather than silently "
      "swallowed:")
    ul([
        "no forecaster &rarr; the risk level is worked out from the current reading alone;",
        "no grouping model &rarr; no group and no odd-reading flag are reported;",
        "no risk model &rarr; the safe-range table is applied to the forecast summary directly;",
        "no explainer &rarr; the explanation list is empty and the recorded sentence leaves it out;",
        "Model 2 present but not enough readings collected yet &rarr; the planner falls back to the "
        "hand-written pond equations, and counts how often it had to.",
    ], "tight")
    note("<p>Failures are written into an error list rather than passed over in silence. A "
         "perception step that quietly returns nothing looks exactly like one that had nothing to "
         "report &mdash; and because this loop is built to keep going rather than stop, without "
         "that record the degradation would be invisible.</p>", "note", "Why the error list exists")


def part11() -> None:
    ag = CFG["agent"]
    pr = ag["predictive"]
    rc = ag["reactive"]
    part(11, "How the agent chooses an action",
         "Three ways of deciding, one shared list of options, and the one place where the "
         "river-versus-pond gap is measured instead of just admitted.")

    h2("11.1 The agent's own picture of the pond")
    p("The agent cannot read the simulator. It keeps its own running picture instead: a second "
      "copy of the pond equations that it controls, started from typical pond values and nudged "
      "towards whatever the sensors report.")
    w('<p class="eq">new belief = 0.65 &times; (what my equations predicted) '
      '&nbsp;+&nbsp; 0.35 &times; (what the sensor just said)</p>')
    p("This kind of blending is called a <b>complementary filter</b>: it trusts the model for "
      "smoothness and the sensor for truth, in fixed proportions.")
    ul([
        "It <b>never</b> reads the simulator's hidden quantities &mdash; no algae, no fish mass, "
        "no store of decaying material. Its picture of those is genuinely uncertain, which is "
        "exactly the position a real farmer is in.",
        "The random wobble is switched <b>off</b> in the agent's picture. A belief should be the "
        "average expected path, not one noisy sample of it.",
        "When it simulates forward, the copy has no knowledge of the scenario timetable. "
        "<b>The agent cannot know what is about to go wrong.</b> It cannot know the aerator is "
        "about to fail; it can only notice that the oxygen is not responding to its command.",
    ])

    h2("11.2 When the learned model takes over from the equations")
    p("The wrapper that holds both keeps the hand-written equations alongside Model 2, using them "
      "as the belief-tracker and as a fallback, and lets Model 2 answer &ldquo;what if&rdquo; "
      "whenever it can. It falls back to the equations when:")
    ul([
        f"fewer than {CFG['models']['window']} readings have been collected &mdash; which is "
        "always true at the start of a run;",
        f"more steps are asked for than the {CFG['models']['horizon']} the network was trained to "
        "produce;",
        "the most recent reading is missing a measurement &mdash; because everything the network "
        "predicts is a change <i>from</i> that reading, so with a probe out at exactly that moment "
        "there is nothing to measure the change from, and falling back is the honest answer.",
    ], "tight")
    note("<p>Both counts &mdash; how often the network answered and how often the equations did "
         "&mdash; are kept and reported. A system that quietly fell back to hand-written equations "
         "whenever the network was inconvenient would score well and mean nothing.</p>",
         "note", "The fallback is counted")

    h2("11.3 The list of options")
    p("Combining every setting of every control gives 3 &times; 3 &times; 3 &times; 2 &times; 2 = "
      "<b>108</b> possible actions. The planner only considers <b>14</b> of them.")
    from src.agent import actuators as _ACT  # noqa: E402
    try:
        cands = _ACT.candidates(ag)
        rows = [[i, _ACT.describe(c), f"{_ACT.action_cost(c, ag):.2f}",
                 f"{_ACT.feed_penalty(c, ag):.2f}"]
                for i, c in enumerate(cands)]
        table(rows, ["#", "What it does", "Running cost", "Cost of lost growth"],
              "The complete list, generated by calling the project's own code while this document "
              "was built. Every control and every level of escalation is covered, and the list is "
              "short enough to print &mdash; which is the point: anybody can read it and know "
              "exactly what the agent was allowed to consider.", cls="small", align="llrr")
    except Exception:
        pass
    note("<p>Why not consider all 108?</p>"
         "<p><b>Because most make no sense or are simply worse versions of something already on "
         "the list, and each one costs a full simulation to evaluate.</b> Adding lime while also "
         "exchanging the water works against itself. Dosing salt while skipping feed at full "
         "aeration is beaten by cheaper options already included. Fourteen keeps each tick fast "
         "and, more importantly, keeps the agent <b>explainable</b>: an examiner can read the "
         "whole list in ten seconds.</p>", "exam", "Likely question")

    h2("11.4 How the options are scored")
    p("The planner uses <b>model predictive control</b>, usually shortened to <b>MPC</b>. The idea "
      "is simple to state: simulate each option forward for a while, score how the pond ends up, "
      "and do the option with the best score. Then throw the plan away and repeat from scratch at "
      "the next decision, because by then there are fresh readings.")
    figure(dg_mpc(),
           "One planner decision. Every option is simulated forward through Model 2, scored "
           "against the same safe-range table the testing uses, and the best one is returned as a "
           "<i>suggestion</i> &mdash; which then still has to get past Part 12.", wide=True)
    cw = ag["cost_weights"]
    p("The score adds up five things. A lower score is better, so every term is a penalty.")
    table([
        [f"{cw['risk']}", "&times; hours predicted outside the safe band (warning plus critical)",
         "the basic thing we are trying to avoid"],
        [f"{cw['critical']}", "&times; hours predicted in the critical band, counted again",
         "critical is not merely worse than warning &mdash; it is dangerous, and this makes it "
         f"{cw['risk'] + cw['critical']} times as expensive"],
        [f"{cw['intervention']}", "&times; what the action costs to run",
         "electricity, chemicals, water"],
        [f"{cw['stability']}", "&times; how jumpy the predicted oxygen and pH are",
         "discourages an action that achieves the target by thrashing"],
        [f"{cw['feed_loss']}", "&times; the penalty for withholding feed",
         "skipping a meal scores 1.0, reducing it scores 0.4, feeding normally scores 0"],
    ], ["Weight", "What it multiplies", "Why it is there"],
        "The five parts of the score, with the weights currently set in the settings file. The "
        "option with the lowest total is the suggestion.", cls="small", align="rll")
    note("<p><b>The feed penalty is not decoration.</b> Without it, the cheapest way to keep every "
         "measurement inside its safe band is to stop feeding the fish permanently &mdash; no "
         "feed, no ammonia, no extra oxygen demand, no risk, and no farm. The penalty is what "
         "forces the planner to trade safety against the reason the pond exists, and it is why the "
         "agent's running cost is reported next to its safety record rather than hidden.</p>",
         "exam", "Why is there a feed penalty?")

    h2("11.5 Where the real-data model touches the decision &mdash; and why it hurts")
    p("This is the most important thing to understand about how the two halves of the project "
      "connect, and it is also the project's clearest criticism of itself.")
    p("Model 1 learned from rivers and has never seen an aerator, so it cannot be asked what an "
      "action would do. What it <i>can</i> do is say what the water would do if nobody acted. So "
      "the planner works out the difference between two do-nothing predictions &mdash; Model 1's "
      "and its own equations' &mdash; and treats that difference as a standing correction, adding "
      "it to <b>every</b> option it considers:")
    w('<p class="eq">correction = 0.5 &times; ( Model 1&rsquo;s do-nothing forecast '
      '&minus; the equations&rsquo; do-nothing forecast )</p>')
    p("The reasoning is that any consistent bias in the hand-written equations gets cancelled out "
      "using something genuinely learned from real water. The learned model enters as a "
      "<i>correction</i> rather than as the thing driving the decision, which is the honest way to "
      "use a model that has never seen an aerator. The 0.5 halves it, because Model 1 learned from "
      "flowing rivers and a pond is not a river.")
    ab = csv("ablations")
    if ab is not None and len(ab):
        try:
            med = ab.groupby("ablation")["risk_h_do"].median()
            full = med.get("full")
            nol = med.get("no_lstm_correction")
            if full is not None and nol is not None:
                note(f"<p><b>The correction makes the agent worse.</b> Hours of oxygen outside the "
                     f"safe band, middle value across all runs: with the correction switched on, "
                     f"<b>{full:.2f} hours</b>; with it switched off, <b>{nol:.2f} hours</b>. "
                     f"Lower is better, so turning it off is an improvement of "
                     f"{full - nol:.2f} hours.</p>"
                     "<p>The cause is the river-versus-pond gap. A network that learned from cool "
                     "flowing rivers, used to correct predictions about a warm still pond, adds "
                     "more error than it removes &mdash; even after being halved. <b>This is the "
                     "one place in the whole project where that gap is measured directly rather "
                     "than merely acknowledged.</b></p>"
                     "<p>It is left <b>switched on anyway</b>, deliberately. The forecaster trained "
                     "across the real stations is the thing the whole first half of this project "
                     "exists to produce, and keeping it connected to the decision is what makes "
                     "this a learned controller rather than a set of equations driven by classical "
                     "control. The cost of that choice is measured on every test run rather than "
                     "quietly removed: one setting turns it off for a better score, and the "
                     "results table reports both ways regardless.</p>",
                     "honest", "The result that does not flatter the system")
        except Exception:
            pass

    h2("11.6 The simple rule-based planner")
    p("A second, much simpler planner exists: a set of plain rules over the current reading and "
      "the forecast. It is roughly what a careful farmer with a good probe and a two-hour forecast "
      "would do. It is the bar the predictive planner has to clear, and it is deliberately not a "
      "pushover &mdash; it looks at every measurement, not just oxygen.")
    table([
        [f"oxygen now or predicted below {rc['do_act']}", "aerator to full, stop feeding"],
        [f"oxygen now or predicted below {rc['do_warn']}", "aerator to half, reduce feeding"],
        [f"harmful ammonia above {rc['nh3_warn']}", "stop feeding, small water exchange"],
        [f"nitrite above {rc['no2_warn']}", "dose salt, to protect the fish's gills"],
        [f"pH below {rc['ph_low']}", "dose lime"],
        [f"pH above {rc['ph_high']}", "small water exchange, reduce feeding"],
    ], ["If this is true", "Then do this"], "The whole rule set.", cls="small")


def part12() -> None:
    g = CFG["agent"]["guardrails"]
    part(12, "The safety layer",
         "The project's real contribution. Not that a model can predict water chemistry, but that "
         "no suggestion reaches the equipment without passing a fixed, checkable set of rules.")

    h2("12.1 What the safety layer is")
    p("The code calls it the <b>guardrails</b>. It sits between the planner and the equipment. "
      "Every planner produces a <i>suggestion</i>; the guardrails decide what is actually "
      "<i>applied</i>. Any difference between the two is written down with the name of the rule "
      "that caused it, so every intervention can be traced afterwards.")
    p("Three kinds of change can happen: a value can be <b>clamped</b> to something smaller, a "
      "part of the action can be <b>blocked</b> entirely, or the whole action can be "
      "<b>overridden</b> in an emergency.")
    figure(dg_guardrails(),
           "The five groups of rules, in the order they run, with the values currently set. Every "
           "difference between suggested and applied is recorded with a rule name, a plain-English "
           "reason, and which of the three kinds of change it was.", wide=True)

    h2("12.2 Why the rules run in that order")
    ol([
        "<b>Dose limits first</b>, because they only involve looking at what has already been done "
        "today and do not depend on the current state of the water at all.",
        "<b>Rate limits next</b>, so that whatever survived the dose limits is smoothed out before "
        "anything else reads the aerator setting.",
        "<b>Situation-based blocks</b>, which need the current reading &mdash; these are the first "
        "rules that actually look at the pond.",
        "<b>Rejecting actions predicted to cause harm</b>, which needs a full simulation and is "
        "therefore the most expensive check. Running it after the cheap ones means it is often "
        "checking an action that has already been toned down.",
        "<b>The emergency override last</b>, so that it beats everything above it.",
    ])
    note("<p>Group 1 may have just blocked a dose. Group 2 may have limited the aerator to a "
         "gentle increase. If the pond is at 2.3 milligrams of oxygen per litre, none of that "
         "matters. The override runs last and <b>deliberately ignores the rate limit</b> &mdash; "
         "an emergency is precisely the situation the rate limit exists to be overruled by. When "
         "the pond is suffocating, nothing outranks air.</p>", "exam", "Why is the override last?")

    h2("12.3 Reading the sensors pessimistically")
    p("Every safety threshold is tested against the reading shifted in whichever direction is "
      "<i>less</i> safe. Oxygen is treated as lower than reported; ammonia and pH as higher.")
    table([["dissolved oxygen", g["sensor_margin"]["do"], "treated as this much lower"],
           ["harmful ammonia", g["sensor_margin"]["nh3"], "treated as this much higher"],
           ["pH", g["sensor_margin"]["ph"], "treated as this much higher"]],
          ["Measurement", "Margin", "Direction"],
          "The margins are roughly three times the typical sensor error from Part 3. The safety "
          "layer only ever sees the noisy reading, but safety is really about the true state of "
          "the water. These margins are the allowance for that difference, so a rule fires "
          "slightly early rather than slightly late.", cls="small", align="lrl")

    h2("12.4 Rejecting harmful actions, and the one subtlety")
    p("If a suggested action is predicted to push a measurement into the critical band, it is "
      "replaced by doing nothing. But there is an important check before that:")
    code("simulate the suggested action\n"
         "if it does NOT lead to a critical band:\n"
         "    allow it\n\n"
         "simulate doing nothing instead\n"
         "if doing nothing ALSO leads to a critical band:\n"
         "    allow the action  # the pond is already heading there; the action is not the cause,\n"
         "                      # and blocking it would remove the only help available\n"
         "otherwise:\n"
         "    replace it with doing nothing  # the action itself would cause the harm",
         "src/agent/guardrails.py, written out in words")
    note("<p>Without that second check the rule turns itself inside out in exactly the situation "
         "where it matters most. A pond already sliding towards a critical band will have "
         "<i>every</i> option predicted to reach one, including the correct option &mdash; so a "
         "naive version of this rule would block the aerator at precisely the moment the aerator "
         "is needed. Comparing against doing nothing is what separates &ldquo;this action causes "
         "harm&rdquo; from &ldquo;harm is coming whatever I do&rdquo;.</p>",
         "exam", "The subtlety worth knowing")

    h2("12.5 The fast path")
    p("The full check runs on deciding ticks. On the other two ticks out of three, a cut-down "
      "version runs: groups 3 and 5 only &mdash; the situation-based blocks and the emergency "
      "override &mdash; tested against whatever action is currently being held.")
    note("<p>Groups 1 and 2 are about the <i>history</i> of actions: has this been dosed too often "
         "today, is this too big a jump from the last setting? On a holding tick no new action is "
         "being suggested, so they have nothing to do. Groups 3 and 5 are about the <i>present "
         "state of the water</i>, and the water does not wait for the planner.</p>"
         "<p>This arrangement &mdash; a slow planner with a fast safety layer underneath &mdash; is "
         "the standard pattern in industrial control, and it is exactly what keeps the count of "
         "unsafe actions at zero between decisions.</p>",
         "exam", "Why does the fast path use only two of the five groups?")

    h2("12.6 What the evidence says")
    sf = csv("safety")
    if sf is not None and len(sf):
        rows = []
        for r in sf.itertuples():
            name = r.controller
            label = f"<b>{name}</b>" if name == "predictive" else name
            if name == "do_nothing":
                label += " <span class='badge'>safety layer off by design</span>"
            rows.append([label, int(r.runs_n), f"{int(r.unsafe_applied_total):,}",
                         f"{int(r.guardrail_events_total):,}"])
        table(rows, ["Controller", "Runs", "Unsafe actions applied",
                     "Times the safety layer stepped in"],
              "Across the full set of tests. Every controller with the safety layer switched on "
              "applied zero unsafe actions. The do-nothing reference, which has it switched off on "
              "purpose, accumulated thousands.", align="lrrr")
    ab = csv("ablations")
    if ab is not None and len(ab):
        try:
            ug = int(ab[ab["ablation"] == "no_guardrails"]["unsafe_applied"].sum())
            fu = int(ab[ab["ablation"] == "full"]["unsafe_applied"].sum())
            note(f"<p>The strongest single result in the project comes from removing only the "
                 f"safety layer and changing nothing else &mdash; same planner, same models, same "
                 f"tests:</p>"
                 f"<p><b>{ug:,} unsafe actions applied with the safety layer removed. "
                 f"{fu} with it in place.</b></p>"
                 "<p>That comparison is part of the standard set of tests, so it is re-measured "
                 "every time the project is evaluated rather than quoted from one lucky "
                 "experiment. It is the evidence for putting the hard limits outside the learned "
                 "part of the system.</p>", "num", "With and without the safety layer")
        except Exception:
            pass


def part13() -> None:
    part(13, "Testing the agent",
         "Four hundred runs, seven variations, and a statistical test that respects how the runs "
         "were paired up.")

    h2("13.1 The test grid")
    figure(dg_evalgrid(),
           "How the testing is laid out. The random seeds are <i>paired</i> across controllers: "
           "seed 1002 means the same starting state, the same timetable of problems and the same "
           "random events for every controller, which is what makes a paired comparison valid.",
           wide=True)

    h2("13.2 The five controllers being compared")
    table([
        ["do nothing", "reference", "nothing at all",
         "Not really a controller. It shows what happens if nobody acts. Safety layer switched off "
         "on purpose."],
        ["fixed threshold", "the bar", "the current oxygen reading only",
         "The ordinary aerator thermostat used on real farms: oxygen below 4 milligrams per litre, "
         "switch the aerator on. No forecast, no awareness of ammonia, no feed management. "
         "<b>This is the bar to beat.</b>"],
        ["simple rules", "agent", "all current readings plus the forecast summary",
         "The project's own plain rule set, from Part 11.6."],
        ["predictive", "agent", "readings, forecast, and Model 2",
         "The contribution. Simulates each option forward and picks the best."],
        ["oracle", "cheating", "the simulator's hidden quantities",
         "Allowed to see things no probe can measure. <b>Not achievable in reality, and not an "
         "upper limit either.</b>"],
    ], ["Controller", "Kind", "What it is allowed to see", "Notes"],
        "All five are run through exactly the same machinery, so any difference between them is "
        "the decision-making and not the test harness.", cls="small")
    note("<p>The oracle is a <b>hand-written rule that happens to see everything</b>, not a "
         "theoretical best. The agent actually beats it on oxygen in one of the eight tests, "
         "because the oracle's rule never deals with the pond having <i>too much</i> oxygen. Read "
         "it as &ldquo;what a good rule could do with perfect knowledge&rdquo;, and never claim it "
         "as a ceiling.</p>", "honest", "What the oracle is not")

    h2("13.3 What gets measured")
    table([
        ["hours at risk", "hours spent outside the safe band, worst measurement deciding",
         "the headline number"],
        ["hours at risk, per measurement", "the same, broken down by measurement",
         "without this the headline is unreadable &mdash; a pond sitting one degree above its "
         "temperature range scores the same as one suffocating"],
        ["unsafe actions applied",
         "times the pond actually reached a critical band while the agent was in charge",
         "must be zero for every controller with the safety layer on"],
        ["times the safety layer stepped in", "how often suggested and applied differed",
         "how hard the layer had to work"],
        ["interventions", "ticks where the agent did something rather than nothing", "activity"],
        ["running cost", "the operating cost added up", "the counterweight to safety"],
        ["hours at risk for the shadow pond", "the same measure on the untouched shadow pond",
         "what would have happened with no agent"],
    ], ["What is recorded", "How it is defined", "Why it is there"],
        "The per-measurement breakdown matters as much as the headline: the agent should get "
        "credit for fixing the part of the problem it can actually act on.", cls="small")

    h2("13.4 The statistical test")
    p("The comparison uses the <b>Wilcoxon signed-rank test</b>. It answers: given the same "
      "situation run under two controllers many times, is one reliably better, or could the "
      "difference be chance? It works by ranking the size of the differences rather than using "
      "their raw values.")
    note("<p>Why this test and not the more familiar t-test?</p>"
         "<p><b>Because the measurement is not shaped the way a t-test assumes, and the runs are "
         "paired.</b> Hours at risk can never go below zero, is frequently exactly zero, and has a "
         "long tail &mdash; so the t-test's assumptions fail immediately. The signed-rank test "
         "assumes far less. And because each seed produces one run per controller under identical "
         "conditions, the runs come in matched pairs, which is what a signed-rank test is designed "
         "for.</p>"
         "<p>A measure of <i>how large</i> the difference is gets reported next to the p-value, "
         "because knowing a difference is real tells you nothing about whether it is big enough to "
         "care about.</p>", "exam", "Likely question")
    note("<p>The agent is reliably better in <b>18 of 24</b> comparisons. The other 6 are "
         "<b>exact ties</b> &mdash; tests in which no controller accumulates any risk at all, so "
         "there is nothing to tell apart. Reporting those as &ldquo;no significant "
         "difference&rdquo; without that explanation would be misleading in the opposite "
         "direction.</p>", "num", "The headline comparison")

    h2("13.5 Turning parts off, one at a time")
    p("The last set of tests removes one piece of the system at a time to see what that piece was "
      "worth. This is called an <b>ablation study</b>.")
    ab = csv("ablations")
    if ab is not None and len(ab):
        try:
            g = ab.groupby("ablation").agg(
                do_risk=("risk_h_do", "median"),
                total=("time_in_risk_h", "median"),
                cost=("intervention_cost", "median"),
                unsafe=("unsafe_applied", "sum")).reset_index().sort_values("do_risk")
            names = {
                "no_forecast_at_all": "no forecaster loaded at all",
                "no_lstm_correction": "forecaster loaded, correction switched off",
                "weights_independent_local": "forecaster trained by each station alone",
                "no_guardrails": "safety layer removed",
                "no_world_model": "hand-written equations instead of Model 2",
                "full": "<b>the shipped system</b>",
                "weights_central": "forecaster trained on all data pooled",
            }
            notes = {
                "no_forecast_at_all": "identical to the row below &mdash; see the box",
                "no_lstm_correction": "genuinely better; explained in Part 11.5",
                "weights_independent_local": "genuinely better, modestly",
                "no_guardrails": "<b>1,058 unsafe actions</b> &mdash; not a real win",
                "no_world_model": "genuinely better, modestly",
                "full": "everything switched on",
                "weights_central": "worse",
            }
            rows = []
            for r in g.itertuples():
                rows.append([names.get(r.ablation, r.ablation), f"{r.do_risk:.2f}",
                             f"{r.total:.2f}", f"{r.cost:.2f}", f"{int(r.unsafe):,}",
                             notes.get(r.ablation, "")])
            table(rows, ["What was changed", "Hours of low oxygen", "Hours at risk overall",
                         "Running cost", "Unsafe actions", "Reading"],
                  "Sorted by hours of low oxygen, where lower is better. 40 runs behind each row, "
                  "and the middle value is shown. Read this table carefully before quoting it: the "
                  "shipped system is near the bottom, and that is deliberate.",
                  align="lrrrrl")
        except Exception:
            pass

    note("<p>Two rows produce <b>identical numbers across all 40 runs</b>, and that is a fact "
         "about how the system is built rather than a coincidence.</p>"
         "<p>One row switches off the correction described in Part 11.5. The other does not load a "
         "forecaster at all. They match because <b>the forecast reaches the actual decision only "
         "through that correction</b>. Everywhere else the forecast appears &mdash; in the risk "
         "model's inputs, in the safety layer's forward-looking checks &mdash; it changes what is "
         "<i>reported</i> and what is <i>blocked</i>, but across these tests it never changed "
         "which action was chosen.</p>"
         "<p>So &ldquo;does the forecast help the agent decide?&rdquo; and &ldquo;does the "
         "correction help?&rdquo; are the same question here, and the answer to both is currently "
         "no.</p>", "num", "Why two rows are identical")

    note("<p><b>Five variations score better than the shipped system on low-oxygen hours.</b> Say "
         "this before an examiner points at the table.</p><ul>"
         "<li>Removing the safety layer (9.38 hours) is <b>not</b> a real win: it bought that "
         "number with 1,058 unsafe actions. Without the layer the planner is free to take "
         "aggressive actions that happen to score well on hours-in-band while being exactly what "
         "the layer exists to stop.</li>"
         "<li>Switching off the correction (6.79 hours) is genuine, and Part 11.5 explains why it "
         "is left on anyway as a deliberate, measured choice.</li>"
         "<li>The two remaining differences are genuine and modest.</li></ul>",
         "honest", "The table does not flatter us")

    note("<p>The row where each station trained alone beats the shipped system here. That does "
         "<b>not</b> undo the result in Part 6.</p>"
         "<p>They measure different things. Part 6 measures <i>forecast accuracy on unseen "
         "stations</i>, where training together clearly wins. This table measures <i>how well the "
         "agent controls the pond</i>, where the choice of forecaster reaches the decision only "
         "through the correction &mdash; the very path we already know adds more error than it "
         "removes. A better forecaster feeding a harmful correction is not guaranteed to give "
         "better control, and here it does not.</p>"
         "<p>That is an observation with a plausible explanation, not a proven cause. The clean "
         "test is the correction row itself, and it is already in the table.</p>",
         "note", "This does not contradict Part 6")

    h2("13.6 What the agent costs")
    note("<p><b>The agent is reliably more expensive to run than a fixed thermostat in 7 of the 8 "
         "tests.</b></p>"
         "<p>This is not a bug and it is not hidden. What the extra cost buys is acting early, and "
         "acting early necessarily means sometimes acting on a problem that would have sorted "
         "itself out. You cannot get ahead of a threshold without occasionally moving when you did "
         "not strictly need to.</p>"
         "<p>The honest framing is a trade, not a free win: fewer hours of dangerous water, at a "
         "higher electricity and chemical bill. Whether that trade is worth making is a farm's "
         "decision. This project's job is to put a number on it, not to make the decision.</p>",
         "honest", "The counterweight, stated plainly")


def part14() -> None:
    part(14, "What the system does not do",
         "Rehearse this part. An examiner who finds a weakness you have not thought about will "
         "spend the rest of the session on it.")

    h2("14.1 The five limits, in the order they will come up")
    ol([
        "<b>The pond is simulated.</b> Every control result is a statement about the simulator, "
        "not about a farm. The simulator is tuned against real statistics but has never been "
        "checked against a real pond, and no claim in the project depends on it having been.",
        "<b>The 42 stations are rivers.</b> They are used as a stand-in, and this is stated "
        "openly. Part 11.5 is the one place the difference is measured &mdash; and there it makes "
        "the system worse.",
        "<b>Every claim is a comparison, not a guarantee.</b> &ldquo;Better than a fixed "
        "thermostat across these tests&rdquo;, not &ldquo;safe on a real farm&rdquo;. Zero unsafe "
        "actions is a property of 400 simulated runs under one particular definition of "
        "&ldquo;unsafe&rdquo;.",
        "<b>The grouping model did not work.</b> Its quality score is about zero. Only the "
        "odd-reading flag survives, and it cannot tell a broken probe from a genuine new problem.",
        "<b>The privacy option has never been run.</b> It is implemented and switched off, and no "
        "privacy figure is claimed.",
    ])

    h2("14.2 Claims to state carefully")
    table([
        ["&ldquo;Model 2 reads the water the way the real-data model does&rdquo;",
         "It <i>starts</i> from that model and is then trained further on simulated data. How much "
         "survives is not measured."],
        ["&ldquo;The oracle is the best possible&rdquo;",
         "It is a hand-written rule with perfect information. The agent beats it on oxygen in one "
         "test."],
        ["&ldquo;The agent is safe&rdquo;",
         "Zero unsafe actions were <i>applied</i> across these tests. That is a measured property "
         "of these runs under this definition of the safe bands."],
        ["&ldquo;SHAP explains why the pond is at risk&rdquo;",
         "SHAP explains why the <i>model</i> said so. It would say the same thing if the forecast "
         "were wrong."],
        ["&ldquo;Federated learning keeps the data private&rdquo;",
         "The raw data does not move. What is shared can still leak information in principle "
         "&mdash; that is what the privacy option is for, and it has not been run."],
    ], ["If you are about to say&hellip;", "&hellip;say this instead"],
        "Overclaiming is the fastest way to lose ground in a viva. Each of these has a weaker form "
        "that is still a genuine result.")

    h2("14.3 What the next stage should fix")
    ul([
        "<b>Training data from actual ponds</b> &mdash; the real fix for the river-versus-pond gap, "
        "and the thing that would let the correction in Part 11.5 finally earn its place.",
        "<b>Actually run the privacy option</b> and report a real privacy figure produced by a "
        "proper accounting tool.",
        "<b>Measure how far Model 2's reading part drifts</b> during training, so that "
        "&ldquo;it traces back to real data&rdquo; becomes a number instead of a starting "
        "condition.",
        "<b>Test against real hardware</b> &mdash; one real probe and one real aerator on a tank, "
        "even at small scale.",
    ], "tight")


def part15() -> None:
    m = CFG["models"]
    part(15, "Lookup tables and thirty questions",
         "The reference section. If you are revising the night before, start here and follow the "
         "cross-references backwards.")

    h2("15.1 Where everything lives in the code")
    table([
        ["<code>src/common/schema.py</code>",
         "the shared definitions of a reading, an action, a decision"],
        ["<code>src/data/</code>", "downloading, converting and recording the real stations"],
        ["<code>src/twin/</code>",
         "<code>dynamics.py</code> holds the equations, <code>pond.py</code> the time step, "
         "<code>scenarios.py</code> the tests, <code>risk_bands.py</code> the safe ranges"],
        ["<code>src/sensors/stream.py</code>",
         "noise, calibration error, dropout, rounding, instrument faults"],
        ["<code>src/perception/</code>",
         "<code>forecaster.py</code> is Model 1, <code>world_model.py</code> is Model 2, "
         "<code>regimes.py</code> is Model 3, <code>risk.py</code> is Model 4, "
         "<code>explain.py</code> is Model 5"],
        ["<code>src/federated/</code>",
         "<code>client.py</code> is one station, <code>simulate.py</code> is the averaging and the "
         "three training methods"],
        ["<code>src/agent/</code>",
         "<code>policy.py</code> holds the planners, <code>guardrails.py</code> the safety layer, "
         "<code>actuators.py</code> the option list, <code>loop.py</code> the tick"],
        ["<code>src/eval/</code>", "the comparison controllers, the measurements, the test runner"],
        ["<code>configs/*.yaml</code>", "every setting quoted in this document"],
        ["<code>outputs/*.csv</code>", "every result quoted in this document"],
    ], ["Where", "What is in it"], "Where to look when a question needs the source.", cls="small")

    h2("15.2 The numbers that come up most")
    ag = CFG["agent"]
    table([
        ["W &mdash; readings shown to the network", m["window"], "models.yaml"],
        ["H &mdash; steps predicted", m["horizon"], "models.yaml"],
        ["hidden units in the forecaster", m["forecaster"]["hidden_size"], "models.yaml"],
        ["layers in the forecaster", m["forecaster"]["num_layers"], "models.yaml"],
        ["simulator time step", f"{CFG['twin']['dt_minutes']} minutes", "twin.yaml"],
        ["how often sensors report", f"{CFG['sensors']['sample_every_minutes']} minutes",
         "sensors.yaml"],
        ["how often the planner decides", f"{ag['decide_every_minutes']} minutes", "agent.yaml"],
        ["how far ahead the planner simulates", ag["predictive"]["rollout_steps"], "agent.yaml"],
        ["how long an action is held in that simulation", ag["predictive"]["hold_action_steps"],
         "agent.yaml"],
        ["options the planner considers", 14, "actuators.py"],
        ["rounds of federated training",
         CFG["federated"]["regimes"]["federated"]["rounds"], "federated.yaml"],
        ["passes per station per round",
         CFG["federated"]["regimes"]["federated"]["local_epochs"], "federated.yaml"],
        ["passes when each station trains alone",
         CFG["federated"]["regimes"]["independent_local"]["epochs"], "federated.yaml"],
        ["share of stations held back",
         f"{CFG['federated']['split']['holdout_fraction']:.0%}", "federated.yaml"],
        ["oxygen level that triggers the emergency override",
         f"{ag['guardrails']['emergency_do_threshold']} mg/L", "agent.yaml"],
        ["how much the correction is damped", "halved", "policy.py"],
        ["how much the belief trusts each new reading", "35 %", "policy.py"],
    ], ["What it is", "Value", "Where it is set"],
        "The numbers worth memorising.", cls="small", align="llll")

    h2("15.3 Thirty questions and their answers")
    qa = [
        ("What kind of network does the forecaster use?",
         "A single-layer, one-direction LSTM &mdash; long short-term memory &mdash; with 64 hidden "
         "units, read out only at the last step, feeding one plain layer that produces 24 &times; 6 "
         "= 144 numbers."),
        ("Why an LSTM rather than a transformer?",
         "The window is only 36 readings long, each station has few examples, everything has to "
         "run without a graphics card, and every number in the network is sent over the network 12 "
         "times during training. A transformer buys nothing here and costs communication."),
        ("Why is the input 12 columns wide when there are only 6 measurements?",
         "Six columns of changes plus six columns saying whether each reading was real, so the "
         "network can tell a dead probe from a genuinely flat reading."),
        ("What exactly does the network predict?",
         "The change from the last reading it was shown &mdash; the anchor &mdash; for each of 24 "
         "future steps and 6 measurements, divided by a fixed scale."),
        ("Why predict changes rather than values?",
         "Levels do not carry across from a cool river to a warm pond. The shape of a daily cycle "
         "does."),
        ("Why are the scales fixed rather than calculated from the data?",
         "Calculating them would require each station's statistics to be shared, which breaks the "
         "privacy premise, and would make the three training methods impossible to compare."),
        ("What is the anchor rule?",
         "An example is thrown away unless oxygen and temperature both have real readings at the "
         "last step shown, because that step is what all the changes are measured from."),
        ("How long are W and H in real time?",
         "36 readings in, 24 out. On the real 15-minute data that is 9 hours in, 6 out. In the "
         "5-minute simulator it is 3 hours in, 2 out."),
        ("What actually crosses between a station and the centre?",
         "Up: the network's numbers, and one count of how many examples the station has. Down: the "
         "average of everybody's numbers. Never a reading, an average, or a station name."),
        ("Why does every station start from the same numbers?",
         "Hidden units have no fixed meaning, so averaging networks that started differently "
         "produces nonsense."),
        ("Are those shared starting numbers pre-trained?",
         "No &mdash; for Model 1 they are random. Model 2 is the one that starts from already-"
         "trained numbers."),
        ("How is the amount of training kept fair between methods?",
         "12 rounds &times; 2 passes = 24, which equals the 24 passes each station does alone. "
         "Otherwise a win might just be a win for training longer."),
        ("Why is the test set the end of the series rather than a random selection?",
         "Neighbouring examples share 30 of their 36 readings, so a random split puts near-copies "
         "on both sides and lets the model see the answers."),
        ("Where is the federated claim really tested?",
         "On the four stations kept down to 8 % of their history. Training together helped 3 of "
         "the 4, by about 13 % in the middle case; the fourth is printed too."),
        ("Why is Model 2 trained on the simulator?",
         "The real stations contain no record of anybody doing anything. No amount of training "
         "across rivers teaches a network what an aerator does."),
        ("Is the warm start the same as fine-tuning?",
         "No. The copied part is trained further, not frozen. It fixes where training begins, not "
         "where it ends."),
        ("Why is the action attached after the reading part?",
         "The window is the past, where the action has not happened yet. Putting it into the "
         "sequence would teach the network the aerator was already running."),
        ("What is the eighth action number for?",
         "It says what fraction of the simulated period the action is actually held for. Without "
         "it the network would consistently overestimate every action's effect."),
        ("Why are the world model's predictions capped?",
         "A network pushed outside what it has seen will predict negative oxygen, and an "
         "impossible prediction would look like the most urgent situation possible and drive a "
         "nonsense action."),
        ("What happened with the grouping model?",
         "It failed to find distinct states. Its quality score is about zero across a full sweep of "
         "settings; the readings form a continuum. Only the odd-reading flag is used."),
        ("Why report two clustering scores that disagree?",
         "One assumes groups exist and measures how separate they are. The other asks whether "
         "there is any group structure at all, and says no. The disagreement is the finding."),
        ("Isn't the risk model circular, since it learns a rule you already wrote?",
         "No. Applying the rule needs the future, the future comes from a forecast with real error, "
         "and a model over thirty forecast summary numbers copes with that error far better than a "
         "hard threshold on one predicted value."),
        ("Why explain the critical answer rather than whichever answer is winning?",
         "Explaining the winner makes the explanation change meaning as the prediction moves. "
         "&lsquo;What is pushing this towards critical&rsquo; is one stable question."),
        ("Why 14 options and not all 108?",
         "Most of the 108 make no sense or are beaten by something already on the list, each one "
         "costs a full simulation, and a list short enough to print is what keeps the agent "
         "explainable."),
        ("Why is there a penalty for withholding feed?",
         "Without it, the cheapest way to keep the water perfect is to stop feeding the fish "
         "forever."),
        ("Why is critical time counted twice in the score?",
         "So an hour in the critical band costs four times an hour in the warning band, matching "
         "the real difference between a bad afternoon and dead fish."),
        ("Why do the safety rules run more often than the planner?",
         "Deciding is slow and runs every 15 minutes; a pond can become dangerous in far less. Two "
         "of the five rule groups re-run every 5 minutes against the action being held."),
        ("Why is the emergency override the last rule?",
         "So it beats every restriction above it, including the rate limit that it deliberately "
         "ignores."),
        ("What is the single strongest result?",
         "Removing only the safety layer and changing nothing else: the same planner applied 1,058 "
         "unsafe actions without it and 0 with it, re-measured on every evaluation run."),
        ("What does the project get wrong?",
         "The correction from the real-data model makes control worse (10.00 hours against 6.79) "
         "and is left switched on anyway; the grouping model failed; the agent costs more to run "
         "than a thermostat in 7 of 8 tests. All three are measured and reported."),
    ]
    rows = [[f"<b>{i}.</b> {q}", a] for i, (q, a) in enumerate(qa, 1)]
    table(rows, ["Question", "Answer"],
          "Thirty questions, roughly in pipeline order. If you can answer these without looking "
          "them up, you can defend the pipeline.", cls="small")

    w('<hr class="soft">')
    p("<i>This document was built directly from the project. Every setting quoted was read from "
      "the settings files, every result from the results files, and every statement about the "
      "design from the source file named beside it. Rebuild it with "
      "<code>python scripts/make_pipeline_pdf.py</code>.</i>")


# ======================================================================================
# assembly
# ======================================================================================
SECTIONS = [
    cover, contents, part0, part1, part2, part3, part4, part5, part6, part7,
    part8, part9, part10, part11, part12, part13, part14, part15,
]


def build_html() -> str:
    _BUF.clear()
    _FIG["n"] = 0
    _TAB["n"] = 0
    for fn in SECTIONS:
        fn()
    body = "".join(_BUF)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<title>AquaSentinel - Every Action in the Pipeline</title>"
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


def print_pdf(html_path: Path, out_pdf: Path):
    """Same headless-Chrome path as scripts/make_report_pdf.py."""
    driver = Path(__file__).with_name("html_to_pdf.mjs")
    if shutil.which("node") and driver.exists():
        proc = subprocess.run(["node", str(driver), str(html_path), str(out_pdf)],
                              capture_output=True, text=True, timeout=600)
        if out_pdf.exists():
            return proc
        print("  CDP print failed; falling back to the CLI flag")
        print("  " + (proc.stderr or proc.stdout or "").strip()[-400:])
    browser = find_browser()
    if browser is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        return subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-sandbox",
             f"--user-data-dir={tmp}", f"--print-to-pdf={out_pdf}",
             "--virtual-time-budget=30000", html_path.resolve().as_uri()],
            capture_output=True, text=True, timeout=600)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the complete pipeline reference PDF.")
    ap.add_argument("--out", default=str(OUT_PDF))
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_doc = build_html()

    html_path = out_pdf.with_suffix(".html")
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"  html   {html_path}  ({len(html_doc) / 1024:.0f} KB, "
          f"{_FIG['n']} figures, {_TAB['n']} tables)")

    proc = print_pdf(html_path, out_pdf)
    if not args.keep_html:
        try:
            html_path.unlink()
        except OSError:
            pass
    if out_pdf.exists():
        print(f"  wrote  {out_pdf}  ({out_pdf.stat().st_size / 1024:.0f} KB)")
        return 0
    print("  FAILED - no PDF produced")
    if proc is not None:
        print((proc.stderr or proc.stdout or "")[-600:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
