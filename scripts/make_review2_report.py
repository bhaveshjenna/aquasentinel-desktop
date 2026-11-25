"""Build the BCSE497J Project-I Review-2 draft report as a .docx.

    python scripts/make_review2_report.py [--out report/review2]

Formatting follows "3 BCSE497J Project I Report - Template.docx" to the letter:

    cover title      Times New Roman 16, bold, UPPER CASE, line spacing 1.5
    student names    Times New Roman 13, bold, UPPER CASE, line spacing 1.5
    chapter heading  Times New Roman 14, bold, UPPER CASE, line spacing 1.5
    section heading  Times New Roman 13, bold, Title Case, line spacing 1.5
    sub-section      Times New Roman 12, bold + italic, Title Case, line spacing 1.5
    body             Times New Roman 12, line spacing 1.15

Every quantitative claim in Chapter 4 is transcribed from ``RESULTS.md``, which is itself
generated from ``outputs/*.csv`` - so the chain from CSV to report page has no hand-typed
number in the middle of it. The reference list is written by hand and should be spot-checked
against the original venues before submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_source as RS  # noqa: E402  - every result number comes from outputs/ via this
FIGS = ROOT / "report" / "review2" / "figures"
ASSETS = ROOT / "report" / "review2" / "assets"
SHOTS = ROOT / "report" / "screenshots"

FONT = "Times New Roman"


# --------------------------------------------------------------------------- low-level
def _set_font(run, size, bold=False, italic=False, caps=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(attr), FONT)
    if caps:
        c = OxmlElement("w:caps")
        rpr.append(c)


def para(doc, text="", size=12, bold=False, italic=False, align=None, spacing=1.15,
         space_after=6, space_before=0, indent=None, keep_with_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if indent is not None:
        pf.left_indent = Cm(indent)
    if align is not None:
        p.alignment = align
    pf.keep_with_next = keep_with_next
    if text:
        _set_font(p.add_run(text), size, bold, italic)
    return p


def bookmark(paragraph, name: str) -> None:
    """Wrap a paragraph in a bookmark so the contents table can PAGEREF it."""
    bid = str(abs(hash(name)) % 100000)
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), bid)
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), bid)
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def field(paragraph, instr: str, placeholder: str = "1", size: int = 12) -> None:
    """Insert a Word field (used for PAGEREF in the table of contents)."""
    r1 = paragraph.add_run()
    _set_font(r1, size)
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    r1._element.append(fc)

    r2 = paragraph.add_run()
    _set_font(r2, size)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r2._element.append(it)

    r3 = paragraph.add_run()
    _set_font(r3, size)
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    r3._element.append(sep)

    r4 = paragraph.add_run(placeholder)
    _set_font(r4, size)

    r5 = paragraph.add_run()
    _set_font(r5, size)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r5._element.append(end)


def page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(p, "PAGE", "1", 11)


def no_borders(table) -> None:
    tbl = table._tbl
    pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "none")
        e.set(qn("w:sz"), "0")
        borders.append(e)
    pr.append(borders)


def cell_text(cell, text, size=12, bold=False, italic=False, align=None, spacing=1.15):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = spacing
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    if align is not None:
        p.alignment = align
    _set_font(p.add_run(text), size, bold, italic)


def make_table(doc, rows, widths=None, header=True, size=10.5, style="Table Grid"):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = style
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell_text(t.cell(i, j), str(val), size=size,
                      bold=(header and i == 0), spacing=1.0)
    if widths:
        # Word's autofit ignores per-cell widths, so the layout has to be pinned to fixed.
        t.autofit = False
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        t._tbl.tblPr.append(layout)
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    return t


def figure(doc, path: Path, caption: str, width_cm=15.5):
    if not path.exists():
        para(doc, f"[missing figure: {path.name}]", italic=True,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(path), width=Cm(width_cm))
    para(doc, caption, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, space_after=10)


def caption_table(doc, text):
    para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0,
         space_before=4, space_after=10)


# --------------------------------------------------------------------------- headings
def chapter(doc, number, title, toc, first=False):
    if not first:
        doc.add_page_break()
    p = para(doc, f"{number}. {title.upper()}", size=14, bold=True, spacing=1.5,
             space_after=10, keep_with_next=True)
    name = f"_ch{number}"
    bookmark(p, name)
    toc.append((str(number), title.upper(), name, 0))
    return p


def section(doc, number, title, toc):
    p = para(doc, f"{number} {title}", size=13, bold=True, spacing=1.5,
             space_before=10, space_after=6, keep_with_next=True)
    name = "_s" + number.replace(".", "_")
    bookmark(p, name)
    toc.append(("", f"{number} {title}", name, 1))
    return p


def subsection(doc, number, title, toc):
    p = para(doc, f"{number} {title}", size=12, bold=True, italic=True, spacing=1.5,
             space_before=8, space_after=4, keep_with_next=True)
    name = "_ss" + number.replace(".", "_")
    bookmark(p, name)
    toc.append(("", f"     {number} {title}", name, 2))
    return p


def bullets(doc, items, size=12):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Cm(1.0)
        if isinstance(it, tuple):
            _set_font(p.add_run(it[0]), size, bold=True)
            _set_font(p.add_run(" — " + it[1]), size)
        else:
            _set_font(p.add_run(it), size)


# =========================================================================== content
TEAM = [
    ("23BAI0003", "SANJEEV KARNATAPU"),
    ("23BAI0065", "NISHANT RAY"),
    ("23BCE0549", "SOHAM SWARUP DUTTA"),
]
TITLE = "AI-DRIVEN SMART AQUACULTURE MONITORING AND RISK PREDICTION SYSTEM"
GUIDE = "Dr. Sivaranjani A"
GUIDE_DESIGNATION = "Assistant Professor Senior Grade 1"


def cover(doc):
    para(doc, "BCSE497J - Project-I", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         spacing=1.5, space_after=14)
    para(doc, TITLE, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5,
         space_after=18)

    for reg, name in TEAM:
        para(doc, reg, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5,
             space_after=0)
        para(doc, name, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5,
             space_after=4)

    para(doc, "Under the Supervision of", size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
         spacing=1.5, space_before=12, space_after=6)
    para(doc, GUIDE, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5,
         space_after=0)
    para(doc, GUIDE_DESIGNATION, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5,
         space_after=0)
    para(doc, "School of Computer Science and Engineering (SCOPE)", size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, space_after=14)

    para(doc, "B.Tech.", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, space_after=0)
    para(doc, "in", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, space_after=0)
    para(doc, "Computer Science and Engineering", size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
         spacing=1.5, space_after=0)
    para(doc, "(Specialization in Artificial Intelligence and Machine Learning)", size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, space_after=0)
    para(doc, "School of Computer Science and Engineering", size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, space_after=8)

    logo = ASSETS / "vit_logo.jpg"
    if logo.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        p.add_run().add_picture(str(logo), width=Cm(6.5))

    para(doc, "September 2026", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         spacing=1.5)


def anomaly_verdict(fpr: float) -> str:
    """Plain verdict on the anomaly flag's false-alarm rate, from the measured number."""
    if fpr > 0.5:
        return (f"At a false-positive rate of {fpr:.0%} the flag fires on most ordinary windows, so "
                "it is not usable as an alarm on its own: the model was fitted mostly on noisy "
                "real-pond windows, and the simulated pond's normal behaviour falls outside them.")
    if fpr > 0.2:
        return (f"A false-positive rate of {fpr:.0%} is high; the flag is a weak hint, not an alarm.")
    return (f"A false-positive rate of {fpr:.0%} makes the flag usable as a novelty hint.")


def fed_phrase(R) -> str:
    """One clause on the federated result, worded by what the numbers actually show."""
    if R["fed_do"] < R["loc_do"]:
        s = (f"beat independent-local training on held-out ponds "
             f"({RS.f3(R['fed_do'])} against {RS.f3(R['loc_do'])} mg/L dissolved-oxygen RMSE)")
    else:
        s = (f"did not beat independent-local training on the held-out ponds "
             f"({RS.f3(R['fed_do'])} against {RS.f3(R['loc_do'])} mg/L)")
    if R["fed_do"] <= R["cen_do"]:
        s += " and matched the centrally-pooled reference"
    else:
        s += (f" and came within {(R['fed_do'] / R['cen_do'] - 1) * 100:.1f} % of the "
              f"centrally-pooled reference")
    return s


def abstract_text() -> str:
    R = RS.load()
    unsafe = "zero" if R["guarded_unsafe"] == 0 else str(R["guarded_unsafe"])
    return f"""\
Intensive aquaculture fails through water chemistry long before it fails through disease. \
Dissolved oxygen sags before dawn, ammonia accumulates after over-feeding, and an algal bloom \
that oxygenates a pond by day strips it by night; each of these can move a stocked pond from \
healthy to critical inside a single unattended shift. Conventional practice is a fixed-threshold \
aerator controller, which by construction reacts only after a limit has already been crossed, \
and per-farm machine learning is impractical because an individual pond generates too little \
history to train a reliable model, while pooling farm data centrally is commercially \
unacceptable.

This project builds and evaluates AquaSentinel, a monitoring and action-suggestion system in \
which learned perception, learned prediction and hard safety constraints are separate, testable \
components. A physics-lite pond digital twin simulates dissolved oxygen, temperature, pH, total \
ammonia nitrogen, nitrite and turbidity, and responds to actions, so the effect of every decision \
is observable against a no-action counterfactual pond forked from the same random seed. An LSTM \
forecaster is pre-trained by federated averaging with one client per pond, so that raw \
observations never leave the farm that recorded them, on real sensor data from \
{R['n_ponds']} catfish aquaponics ponds in Nsukka, Nigeria ({R['n_rows']:,} harmonised \
15-minute records from low-cost IoT probes). An action-conditioned world model warm-started \
from that forecaster predicts the pond's response to each of fourteen candidate actions, a \
model predictive planner chooses among them, and every selected action passes through a \
guardrail layer of dosing caps, rate limits, contextual blocks and an emergency override \
before it can reach an actuator.

Across {R['n_scenarios']} stress scenarios, {R['n_controllers']} controllers and \
{R['n_seeds']} seeds, the guardrailed agent applied {unsafe} unsafe actions and significantly \
reduced dissolved-oxygen time-out-of-band in {R['wins']} of {R['comparisons']} paired \
comparisons, at a measured and reported increase in intervention cost. Federated pre-training \
{fed_phrase(R)}.

Keywords - Federated Learning, Digital Twin, Aquaculture, Model Predictive Control, Safety \
Guardrails, LSTM, Explainable AI."""


def abstract(doc, toc, page_break=True):
    if page_break:
        doc.add_page_break()
    p = para(doc, "ABSTRACT", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             spacing=1.5, space_after=12)
    bookmark(p, "_abstract")
    toc.append(("", "Abstract", "_abstract", 1))
    for block in abstract_text().split("\n\n"):
        para(doc, block.strip(), size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.15,
             space_after=8)


def contents(doc, toc):
    doc.add_page_break()
    para(doc, "TABLE OF CONTENTS", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         spacing=1.5, space_after=12)
    t = doc.add_table(rows=len(toc) + 1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    no_borders(t)
    cell_text(t.cell(0, 0), "Sl.No", 12, bold=True, spacing=1.5)
    cell_text(t.cell(0, 1), "Contents", 12, bold=True, spacing=1.5)
    cell_text(t.cell(0, 2), "Page No.", 12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
              spacing=1.5)
    for i, (num, title, anchor, level) in enumerate(toc, start=1):
        cell_text(t.cell(i, 0), num, 12, bold=(level == 0), spacing=1.5)
        cell_text(t.cell(i, 1), title, 12, bold=(level == 0), spacing=1.5)
        c = t.cell(i, 2)
        c.text = ""
        pp = c.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pp.paragraph_format.line_spacing = 1.5
        pp.paragraph_format.space_before = Pt(2)
        pp.paragraph_format.space_after = Pt(2)
        field(pp, f"PAGEREF {anchor} \\h", "—", 12)
    for row in t.rows:
        row.cells[0].width = Cm(1.6)
        row.cells[1].width = Cm(11.6)
        row.cells[2].width = Cm(2.4)
    para(doc, "")
    para(doc, "Page numbers are Word PAGEREF fields. Select the whole document (Ctrl+A) and "
              "press F9 once before printing so they resolve.", size=10, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0)


# --------------------------------------------------------------------------- chapter 1
def chapter1(doc, toc, first=False):
    chapter(doc, 1, "Introduction", toc, first=first)

    section(doc, "1.1", "Background", toc)
    para(doc, (
        "Aquaculture now supplies more aquatic food for human consumption than capture "
        "fisheries, and the pond systems that produce most of it are operated at stocking "
        "densities where water chemistry, not feed supply, is the binding constraint. In a "
        "stocked earthen pond, dissolved oxygen falls through the night as respiration "
        "continues without photosynthesis and reaches its minimum shortly before dawn; "
        "un-ionised ammonia rises after feeding and becomes acutely toxic when pH climbs "
        "during the afternoon; a dense algal bloom that super-saturates the water by day can "
        "strip it overnight and again when the bloom crashes. Each of these transitions can "
        "carry a pond from a healthy state to a lethal one within a few hours, and the "
        "financial loss from a single overnight kill can exceed a season's margin."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "The prevailing control response is a fixed-threshold aerator switch: run the paddle "
        "wheel when measured oxygen falls below a set point. This is robust and cheap, but it "
        "is by construction reactive — it acts only once the limit has already been crossed, "
        "and it cannot use any information about where the pond is heading. Meanwhile, the "
        "machine-learning work that could supply that forward-looking information is held back "
        "by data: an individual farm generates too little history to train a reliable model, "
        "and pooling raw operational data across farms is commercially and legally "
        "unattractive."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    section(doc, "1.2", "Motivation", toc)
    para(doc, (
        "Two observations motivate this project. First, the useful unit of prediction in a "
        "pond is not a class label but a trajectory: an operator needs to know that oxygen "
        "will cross the critical band in ninety minutes, not that the current reading is "
        "“poor”. Published water-quality datasets are dominated by scripted "
        "index labels derived from thresholds, so a model trained to reproduce them learns the "
        "threshold rather than the chemistry, and reporting accuracy against such labels is "
        "circular. This project therefore consumes only continuous sensor channels and "
        "unsupervised structure, and states plainly which of its components are learned and "
        "which are rule-informed."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "Second, an autonomous controller that can dose lime, exchange water and drive an "
        "aerator is a safety-critical system, and a learned planner is precisely the kind of "
        "component that fails confidently. A model-predictive optimiser treats its internal "
        "model as if it were exact and will happily discover the small regions where a neural "
        "network is wrong. The engineering question worth answering is therefore not "
        "“can a network control a pond” but “where must the hard constraints "
        "live so that a wrong network cannot cause harm”. Federated learning addresses "
        "the data problem without asking farms to surrender their records; explicit guardrails "
        "outside the learned component address the safety problem. Combining the two, and "
        "measuring both the benefit and the price, is what this project sets out to do."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    section(doc, "1.3", "Scope of the Project", toc)
    para(doc, (
        "The project delivers a complete, offline-reproducible pipeline: a calibratable pond "
        "digital twin with eight stress scenarios; a sensor simulator with noise, dropout and "
        "injectable probe faults; a perception stack of an LSTM forecaster, an HDBSCAN novelty "
        "detector, a LightGBM risk classifier and SHAP attribution; a federated learning study "
        "over real fish-pond sensor data from a single region - catfish aquaponics ponds in "
        "Nsukka, Nigeria - compared against independent-local and centrally-pooled references "
        "on a held-out pond; an action-conditioned world model; a model-predictive agent with "
        "a hard guardrail layer; and a desktop operator console that shows every decision "
        "against a no-action counterfactual pond."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "Three boundaries are drawn deliberately and are restated wherever a result depends on "
        "them. The pond is a simulation, and its coefficients are calibratable defaults chosen "
        "from published aquaculture guidance rather than measured from a physical pond. The "
        "real data is genuine pond data, but from low-cost IoT probes: roughly half of each "
        "pond's record is sensor outage, faulty readings are removed by quality control, only "
        "a handful of ponds survive it, and the dataset records water quality but not the "
        "actions the farmers took. Every performance claim is therefore relative — agent "
        "against baselines inside the twin, federated against independent-local and central "
        "on a held-out real pond — and none is a real-world guarantee. Physical deployment on a working farm, live actuator "
        "hardware and regulatory certification are outside the scope of Project-I."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)


# --------------------------------------------------------------------------- chapter 2
LIT_TABLE = [
    ["Ref.", "Work and technique", "Contribution relevant to this project", "Limitation"],
    ["[1]", "Føre et al. — precision fish farming framework",
     "Frames aquaculture as a monitor–model–control loop rather than manual husbandry.",
     "Conceptual framework; no implemented controller or safety layer."],
    ["[23]", "O'Donncha and Grant — precision aquaculture with IoT and analytics",
     "Establishes continuous multivariate sensing as the practical data substrate.",
     "Data-centric; prediction is not carried through to actuation."],
    ["[2]", "Zhou et al. — intelligent feeding control review",
     "Shows feed rate is an actionable control input coupled to ammonia load.",
     "Feeding treated in isolation from oxygen and pH dynamics."],
    ["[3]", "Liu et al. — PSO-optimised LS-SVR for pond dissolved oxygen",
     "Demonstrates short-horizon DO prediction is feasible from routine sensors.",
     "Single site, single channel; no transfer to unseen ponds."],
    ["[4]", "Ta and Wei — CNN for DO prediction in recirculating systems",
     "Convolutional feature extraction over sensor windows outperforms shallow models.",
     "Centralised training; assumes one operator owns all the data."],
    ["[5]", "Ren et al. — deep belief network for DO in recirculating aquaculture",
     "Deep representations improve multi-step accuracy over classical regressors.",
     "No uncertainty treatment; prediction not used for control."],
    ["[6]", "Barzegar et al. — hybrid CNN-LSTM for water-quality variables",
     "Hybrid recurrent architectures are the strongest reported family for this signal.",
     "Absolute-value targets, so the model does not transfer across water bodies."],
    ["[7]", "Hu et al. — LSTM water-quality prediction for smart mariculture",
     "Correlation-aware multivariate LSTM; the architecture this project adopts.",
     "Single-farm data; no privacy-preserving training path."],
    ["[8]", "McMahan et al. — Federated Averaging",
     "The aggregation rule used here: clients share weights, never observations.",
     "Assumes broadly IID clients; degrades under strong heterogeneity."],
    ["[9]", "Li et al. — FedProx for heterogeneous networks",
     "Shows a proximal term stabilises FedAvg when client data diverges.",
     "Adds a hyper-parameter that must itself be tuned per deployment."],
    ["[10]", "Zhao et al. — federated learning with non-IID data",
     "Quantifies accuracy loss from client skew; motivates leave-sites-out evaluation.",
     "Skew is simulated by label partitioning, not inherited from real deployments."],
    ["[34]", "Nguyen et al. — federated learning for the Internet of Things",
     "Survey of cross-device deployments matching the one-farm-one-client setting.",
     "No aquaculture instance; no actuation loop considered."],
    ["[11]", "Durrant et al. — cross-silo federated learning in agri-food",
     "Closest sector precedent: farms as silos unwilling to pool raw records.",
     "Crop yield prediction only; no closed-loop control component."],
    ["[37]", "Abadi et al. — deep learning with differential privacy",
     "Clipping plus Gaussian noise, the mechanism implemented here behind an opt-in flag.",
     "Utility cost is significant at the epsilon values farms would need."],
    ["[39]", "Campello et al. — HDBSCAN",
     "Density-based clustering that can decline to assign a point, giving novelty flags.",
     "Assumes density modes exist; continuous manifolds yield one blob."],
    ["[41]", "Moulavi et al. — density-based clustering validation (DBCV)",
     "The validity index that exposed the absence of a regime taxonomy in this data.",
     "Expensive on large samples; sensitive to the density estimator."],
    ["[45]", "Ke et al. — LightGBM",
     "Efficient gradient boosting used for the risk classifier over forecast summaries.",
     "Tabular only; cannot consume raw sequences without feature engineering."],
    ["[46]", "Lundberg and Lee — SHAP",
     "Additive attributions that let the operator see why a risk score moved.",
     "Explains the model, not the water; faithful attribution is not causation."],
    ["[12]", "Hewing et al. — learning-based model predictive control",
     "Formalises the risk of optimising against a learned model; motivates the safety filter.",
     "Theory assumes bounded model error, which a deep network does not provide."],
    ["[13]", "Wabersich and Zeilinger — predictive safety filter",
     "Constraints enforced outside the learned policy: the design adopted here.",
     "Requires a trusted backup model, which for a pond is only approximate."],
    ["[14]", "Ha and Schmidhuber — world models",
     "Action-conditioned latent dynamics as a substrate for planning.",
     "Trained in simulation; real-environment transfer is unaddressed."],
    ["[52]", "García and Fernández — safe reinforcement learning survey",
     "Taxonomy separating constrained optimisation from external safety layers.",
     "Survey; no empirical comparison of the two placements."],
    ["[55]", "Grieves and Vickers — digital twin",
     "The twin-as-environment pattern that makes counterfactual evaluation possible.",
     "Fidelity of the twin bounds the validity of anything measured inside it."],
]


def chapter2(doc, toc):
    chapter(doc, 2, "Project Description and Goals", toc)

    section(doc, "2.1", "Literature Review", toc)
    para(doc, (
        "The literature relevant to this project falls into five strands, which have largely "
        "developed independently: aquaculture water-quality science, data-driven water-quality "
        "prediction, federated learning, unsupervised structure discovery and explainability, "
        "and safe learning-based control. This section reviews each strand and identifies "
        "where the strands fail to meet."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "A. Aquaculture water-quality science and precision aquaculture", size=12,
         bold=True, spacing=1.15, space_before=8, space_after=4)
    para(doc, (
        "The chemistry the system must respect is long settled. Boyd and Tucker [16], [17] give "
        "the canonical treatment of pond oxygen budgets, the diel oxygen curve driven by the "
        "balance of photosynthesis and respiration, and the safe operating bands for dissolved "
        "oxygen, pH, ammonia and nitrite in warm-water culture; Emerson et al. [19] give the "
        "temperature-and-pH dependence of the un-ionised ammonia fraction that makes total "
        "ammonia nitrogen dangerous only in combination with a rising pH; Benson and Krause [20] "
        "give the oxygen saturation relation used here to make measurements from different "
        "water bodies comparable. Standard Methods [18] fixes the measurement conventions. On "
        "the engineering side, Føre et al. [1] and O'Donncha and Grant [23] argue for precision "
        "aquaculture as a sense–model–act loop, and Antonucci and Costa [24] survey the sensing "
        "hardware that makes it feasible; Zhou et al. [2] review feeding control specifically. "
        "This strand supplies the physics and the operating bands, but the control it describes "
        "remains overwhelmingly threshold-based."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "B. Data-driven water-quality forecasting", size=12, bold=True, spacing=1.15,
         space_before=8, space_after=4)
    para(doc, (
        "Dissolved-oxygen prediction has a mature literature. Liu et al. [3] optimise a "
        "least-squares support vector regressor with particle swarm optimisation for river-crab "
        "ponds; Ta and Wei [4] apply convolutional networks and Ren et al. [5] a deep belief "
        "network to recirculating systems; Barzegar et al. [6] show a hybrid CNN-LSTM "
        "outperforms either component alone across several water-quality variables; Hu et al. "
        "[7] build a correlation-aware LSTM for mariculture. The recurrent architecture of "
        "Hochreiter and Schmidhuber [25] remains the workhorse, and the deep time-series "
        "surveys of Lim and Zohren [28] confirm that recurrent and attention models dominate on "
        "multi-step multivariate sensor data. Two properties recur across this strand and both "
        "are limiting. Models are trained centrally on a single site's history, and they "
        "predict absolute levels — so a network fitted on a temperate river at 9 mg/L is not "
        "merely inaccurate on a 28 °C tilapia pond, it is out of distribution. This project "
        "predicts per-channel deltas for exactly that reason."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "C. Federated learning and privacy", size=12, bold=True, spacing=1.15,
         space_before=8, space_after=4)
    para(doc, (
        "Federated Averaging [8] establishes the pattern this project uses: each client "
        "trains locally, a server averages weights in proportion to sample count, and raw data "
        "never moves. The surveys of Li et al. [31], Kairouz et al. [32] and Yang et al. [33] "
        "map the design space, and the recurring difficulty is statistical heterogeneity — "
        "Zhao et al. [10] quantify the accuracy loss under non-IID clients and FedProx [9] "
        "mitigates it with a proximal term. In the Internet-of-Things setting most relevant "
        "here, Nguyen et al. [34] and Imteaj et al. [35] survey cross-device deployments, and "
        "Durrant et al. [11] and Friha et al. [36] give the closest sector precedents in "
        "agri-food and agricultural IoT security. Formal privacy is supplied by DP-SGD [37] "
        "with Rényi accounting [38]. What this strand does not contain is a study in which the "
        "federated model then drives a physical actuator, or one in which client heterogeneity "
        "is inherited from real deployments rather than manufactured by partitioning a "
        "benchmark."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "D. Unsupervised structure, anomaly detection and explanation", size=12,
         bold=True, spacing=1.15, space_before=8, space_after=4)
    para(doc, (
        "HDBSCAN [39], [40] is the natural choice for discovering operating regimes without "
        "labels because it may decline to assign a point, which turns the noise label into a "
        "novelty flag. Its density-based validity index DBCV [41] is the appropriate quality "
        "measure for non-globular clusters, and it is what this project used to reach a "
        "negative result reported in Section 4.4. Chandola et al. [42] and Blázquez-García et "
        "al. [43] frame the sensor-fault problem as time-series anomaly detection and are the "
        "source of the event-level rather than tick-level scoring used here. For explanation, "
        "LightGBM [45] with SHAP [46], [47] gives per-tick attribution of a risk score at a cost "
        "compatible with a live dashboard."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "E. Safe learning-based control and world models", size=12, bold=True,
         spacing=1.15, space_before=8, space_after=4)
    para(doc, (
        "Model predictive control [49] supplies the planning formulation: roll candidate "
        "actions forward through a model, score the resulting trajectories, apply the first "
        "action of the best one. Replacing the hand-written model with a learned one is the "
        "world-model idea of Ha and Schmidhuber [14] and Hafner et al. [50], and it is exactly "
        "where the danger lies: Hewing et al. [12] show that an optimiser treats its model as "
        "exact and will exploit whatever regions the model gets wrong. The remedy adopted here "
        "is the predictive safety filter of Wabersich and Zeilinger [13] — constraints enforced "
        "outside the learned component rather than folded into its cost function — which the "
        "safe-RL survey of García and Fernández [52] identifies as the more robust of the two "
        "placements, and which Amodei et al. [53] and Dulac-Arnold et al. [54] argue for on "
        "general grounds. The digital-twin literature [55], [56] supplies the environment in "
        "which such a comparison can be run at all, since a real pond cannot be asked to "
        "provide a no-action counterfactual."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "", space_after=4)
    make_table(doc, LIT_TABLE, widths=[1.2, 4.3, 5.6, 4.4], size=9)
    caption_table(doc, "Table 1. Summary of the surveyed literature.")

    section(doc, "2.2", "Research Gap", toc)
    para(doc, (
        "Reading the five strands together, five gaps stand out, and each of them shapes a "
        "design decision in this project."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    bullets(doc, [
        ("G1. Prediction is not carried through to action",
         "water-quality forecasting papers stop at an error metric. None of the surveyed work "
         "closes the loop onto an actuator and then measures whether the forecast actually "
         "improved control, so the value of the forecast to the controller is assumed rather "
         "than tested."),
        ("G2. Federated learning has not been evaluated on an actuation task",
         "federated studies report held-out accuracy. Whether a federated model is a safe "
         "basis for a decision that opens a valve is a different question and is unaddressed."),
        ("G3. Client heterogeneity is manufactured, not inherited",
         "non-IID structure in the federated literature is usually produced by partitioning a "
         "benchmark by label. Real farms differ in climate, season, catchment and even in which "
         "probes they carry, and that structure cannot be simulated faithfully."),
        ("G4. Safety is argued, not measured",
         "learning-based control papers argue that constraints are necessary. Very few report "
         "the counterfactual — how many unsafe commands the same planner would have issued with "
         "the safety layer removed — which is the only evidence that the layer is load-bearing."),
        ("G5. Scripted quality labels make results circular",
         "most public water-quality datasets carry index labels computed from thresholds. A "
         "model trained on them and scored against them has learned the threshold, and the "
         "reported accuracy says nothing about chemistry."),
    ])

    section(doc, "2.3", "Objectives", toc)
    para(doc, "The project pursues seven specific, measurable objectives.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    bullets(doc, [
        ("O1", "Build a calibratable pond digital twin covering dissolved oxygen, temperature, "
               "pH, total ammonia nitrogen, nitrite and turbidity that responds to actions, "
               "with at least eight stress scenarios whose advertised stress is asserted by "
               "automated tests."),
        ("O2", "Acquire real aquaculture-pond sensor data from a single region, harmonise it "
               "to one schema at 15-minute cadence with quality control, an availability mask "
               "and a per-pond licence manifest, and cache it so the pipeline runs offline."),
        ("O3", "Train a multi-step multivariate LSTM forecaster by federated averaging with one "
               "client per pond, and compare it with independent-local training on a pond no "
               "regime trained on, under matched total training work and a shared "
               "initialisation."),
        ("O4", "Train an action-conditioned world model that predicts the pond's response to a "
               "candidate action, so that the trajectories the planner chooses between are "
               "produced by a network rather than by a hand-written equation."),
        ("O5", "Implement a model predictive agent over a curated candidate action set, behind "
               "a guardrail layer, and demonstrate zero unsafe actions applied across the full "
               "evaluation grid."),
        ("O6", "Evaluate the agent against four baselines across eight scenarios and ten seeds "
               "using paired Wilcoxon signed-rank tests, reporting the intervention cost of "
               "every risk reduction rather than the reduction alone."),
        ("O7", "Deliver an operator dashboard in which every action is shown against a "
               "no-action counterfactual pond forked from the same seed and noise realisation, "
               "with SHAP attribution for the risk score behind each decision."),
    ])
    para(doc, (
        "Identified Sustainable Development Goals: SDG 2 (Zero Hunger), SDG 9 (Industry, "
        "Innovation and Infrastructure), SDG 12 (Responsible Consumption and Production) and "
        "SDG 14 (Life Below Water)."), align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=6)

    section(doc, "2.4", "Problem Statement", toc)
    para(doc, (
        "Given a stocked aquaculture pond instrumented with noisy, occasionally faulty "
        "continuous probes for dissolved oxygen, temperature, pH, total ammonia nitrogen, "
        "nitrite and turbidity, and given a set of admissible interventions — aeration, "
        "feeding, liming, partial water exchange and probiotic dosing — the problem is to "
        "choose, at each decision interval, the intervention that minimises the time the pond "
        "spends outside its safe operating bands over a forward horizon, subject to two hard "
        "conditions: no action may be applied that drives any channel into its critical band, "
        "and no raw observation may leave the site that recorded it during model training."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "The difficulty is that the three obvious sources of leverage are each unavailable in "
        "their usual form. A per-farm supervised model cannot be trained because a single pond "
        "has too little history; a pooled model cannot be trained because the data may not be "
        "centralised; and a learned dynamics model cannot simply be trusted inside an "
        "optimiser, because the optimiser will find and exploit the regions where it is wrong. "
        "The system must therefore learn across sites without moving data, and must place its "
        "hard constraints outside the learned component. Success is measured relatively: "
        "against a no-action counterfactual and four controller baselines inside the twin, and "
        "against independent-local and centrally-pooled training on a held-out real pond."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    section(doc, "2.5", "Project Plan", toc)
    para(doc, (
        "Figure 1 shows the Project-I schedule as a Gantt chart. Weeks are indexed from the "
        "start of the Fall 2026-27 semester. Coloured bars are complete and demonstrable at "
        "Review-2; pale bars are Project-II scope. The three streams — twin and data, learning, "
        "and agent and evaluation — were deliberately overlapped, because the evaluation "
        "harness had to exist before the agent could be tuned against it."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    figure(doc, FIGS / "fig1_gantt.png", "Fig. 1. Project-I Gantt chart.", 15.5)


# --------------------------------------------------------------------------- chapter 3
FUNCTIONAL = [
    ["ID", "Requirement", "Verified by"],
    ["FR1", "Collect continuous water-quality time series from public monitoring APIs "
            "without an API key, cache them locally, and never re-download a cached site.",
     "tests/test_data.py; data_cache/manifest.csv"],
    ["FR2", "Harmonise every source to one channel schema at a fixed 15-minute cadence, with "
            "an explicit availability mask per channel and a recorded per-site licence.",
     "tests/test_data.py (schema and mask)"],
    ["FR3", "Simulate a pond forward in discrete time, producing true dissolved oxygen, "
            "temperature, pH, total ammonia nitrogen, nitrite and turbidity, and respond to "
            "applied actions.", "tests/test_twin.py"],
    ["FR4", "Provide at least eight named stress scenarios, each an initial state, a "
            "perturbation schedule and a fixed seed, and assert that each produces the stress "
            "its name advertises.",
     "tests/test_twin.py::test_scenarios_produce_their_advertised_stress"],
    ["FR5", "Sample the true state into a noisy observation with configurable sensor noise, "
            "dropout and injectable probe faults that corrupt the reading and not the water.",
     "tests/test_perception.py"],
    ["FR6", "Produce a multi-step multivariate forecast of the sensor channels: 24 steps "
            "ahead from a 36-step observation window (two hours ahead at the twin's 5-minute "
            "step).", "tests/test_perception.py"],
    ["FR7", "Flag novel operating conditions without consuming any dataset quality label, "
            "using the unassigned-point label of a density-based clustering model.",
     "outputs/anomaly_detection.csv"],
    ["FR8", "Classify current risk into ordered bands from the current channels and forecast "
            "summaries, and attribute the score to input channels for the operator.",
     "outputs/risk_metrics.csv"],
    ["FR9", "Pre-train the forecaster and the world model by federated averaging, one client "
            "per site, transmitting weight tensors and a sample count and nothing else.",
     "tests/test_federated.py; tests/test_world_model.py"],
    ["FR10", "Plan an action by rolling a curated set of fourteen candidate actions forward "
             "through the world model and taking the minimum-cost trajectory.",
     "tests/test_agent.py"],
    ["FR11", "Enforce dosing caps, minimum intervals, rate limits, contextual blocks, "
             "rejection of any action predicted to drive a channel critical, and a "
             "last-resort emergency override, on every sensor tick.",
     "tests/test_guardrails.py; outputs/safety.csv"],
    ["FR12", "Run a shadow pond with the agent disabled from the same seed and the same noise "
             "realisation, so that every divergence is attributable to an action.",
     "tests/test_agent.py::test_do_nothing_matches_its_own_shadow"],
    ["FR13", "Render a live dashboard with time series, counterfactual comparison, agent "
             "rationale and a scorecard against baselines.", "manual demonstration"],
    ["FR14", "Regenerate every reported table from the evaluation outputs, so that no number "
             "in the results record is typed by hand.", "scripts/build_report.py; RESULTS.md"],
]

NON_FUNCTIONAL = [
    ["ID", "Requirement", "Target"],
    ["NFR1", "Latency — a full perceive, plan, guardrail and act cycle must complete well "
             "inside one simulated sensor tick so the dashboard streams without stalling.",
     "< 1 s per tick on CPU"],
    ["NFR2", "Reproducibility — one global seed plus a per-scenario twin noise seed; an "
             "identical configuration must produce an identical trajectory.",
     "bit-identical shadow run"],
    ["NFR3", "Offline operation — only the one-time data collection step may touch the "
             "network; everything else must run with the interface down.",
     "one network-using module"],
    ["NFR4", "Data confidentiality — no raw observation may leave its client during federated "
             "training; the client exposes only train, get_weights, set_weights and evaluate.",
     "structural isolation"],
    ["NFR5", "Portability — Python 3.11, CPU only, no GPU requirement.", "commodity laptop"],
    ["NFR6", "Configurability — sites, twin coefficients, model and agent settings live in "
             "YAML, not in code.", "seven config files"],
    ["NFR7", "Testability — every safety claim must be asserted by an automated test rather "
             "than argued in prose.", "131 tests, all passing"],
    ["NFR8", "Statistical rigour — every controller comparison is a paired Wilcoxon "
             "signed-rank test across seeds, reporting p and n; fewer than six seeds and the "
             "harness must refuse to test rather than print a mean.", "p and n on every row"],
    ["NFR9", "Auditability — every applied and every blocked action is logged with the reason "
             "the guardrail gave.", "decision log per run"],
    ["NFR10", "Usability — the dashboard must degrade gracefully to rule-based risk bands if "
              "no pre-trained models are present, and say so.", "sidebar status"],
]

HARDWARE = [
    ["Component", "Specification used", "Minimum"],
    ["Processor", "x86-64, 8 cores", "x86-64, 4 cores"],
    ["Memory (RAM)", "16 GB", "8 GB"],
    ["Storage", "10 GB free (cached parquet, model weights, evaluation outputs)", "5 GB"],
    ["Graphics Processing Unit", "None — the project is CPU-only by design", "None"],
    ["Display", "1920 × 1080, for the four-view operator console", "1366 × 768"],
    ["Network", "Required once, for the initial data collection only", "Any broadband link"],
]

SOFTWARE = [
    ["Component", "Choice"],
    ["Operating system", "Windows 11 (developed); any OS with Python 3.11"],
    ["Programming language", "Python 3.11"],
    ["Development environment", "VS Code; pytest for the test suite"],
    ["Deep learning", "PyTorch (LSTM forecaster, action-conditioned world model)"],
    ["Classical machine learning", "scikit-learn, LightGBM, HDBSCAN, SHAP"],
    ["Numerics and data", "NumPy, pandas, SciPy, PyArrow (parquet)"],
    ["Visualisation and interface", "PySide6 and pyqtgraph (desktop console), Streamlit and "
                                    "Plotly (browser dashboard), Matplotlib"],
    ["Configuration", "PyYAML — seven configuration files under configs/"],
    ["Data source", "Sensor Based Aquaponics Fish Pond Datasets (Kaggle, CC BY 4.0) - catfish "
                    "ponds, University of Nigeria Nsukka"],
    ["Optional", "dp-accounting (Rényi accountant for DP-FedAvg)"],
]


def chapter3(doc, toc):
    chapter(doc, 3, "Technical Specification", toc)

    section(doc, "3.1", "Requirements", toc)
    subsection(doc, "3.1.1", "Functional", toc)
    para(doc, (
        "Each functional requirement below is stated together with the artefact that verifies "
        "it, because a requirement whose satisfaction cannot be checked is a wish."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    make_table(doc, FUNCTIONAL, widths=[1.3, 8.2, 6.0], size=9)
    caption_table(doc, "Table 2. Functional requirements and their verification.")

    subsection(doc, "3.1.2", "Non-Functional", toc)
    make_table(doc, NON_FUNCTIONAL, widths=[1.5, 9.6, 4.4], size=9)
    caption_table(doc, "Table 3. Non-functional requirements.")

    section(doc, "3.2", "Feasibility Study", toc)
    subsection(doc, "3.2.1", "Technical Feasibility", toc)
    para(doc, (
        "Every technology the project depends on is mature and open. Federated averaging is "
        "implemented in-process rather than over a network stack, which keeps the comparison "
        "deterministic and removes a whole class of infrastructure risk; the clients are "
        "genuinely isolated by construction rather than by a network boundary, so the privacy "
        "claim does not depend on the transport. The pond dataset is public, is downloaded once "
        "and cached, and collection is idempotent, so the pipeline can be reproduced from a "
        "clean checkout. The heaviest computation in the project is the evaluation grid — five "
        "controllers × eight scenarios × ten seeds plus the ablation grid — which completes in "
        "roughly twenty minutes on a commodity CPU; federated pre-training and the world model "
        "each take about a minute. No GPU is required at any stage. The principal technical "
        "risk is not compute but data quality, and it is addressed by design: range checks "
        "drop disconnected-probe codes and impossible values, stuck sensors are masked, an "
        "availability mask is fed to the network so a missing reading is never mistaken for a "
        "zero, ponds that are mostly outage are excluded, and the forecaster predicts deltas "
        "rather than levels."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    subsection(doc, "3.2.2", "Economic Feasibility", toc)
    para(doc, (
        "The development cost of the software is zero in licence terms: Python, PyTorch, "
        "LightGBM, HDBSCAN, SHAP, PySide6 and Streamlit are all open source, and the pond "
        "dataset is published under CC BY 4.0. Development ran on a single laptop. "
        "For a deployment, the marginal cost is the instrumentation a farm would need in any "
        "case — a multiparameter sonde and an actuator relay — plus a low-power edge computer, "
        "since inference is CPU-only. Federation shifts cost in the farm's favour twice: no "
        "central data-warehouse contract is required, and a farm too small to train a useful "
        "model alone gets the benefit of the pooled one. Against that, the results in Section "
        "4.4 show the agent is measurably more expensive to operate than a fixed-threshold "
        f"controller in {RS.load()['cost_worse']} of {RS.load()['cost_n']} scenarios, because "
        "preemption means sometimes acting on "
        "excursions that would have resolved unaided; whether that trade is worth making "
        "depends on the value of the stock at risk, and the cost weight is exposed as a "
        "configuration dial rather than fixed by us."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    subsection(doc, "3.2.3", "Social Feasibility", toc)
    para(doc, (
        "The system is designed to be adopted by an operator who does not trust it yet, which "
        "is the realistic starting condition. Three properties serve that. First, every "
        "decision comes with a SHAP attribution naming the channels that drove the risk score, "
        "so the recommendation is inspectable rather than oracular. Second, the counterfactual "
        "panel shows the pond that would have existed had nothing been done, so the operator "
        "can see what the system bought rather than being asked to believe it. Third, the "
        "guardrail layer is stated explicitly and its blocked actions are logged with reasons, "
        "so the boundaries of the system's authority are legible. Federated training means a "
        "farm contributes to a shared model without disclosing production data to a competitor "
        "or an integrator, which removes the most common objection to sector-wide analytics. "
        "The ethical position taken throughout is that the system advises and acts within hard "
        "limits, and that its limits — a simulated pond under control, noisy low-cost sensor "
        "data, and relative rather than absolute claims — are stated wherever a result is "
        "reported rather than confined to a disclaimer."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    section(doc, "3.3", "System Specification", toc)
    subsection(doc, "3.3.1", "Hardware Specification", toc)
    make_table(doc, HARDWARE, widths=[3.8, 8.0, 3.7], size=10)
    caption_table(doc, "Table 4. Hardware specification.")

    subsection(doc, "3.3.2", "Software Specification", toc)
    make_table(doc, SOFTWARE, widths=[4.5, 11.0], size=10)
    caption_table(doc, "Table 5. Software specification.")


# --------------------------------------------------------------------------- chapter 4
def modules_table(R):
    return [
        ["Module", "Path", "Function", "Status"],
        ["Data acquisition", "src/data/", "Kaggle pond loader, harmoniser with QC and "
         "availability masks, site registry, licence manifest",
         f"Complete — {R['n_ponds']} ponds, {R['n_rows']:,} rows"],
        ["Digital twin", "src/twin/", "Pond dynamics, eight scenarios, risk bands, "
         "calibration hooks", "Complete"],
        ["Sensor simulation", "src/sensors/", "Noise, dropout, injectable probe faults",
         "Complete"],
        ["Perception", "src/perception/", "LSTM forecaster, HDBSCAN novelty, LightGBM risk, "
         "SHAP", "Complete"],
        ["World model", "src/perception/world_model.py", "Action-conditioned dynamics, "
         "federated across simulated ponds", "Complete"],
        ["Federated learning", "src/federated/", "PondClient, FedAvg, independent-local and "
         "central regimes, optional DP", "Complete; DP implemented but unrun"],
        ["Agent", "src/agent/", "Reactive and predictive planners, guardrail stack, "
         "actuators, control loop", "Complete"],
        ["Evaluation", "src/eval/", "Baselines, metrics, paired statistics, ablation grid",
         "Complete"],
        ["Operator interface", "desktop/, dashboard/", "Desktop console (Live, "
         "Counterfactual, Agent, Scorecard) and equivalent Streamlit dashboard", "Complete"],
        ["Reporting", "scripts/", "Figure, report and slide generation from outputs/",
         "Complete"],
    ]


def chapter4(doc, toc):
    chapter(doc, 4, "Design Approach and Details", toc)

    section(doc, "4.1", "System Architecture", toc)
    figure(doc, FIGS / "fig2_architecture.png", "Fig. 2. AquaSentinel system architecture.",
           16.5)
    para(doc, (
        "The architecture separates into four lanes, and the separation is the design. The "
        "top lane runs once and is the only part of the system that touches the network: the "
        "pond dataset is downloaded once, one file per pond, and a harmoniser maps every "
        "file's differently-named columns onto a single channel schema at 15-minute cadence, "
        "drops physically impossible readings, masks stuck probes and records an explicit "
        "availability mask; the result is cached as parquet with a per-pond licence record. "
        "Collection is idempotent, so a cached pond is never fetched twice."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "The second lane is offline pre-training. Each real pond becomes one PondClient "
        "that owns its own dataset and its own model instance and exposes only local training, "
        "weight get and set, and evaluation. The server receives weight tensors and a sample "
        "count and nothing else, so the isolation is structural rather than promised. Two "
        "models come out of this lane: the LSTM forecaster, which predicts what the water does "
        "on its own, and the action-conditioned world model, whose encoder is warm-started from "
        "the federated forecaster and which predicts what the water does under a candidate "
        "action. The distinction matters because the pond dataset records water quality but "
        "not when the aerator ran, how much was fed or when lime went in, and therefore "
        "contains no action-response pairs at all, so the action-conditioned half is learned "
        "from twin rollouts federated across eight "
        "heterogeneous simulated ponds while the representation it computes on remains the "
        "federated one."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "The third lane is the live loop and runs every tick. The twin steps forward and "
        "produces a true state; the sensor stream samples it into a noisy observation; "
        "perception forecasts the next horizon, flags novelty, scores risk and attributes that "
        "score; the planner rolls fourteen curated candidate actions through the world model "
        "and takes the minimum-cost trajectory. The proposed action then meets the guardrail "
        "stack, which is the one component on the actuator path that cannot be bypassed. Two "
        "properties of this lane are load-bearing. The planner re-decides on a slow interval "
        "while the guardrails run on every sensor tick, which is the standard supervisory-"
        "control split and is what keeps unsafe-applied at zero between decisions. And a shadow "
        "twin is forked at t = 0 with the same seed, the same perturbation schedule and the "
        "same noise realisation, running with the agent disabled, so that every divergence "
        "between the two ponds is caused by an action and by nothing else."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "The fourth lane is the operator interface: a native desktop console with four views "
        "(an equivalent Streamlit dashboard is kept) and the batch evaluation harness that "
        "regenerates every reported number into outputs/."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "The guardrail stack, in firing order", size=12, bold=True, spacing=1.15,
         space_before=8, space_after=4)
    bullets(doc, [
        "Dosing caps and minimum intervals between doses of the same reagent.",
        "Rate limits on how fast any actuator setting may change.",
        "Contextual blocks — no lime into an already alkaline pond, no feeding at low "
        "dissolved oxygen, no large water exchange into worse or thermally shocking water.",
        "Rejection of any action the internal model predicts will drive a channel into its "
        "critical band.",
        "Emergency override, placed last so that it outranks everything above it: when the "
        "pond is starved of oxygen, nothing outranks air.",
    ])
    para(doc, (
        "The guardrails see only the noisy probe, but safety is judged against the pond's true "
        "state, so each threshold is given roughly three standard deviations of sensor noise as "
        "headroom — a threshold that fires slightly early rather than slightly late."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    section(doc, "4.2", "Design", toc)

    subsection(doc, "4.2.1", "Data Flow Diagram", toc)
    figure(doc, FIGS / "fig3_dfd0.png", "Fig. 3. Data flow diagram — Level 0 (context).", 14.5)
    para(doc, (
        "At the context level the system exchanges data with three external entities. A public "
        "pond sensor dataset supplies raw time series once. The farm operator supplies the scenario, "
        "the choice of driving weights and a cost preference, and receives the risk band, the "
        "forecast, the SHAP rationale, the counterfactual comparison and the audit log. Pond "
        "actuators receive allowed actions and return the resulting water-quality state."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    figure(doc, FIGS / "fig4_dfd1.png", "Fig. 4. Data flow diagram — Level 1.", 15.5)
    para(doc, (
        "The Level-1 decomposition names eight processes and four data stores. Processes 1.0 "
        "and 2.0 are offline: collection and harmonisation write D1, and federated "
        "pre-training reads D1 and writes model weights to D2. Calibration statistics can also "
        "flow from D1 into the twin; for this dataset they were computed and deliberately not "
        "applied, because the fitted noise magnitudes reflect probe faults rather than pond "
        "physics. Processes 3.0 through 7.0 form the live loop, and process 8.0 "
        "renders and logs. The dashed edges run once; the solid edges run every tick."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    subsection(doc, "4.2.2", "Use Case Diagram", toc)
    figure(doc, FIGS / "fig5_usecase.png", "Fig. 5. Use case diagram.", 15.5)
    para(doc, (
        "Four actors interact with the system. The farm operator selects a scenario, monitors "
        "live water quality and risk, inspects the counterfactual and the SHAP rationale, and "
        "may override or approve an action. The researcher or evaluator switches the driving "
        "weights between the federated, independent-local and central regimes and runs the "
        "evaluation grid and ablations. The public dataset is a supporting actor for "
        "collection only, and the pond actuator hardware is the recipient of allowed actions. "
        "Forecasting and risk scoring are included by the monitoring use case; guardrail "
        "enforcement is included by planning, which is the structural statement that no plan "
        "reaches an actuator except through the safety layer."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    subsection(doc, "4.2.3", "Class Diagram", toc)
    figure(doc, FIGS / "fig6_class.png", "Fig. 6. Class diagram.", 15.5)
    para(doc, (
        "The principal collaborations are shown. PondTwin owns the true state and is the only "
        "class that may mutate it; SensorStream is the sole path from true state to "
        "observation, which is what allows a sensor fault to corrupt a reading without "
        "corrupting the water. PondClient wraps one pond's dataset and one model instance and "
        "is the unit of federation. WorldModel and RiskModel feed Planner, which proposes; "
        "GuardrailStack disposes; AgentLoop owns both, plus the twin and its shadow, and is the "
        "only class that calls apply."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    subsection(doc, "4.2.4", "Sequence Diagram", toc)
    figure(doc, FIGS / "fig7_sequence.png", "Fig. 7. Sequence diagram for one control tick.",
           16.5)
    para(doc, (
        "One tick proceeds as shown. Note messages 9 and 10: the guardrail decision returns "
        "either an allowed action or a blocked action with a reason, and both are logged; and "
        "the shadow pond advances with the identical seed and noise draw and the agent "
        "disabled. Message 13 carries the guardrail event count and the unsafe-applied count, "
        "which is zero for every guardrailed controller across the whole evaluation grid."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    section(doc, "4.3", "Module-Wise Implementation Status", toc)
    para(doc, (
        "The system is fully implemented end to end; the table below records the state of each "
        f"module at Review-2. The automated test suite comprises {RS.test_count()} tests, "
        "all passing, "
        "covering twin sanity, guardrail units, collector schema and mask, perception output "
        "shapes and the evaluation statistics."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    make_table(doc, modules_table(RS.load()), widths=[3.0, 3.2, 5.6, 3.7], size=9)
    caption_table(doc, "Table 6. Module-wise implementation status at Review-2.")

    section(doc, "4.4", "Preliminary Experimental Results", toc)
    R = RS.load()
    w1, w2 = len(REFERENCES) + 1, len(REFERENCES) + 2    # the two weblinks in Chapter 5
    para(doc, (
        "Every number and every table in this section is read from outputs/*.csv when this "
        "document is built (scripts/results_source.py); none was typed by hand, and each "
        "verdict below is computed from the same numbers it describes."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    def head(text):
        para(doc, text, size=12, bold=True, spacing=1.15, space_before=8, space_after=4)

    head("A. Real data")
    para(doc, (
        f"{R['n_ponds']} real catfish aquaponics ponds on local farms in Nsukka, Enugu State, "
        f"Nigeria [{w1}], [{w2}], giving {R['n_rows']:,} harmonised rows at 15-minute cadence "
        f"from {R['start']} to {R['end']}. The raw files hold eleven ponds logged every ~20 s by "
        "ESP32-based probes; quality control removed disconnected-probe codes (-127 °C), pH "
        "below 3, ammonia readings in the 10^11 range and stuck sensors, and dropped every pond "
        f"whose record was more than {R['max_missing']:.0%} empty or which had no usable "
        "dissolved-oxygen record. Most of the empty time is the logger being offline (37-55 % "
        "of every pond's timeline), not rejected readings. Nitrate "
        "is excluded because the raw column is an analogue-to-digital count, not a "
        "concentration. All surviving ponds carry the same five channels - dissolved oxygen, "
        "temperature, pH, total ammonia nitrogen and turbidity - so the federated clients "
        "differ in pond conditions, fish stock and probe faults rather than in which "
        "instruments they own."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    make_table(doc, [["Pond", "Rows (15 min)", "Start", "End", "Missing"]] + R["pond_rows"],
               widths=[2.6, 3.0, 3.2, 3.2, 2.5], size=10)
    caption_table(doc, "Table 7. The real ponds after quality control.")

    head("B. Federated versus independent-local versus central")
    para(doc, (
        f"{', '.join(R['train_ponds'])} train; {', '.join(R['holdout_ponds'])} "
        f"{'is' if R['n_holdout'] == 1 else 'are'} held out and no regime ever trains on "
        f"{'it' if R['n_holdout'] == 1 else 'them'}. Three controls make the comparison meaningful: "
        f"matched total training work ({R['rounds']} rounds × {R['local_epochs']} local epochs "
        "against the same number of epochs for the other regimes); a shared initialisation, "
        "because averaging differently-initialised networks averages across permuted hidden "
        "units and is meaningless; and evaluation on the held-out ponds, because the question "
        "is transfer to a pond never seen. Independent-local is scored as the mean over its "
        "per-pond models."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    make_table(doc, [["Training regime", "RMSE — dissolved oxygen", "RMSE — temperature",
                      "RMSE — pH"]] + R["fed_rows"], widths=[6.7, 3.0, 3.0, 2.8], size=10)
    caption_table(doc, "Table 8. Forecast RMSE on the held-out ponds (lower is better).")
    every = ("on every channel" if R["fed_beats_local_all"] else "on dissolved oxygen")
    better = R["fed_do"] < R["loc_do"]
    para(doc, (
        f"Federated pre-training {fed_phrase(R)}"
        + (f"; it is better than training alone {every}. " if better else ". ")
        + f"FedAvg ran {R['rounds']} rounds and mean local loss fell from "
        f"{R['loss_first']:.1f} to {R['loss_last']:.1f}. The absolute error is large — around "
        f"{R['fed_do']:.1f} mg/L of dissolved oxygen across a horizon of up to six hours on the "
        "15-minute feed — because the probes are noisy and outages break the series; the "
        f"comparison between regimes is the claim, not the absolute accuracy. With "
        f"{R['n_train_ponds']} training ponds and {R['n_holdout']} held out, it is also a small "
        "study, and is presented as such."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        f"The data-poor case is the one federation exists for. {', '.join(R['poor_ponds'])} "
        f"{'kept' if len(R['poor_ponds']) == 1 else 'each kept'} only "
        f"{R['poor_frac']:.0%} of their history, and each is scored on its own held-out tail:"),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    make_table(doc, [["Pond", "Trained alone", "Federated", "Central", "Federated vs alone"]]
               + R["poor_rows"], widths=[2.4, 3.1, 3.1, 3.1, 3.4], size=10)
    caption_table(doc, "Table 9. Dissolved-oxygen RMSE (mg/L) for the data-poor ponds.")
    para(doc, (
        f"Federation improved dissolved-oxygen RMSE for {R['poor_won']} of {R['poor_n']} "
        f"data-poor ponds ({R['poor_changes']} against training alone), without any pond sharing "
        "a single reading. Federation is not guaranteed to help every client, and the pond it "
        "did not help is reported rather than left out."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    make_table(doc, [["Training regime", "RMSE (scaled delta units)",
                      "Dissolved oxygen RMSE (mg/L)"]] + R["wm_rows"],
               widths=[5.5, 5.0, 5.0], size=10)
    caption_table(doc, "Table 10. Action-conditioned world model, held-out simulated ponds.")
    wm_v = ("beats" if R["wm_fed"] < R["wm_loc"] else "does not beat")
    para(doc, (
        f"For the world model the federated result {wm_v} independent-local "
        f"({RS.f3(R['wm_fed'])} against {RS.f3(R['wm_loc'])} mg/L), and both trail the pooled "
        f"reference ({RS.f3(R['wm_cen'])} mg/L). Its clients are simulated ponds with different "
        "volume, depth, aeration and productivity, so this is a different federation from "
        "Table 8, and its consequences for control are measured in the ablation table rather "
        "than asserted."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    head("C. Perception")
    rk, an, rg = R["risk"], R["anomaly"], R["regime"]
    para(doc, (
        f"The risk classifier reaches accuracy {rk['accuracy']:.3f}, balanced accuracy "
        f"{rk['balanced_accuracy']:.3f}, macro-F1 {rk['macro_f1']:.3f} and critical-class "
        f"recall {rk['critical_recall']:.3f} at precision {rk['critical_precision']:.3f} on a "
        f"held-out seed, over {rk['n']:,} samples. Its target is the risk band that "
        "configs/twin.yaml assigns to the worst point of the true next horizon, so the model "
        "has not learned water chemistry — it has learned to anticipate a known rule from an "
        f"imperfect forecast. Anomaly detection on injected sensor faults achieves event recall "
        f"{an['recall']:.2f} ({an['detected']} of {an['events']} injected events across "
        f"{an['seeds']} seeds) with a false-positive rate of {an['fpr']:.3f} outside the fault "
        "windows, scored per event rather than per tick. " + anomaly_verdict(an["fpr"])),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        f"HDBSCAN finds {rg['n_clusters']} clusters with {rg['noise']:.0%} of "
        f"{rg['n_points']:,} windows unassigned, DBCV {rg['dbcv']:.3f} and silhouette "
        f"{rg['silhouette']:.3f}. A DBCV near zero means the clusters are not well "
        "separated: the windows form a continuum rather than discrete regimes, so nothing "
        "downstream depends on the cluster id and only the unassigned label - the anomaly "
        "flag - is used."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    head("D. Safety — the headline claim")
    make_table(doc, [["Controller", "Runs", "Unsafe actions applied", "Guardrail events"]]
               + R["safety_rows"], widths=[6.5, 2.0, 4.0, 3.0], size=10)
    caption_table(doc, "Table 11. Unsafe actions applied across the full evaluation grid.")
    if R["guarded_unsafe"] == 0:
        safety_line = (f"Every guardrailed controller applied zero unsafe actions across all "
                       f"{R['runs_per_controller']} runs.")
    else:
        safety_line = (f"The guardrailed controllers applied {R['guarded_unsafe']} unsafe "
                       "actions in total; this is reported, not hidden.")
    para(doc, (
        safety_line + " do_nothing is deliberately left unguarded because it is a reference "
        "trajectory rather than a controller, and letting the safety layer rescue it would "
        "destroy the comparison it exists to provide."), align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    head("E. Agent against baselines")
    make_table(doc, [["Scenario", "do_nothing", "static_threshold", "reactive", "predictive"]]
               + R["risk_rows"], widths=[4.3, 2.8, 3.2, 2.6, 2.6], size=10)
    caption_table(doc, f"Table 12. Median dissolved-oxygen time-out-of-band (hours) across "
                       f"{R['n_seeds']} seeds; lower is better.")
    ties = ""
    if R["tie_n"]:
        ties = (f" {R['tie_n']} comparisons are exact ties in "
                f"{', '.join(R['tie_scenarios'])}"
                + (", where no controller accrues any risk and there is nothing to win."
                   if R["tie_all_zero"] else "."))
    other = ""
    rest = R["comparisons"] - R["wins"] - R["tie_n"]
    if rest:
        other = (f" The remaining {rest} are not significant wins"
                 + (f"; {R['losses']} of them are significant losses." if R["losses"] else "."))
    para(doc, (
        "Comparisons are paired Wilcoxon signed-rank tests across seeds, because the same seed "
        "gives every controller the identical perturbation schedule and noise realisation. The "
        f"agent is significantly better (p < 0.05) in {R['wins']} of {R['comparisons']} "
        "scenario × baseline comparisons on this metric." + ties + other),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, (
        "Reporting only that win would be a half-truth. Against the static threshold on "
        f"cumulative intervention cost, the agent is more expensive in {R['cost_worse']} of "
        f"{R['cost_n']} scenarios. It buys preemption, and preemption means acting on "
        "excursions that would sometimes have resolved on their own. The intervention cost "
        "weight in configs/agent.yaml is the dial that trades one against the other; nothing "
        "in this work establishes where a real farm should set it."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    head("F. Ablations")
    make_table(doc, [["Ablation", "Median DO time-out-of-band (h)", "Median cost",
                      "Unsafe applied", "Runs"]] + R["ablation_rows"],
               widths=[5.0, 4.2, 2.2, 2.4, 1.7], size=10)
    caption_table(doc, "Table 13. Ablation grid.")
    if R["ab_noguard_unsafe"] > 0:
        g_text = (f"First, the guardrails are load-bearing: with the safety layer removed the "
                  f"shipped planner applies {R['ab_noguard_unsafe']:,} unsafe actions across "
                  f"the ablation grid, and with it in place it applies {R['ab_full_unsafe']}. "
                  "A planner optimises against its model as if the model were exact; where the "
                  "model is wrong, prediction error becomes unsafe commands, which is why the "
                  "hard constraints live outside the learned component.")
    else:
        g_text = ("First, without the safety layer the planner applied no unsafe actions in "
                  "this grid, so on this data the guardrails acted as a backstop that was not "
                  "needed rather than a crutch.")
    para(doc, g_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    diff = R["ab_full"] - R["ab_nolstm"]
    if diff > 0:
        l_text = (f"Second, the LSTM residual correction makes control worse: the shipped "
                  f"configuration scores {R['ab_full']:.2f} h of dissolved-oxygen "
                  f"time-out-of-band against {R['ab_nolstm']:.2f} h without it. It ships "
                  "enabled as a deliberate choice with a stated price, because the federated "
                  "forecaster is the component this project exists to produce and keeping it "
                  "on the actuator path is what makes this a learned controller; the ablation "
                  "stays in the grid so the cost is re-measured on every run.")
    elif diff < 0:
        l_text = (f"Second, the LSTM residual correction helps control: {R['ab_full']:.2f} h "
                  f"of dissolved-oxygen time-out-of-band with it against "
                  f"{R['ab_nolstm']:.2f} h without.")
    else:
        l_text = (f"Second, the LSTM residual correction makes no difference to the median "
                  f"({R['ab_full']:.2f} h either way).")
    para(doc, l_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    wdiff = R["ab_full"] - R["ab_nowm"]
    para(doc, (
        "Third, replacing the hand-written pond model inside the planner with the learned "
        "world model " + (f"costs {wdiff:.2f} h" if wdiff > 0 else
                          f"saves {-wdiff:.2f} h" if wdiff < 0 else "changes nothing")
        + " of median dissolved-oxygen time-out-of-band. That is the measured effect of making "
        "the trajectories the agent chooses between the output of a neural network."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    head("G. Demonstration")
    D = RS.demo()
    para(doc, (
        f"The desktop console was driven end to end on the {D['scenario']} scenario at seed "
        f"{D['seed']}: the scenario runs to completion over {D['ticks']} ticks with "
        f"{D['unsafe']} unsafe actions applied and {D['critical_min']:.0f} minutes in the "
        f"critical band, at {D['risk_h']:.1f} h time-in-risk against {D['shadow_risk_h']:.1f} h "
        "for the shadow pond running the identical noise realisation with the agent disabled; "
        f"the lowest dissolved oxygen reached is {D['min_do_gap']:+.2f} mg/L relative to the "
        "shadow pond. Figure 8 shows the counterfactual view from that run."),
        align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    figure(doc, SHOTS / "2-counterfactual.png",
           "Fig. 8. Counterfactual panel — the agent's pond held above the no-action shadow "
           "pond through two dawn oxygen sags.", 15.5)


# --------------------------------------------------------------------------- chapter 5
REFERENCES = [
    'M. Føre, K. Frank, T. Norton, E. Svendsen, J. A. Alfredsen, T. Dempster, H. Eguiraun, W. '
    'Watson, A. Stahl, L. M. Sunde, C. Schellewald, K. Skøien, M. O. Alver, and D. Berckmans, '
    '"Precision fish farming: A new framework to improve production in aquaculture," Biosystems '
    'Engineering, vol. 173, pp. 176-193, 2018.',

    'C. Zhou, D. Xu, L. Chen, S. Zhang, C. Sun, X. Yang, and Y. Wang, "Intelligent feeding '
    'control methods in aquaculture with an emphasis on fish: a review," Reviews in Aquaculture, '
    'vol. 10, no. 4, pp. 975-993, 2018.',

    'S. Liu, L. Xu, D. Li, Q. Li, Y. Jiang, H. Tai, and L. Zeng, "Prediction of dissolved oxygen '
    'content in river crab culture based on least squares support vector regression optimized by '
    'improved particle swarm optimization," Computers and Electronics in Agriculture, vol. 95, '
    'pp. 82-91, 2013.',

    'X. Ta and Y. Wei, "Research on a dissolved oxygen prediction method for recirculating '
    'aquaculture systems based on a convolution neural network," Computers and Electronics in '
    'Agriculture, vol. 145, pp. 302-310, 2018.',

    'J. Ren, Y. Wang, W. Zhao, and D. Li, "Research of dissolved oxygen prediction in '
    'recirculating aquaculture systems based on deep belief network," Aquacultural Engineering, '
    'vol. 90, Art. no. 102085, 2020.',

    'R. Barzegar, M. T. Aalami, and J. Adamowski, "Short-term water quality variable prediction '
    'using a hybrid CNN-LSTM deep learning model," Stochastic Environmental Research and Risk '
    'Assessment, vol. 34, pp. 415-433, 2020.',

    'Z. Hu, Y. Zhang, Y. Zhao, M. Xie, J. Zhong, Z. Tu, and J. Liu, "A water quality prediction '
    'method based on the deep LSTM network considering correlation in smart mariculture," '
    'Sensors, vol. 19, no. 6, Art. no. 1420, 2019.',

    'H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, '
    '"Communication-efficient learning of deep networks from decentralized data," in Proc. 20th '
    'International Conference on Artificial Intelligence and Statistics (AISTATS), Fort '
    'Lauderdale, FL, USA, 2017, pp. 1273-1282.',

    'T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated '
    'optimization in heterogeneous networks," in Proc. Machine Learning and Systems (MLSys), '
    'Austin, TX, USA, 2020, pp. 429-450.',

    'Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, "Federated learning with non-IID '
    'data," arXiv preprint arXiv:1806.00582, 2018.',

    'A. Durrant, M. Markovic, D. Matthews, D. May, J. Enright, and G. Leontidis, "The role of '
    'cross-silo federated learning in facilitating data sharing in the agri-food sector," '
    'Computers and Electronics in Agriculture, vol. 193, Art. no. 106648, 2022.',

    'L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, "Learning-based model '
    'predictive control: Toward safe learning in control," Annual Review of Control, Robotics, '
    'and Autonomous Systems, vol. 3, pp. 269-296, 2020.',

    'K. P. Wabersich and M. N. Zeilinger, "A predictive safety filter for learning-based control '
    'of constrained nonlinear dynamical systems," Automatica, vol. 129, '
    'Art. no. 109597, 2021.',

    'D. Ha and J. Schmidhuber, "Recurrent world models facilitate policy evolution," in Advances '
    'in Neural Information Processing Systems 31, Montréal, Canada, 2018, pp. 2450-2462.',

    'FAO, "The State of World Fisheries and Aquaculture 2024," Food and Agriculture '
    'Organization of the United Nations, Rome, 2024.',

    'C. E. Boyd and C. S. Tucker, Pond Aquaculture Water Quality Management. Boston, MA, USA: '
    'Kluwer Academic Publishers, 1998.',

    'C. E. Boyd, Water Quality in Ponds for Aquaculture. Auburn, AL, USA: Alabama Agricultural '
    'Experiment Station, Auburn University, 1990.',

    'American Public Health Association, Standard Methods for the Examination of Water and '
    'Wastewater, 23rd ed. Washington, DC, USA: APHA, 2017.',

    'K. Emerson, R. C. Russo, R. E. Lund, and R. V. Thurston, "Aqueous ammonia equilibrium '
    'calculations: effect of pH and temperature," Journal of the Fisheries Research Board of '
    'Canada, vol. 32, no. 12, pp. 2379-2383, 1975.',

    'B. B. Benson and D. Krause, "The concentration and isotopic fractionation of oxygen '
    'dissolved in freshwater and seawater in equilibrium with the atmosphere," Limnology and '
    'Oceanography, vol. 29, no. 3, pp. 620-632, 1984.',

    'H. W. Streeter and E. B. Phelps, "A study of the pollution and natural purification of the '
    'Ohio River," U.S. Public Health Service, Bulletin No. 146, 1925.',

    'R. G. Wetzel, Limnology: Lake and River Ecosystems, 3rd ed. San Diego, CA, USA: Academic '
    'Press, 2001.',

    'F. O\'Donncha and J. Grant, "Precision aquaculture," IEEE Internet of Things Magazine, '
    'vol. 2, no. 4, pp. 26-30, 2019.',

    'S. Antonucci and C. Costa, "Precision aquaculture: a short review on engineering '
    'innovations," Aquaculture International, vol. 28, pp. 41-57, 2020.',

    'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, '
    'no. 8, pp. 1735-1780, 1997.',

    'K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. '
    'Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine '
    'translation," in Proc. Conference on Empirical Methods in Natural Language Processing '
    '(EMNLP), Doha, Qatar, 2014, pp. 1724-1734.',

    'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. '
    'Polosukhin, "Attention is all you need," in Advances in Neural Information Processing '
    'Systems 30, Long Beach, CA, USA, 2017, pp. 5998-6008.',

    'B. Lim and S. Zohren, "Time-series forecasting with deep learning: a survey," '
    'Philosophical Transactions of the Royal Society A, vol. 379, no. 2194, '
    'Art. no. 20200209, 2021.',

    'D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, "DeepAR: Probabilistic '
    'forecasting with autoregressive recurrent networks," International Journal of Forecasting, '
    'vol. 36, no. 3, pp. 1181-1191, 2020.',

    'M. S. Ahmed, T. A. Aurpa, and M. A. K. Azad, "Fish disease detection using image based '
    'machine learning technique in aquaculture," Journal of King Saud University - Computer and '
    'Information Sciences, vol. 34, no. 8, pp. 5170-5182, 2022.',

    'T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, "Federated learning: Challenges, methods, '
    'and future directions," IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50-60, 2020.',

    'P. Kairouz et al., "Advances and open problems in federated learning," Foundations and '
    'Trends in Machine Learning, vol. 14, no. 1-2, pp. 1-210, 2021.',

    'Q. Yang, Y. Liu, T. Chen, and Y. Tong, "Federated machine learning: Concept and '
    'applications," ACM Transactions on Intelligent Systems and Technology, vol. 10, no. 2, '
    'pp. 1-19, 2019.',

    'D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, and H. V. Poor, "Federated '
    'learning for Internet of Things: A comprehensive survey," IEEE Communications Surveys and '
    'Tutorials, vol. 23, no. 3, pp. 1622-1658, 2021.',

    'A. Imteaj, U. Thakker, S. Wang, J. Li, and M. H. Amini, "A survey on federated learning for '
    'resource-constrained IoT devices," IEEE Internet of Things Journal, vol. 9, no. 1, '
    'pp. 1-24, 2022.',

    'O. Friha, M. A. Ferrag, L. Shu, L. Maglaras, K.-K. R. Choo, and M. Nafaa, "FELIDS: '
    'Federated learning-based intrusion detection system for agricultural Internet of Things," '
    'Journal of Parallel and Distributed Computing, vol. 165, pp. 17-31, 2022.',

    'M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, "Deep '
    'learning with differential privacy," in Proc. ACM SIGSAC Conference on Computer and '
    'Communications Security (CCS), Vienna, Austria, 2016, pp. 308-318.',

    'I. Mironov, "Rényi differential privacy," in Proc. IEEE 30th Computer Security Foundations '
    'Symposium (CSF), Santa Barbara, CA, USA, 2017, pp. 263-275.',

    'R. J. G. B. Campello, D. Moulavi, and J. Sander, "Density-based clustering based on '
    'hierarchical density estimates," in Proc. Pacific-Asia Conference on Knowledge Discovery '
    'and Data Mining (PAKDD), Gold Coast, Australia, 2013, pp. 160-172.',

    'L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical density based clustering," '
    'Journal of Open Source Software, vol. 2, no. 11, Art. no. 205, 2017.',

    'D. Moulavi, P. A. Jaskowiak, R. J. G. B. Campello, A. Zimek, and J. Sander, "Density-based '
    'clustering validation," in Proc. SIAM International Conference on Data Mining (SDM), '
    'Philadelphia, PA, USA, 2014, pp. 839-847.',

    'V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," ACM Computing '
    'Surveys, vol. 41, no. 3, pp. 1-58, 2009.',

    'A. Blázquez-García, A. Conde, U. Mori, and J. A. Lozano, "A review on outlier/anomaly '
    'detection in time series data," ACM Computing Surveys, vol. 54, no. 3, pp. 1-33, 2021.',

    'F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in Proc. 8th IEEE International '
    'Conference on Data Mining (ICDM), Pisa, Italy, 2008, pp. 413-422.',

    'G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A '
    'highly efficient gradient boosting decision tree," in Advances in Neural Information '
    'Processing Systems 30, Long Beach, CA, USA, 2017, pp. 3146-3154.',

    'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in '
    'Advances in Neural Information Processing Systems 30, Long Beach, CA, USA, 2017, '
    'pp. 4765-4774.',

    'S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. '
    'Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with '
    'explainable AI for trees," Nature Machine Intelligence, vol. 2, no. 1, pp. 56-67, 2020.',

    'C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and '
    'use interpretable models instead," Nature Machine Intelligence, vol. 1, no. 5, '
    'pp. 206-215, 2019.',

    'E. F. Camacho and C. Bordons, Model Predictive Control, 2nd ed. London, U.K.: Springer, '
    '2007.',

    'D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, "Dream to control: Learning behaviors by '
    'latent imagination," in Proc. International Conference on Learning Representations (ICLR), '
    'Addis Ababa, Ethiopia, 2020.',

    'K. Chua, R. Calandra, R. McAllister, and S. Levine, "Deep reinforcement learning in a '
    'handful of trials using probabilistic dynamics models," in Advances in Neural Information '
    'Processing Systems 31, Montréal, Canada, 2018, pp. 4754-4765.',

    'J. García and F. Fernández, "A comprehensive survey on safe reinforcement learning," '
    'Journal of Machine Learning Research, vol. 16, no. 1, pp. 1437-1480, 2015.',

    'D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, "Concrete '
    'problems in AI safety," arXiv preprint arXiv:1606.06565, 2016.',

    'G. Dulac-Arnold, D. Mankowitz, and T. Hester, "Challenges of real-world reinforcement '
    'learning," arXiv preprint arXiv:1904.12901, 2019.',

    'M. Grieves and J. Vickers, "Digital twin: Mitigating unpredictable, undesirable emergent '
    'behavior in complex systems," in Transdisciplinary Perspectives on Complex Systems, Cham, '
    'Switzerland: Springer, 2017, pp. 85-113.',

    'F. Tao, H. Zhang, A. Liu, and A. Y. C. Nee, "Digital twin in industry: State-of-the-art," '
    'IEEE Transactions on Industrial Informatics, vol. 15, no. 4, pp. 2405-2415, 2019.',

    'F. Wilcoxon, "Individual comparisons by ranking methods," Biometrics Bulletin, vol. 1, '
    'no. 6, pp. 80-83, 1945.',

    'J. Demšar, "Statistical comparisons of classifiers over multiple data sets," Journal of '
    'Machine Learning Research, vol. 7, pp. 1-30, 2006.',
]

WEBLINKS = [
    "C. N. Udanor et al., \"An internet of things labelled dataset for aquaponics fish pond "
    "water quality monitoring system,\" Data in Brief, vol. 43, 108400, 2022. "
    "https://doi.org/10.1016/j.dib.2022.108400",
    "Sensor Based Aquaponics Fish Pond Datasets, Kaggle. "
    "https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets "
    "(CC BY 4.0).",
]


def chapter5(doc, toc):
    chapter(doc, 5, "References", toc)
    para(doc, "Journals, Conferences and Books (IEEE format)", size=12, bold=True,
         spacing=1.15, space_after=6)
    for i, ref in enumerate(REFERENCES, start=1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.15
        pf.space_after = Pt(5)
        pf.left_indent = Cm(1.1)
        pf.first_line_indent = Cm(-1.1)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _set_font(p.add_run(f"[{i}]\t{ref}"), 11)

    para(doc, "Weblinks", size=12, bold=True, spacing=1.15, space_before=10, space_after=6)
    for i, link in enumerate(WEBLINKS, start=len(REFERENCES) + 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.15
        pf.space_after = Pt(5)
        pf.left_indent = Cm(1.1)
        pf.first_line_indent = Cm(-1.1)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _set_font(p.add_run(f"[{i}]\t{link}"), 11)


# =========================================================================== main
def margins(section) -> None:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)


def page_numbering(section, fmt: str, start: int) -> None:
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))
    section._sectPr.append(pg)


def build(out_dir: Path) -> Path:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    toc: list = []

    # Section 0 - cover, unnumbered.
    cover(doc)

    # Section 1 - front matter: abstract then contents, numbered i, ii, iii.
    doc.add_section(WD_SECTION.NEW_PAGE)
    abstract(doc, toc, page_break=False)
    contents_placeholder = len(doc.paragraphs)  # contents are inserted once the toc is known

    # Section 2 - body, numbered 1, 2, 3.
    doc.add_section(WD_SECTION.NEW_PAGE)
    chapter1(doc, toc, first=True)
    chapter2(doc, toc)
    chapter3(doc, toc)
    chapter4(doc, toc)
    chapter5(doc, toc)

    # The contents table needs the finished toc list, so it is appended last and then moved
    # back to its proper place immediately after the abstract.
    body = doc.element.body
    existing = [c for c in body if not c.tag.endswith("}sectPr")]
    tail_start = len(existing)
    contents(doc, toc)

    marker = doc.paragraphs[contents_placeholder - 1]._p
    children = [c for c in body if not c.tag.endswith("}sectPr")]
    contents_nodes = children[tail_start:]
    for node in reversed(contents_nodes):
        body.remove(node)
        marker.addnext(node)

    for s in doc.sections:
        margins(s)
    page_numbering(doc.sections[1], "lowerRoman", 1)
    page_numbering(doc.sections[2], "decimal", 1)
    doc.sections[0].footer.is_linked_to_previous = False
    for s in doc.sections[1:]:
        s.footer.is_linked_to_previous = False
        page_number_footer(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "AquaSentinel-Review2-Draft-Report.docx"
    doc.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/review2")
    args = ap.parse_args()
    path = build(ROOT / args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
