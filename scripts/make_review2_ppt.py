"""Build the BCSE497J Project-I Review-2 presentation from the supplied VIT template.

    python scripts/make_review2_ppt.py [--out report/review2]

The template ("4 Review 2 PPT Template.pptx") is opened and filled in place, so the SCOPE
master, footers and slide numbering are the university's, not ours. Three extra slides are
appended and then moved into position - a second literature-review slide (the template's own
note allows two) and two more results slides.

Every number on the results slides is transcribed from RESULTS.md, which is generated from
outputs/*.csv by scripts/build_report.py.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_source as RS  # noqa: E402  - every result number comes from outputs/ via this
TEMPLATE = ROOT / "guidelines and templates" / "4 Review 2 PPT Template.pptx"
FIGS = ROOT / "report" / "review2" / "figures"
SHOTS = ROOT / "report" / "screenshots"
ASSETS = ROOT / "report" / "review2" / "assets"

REVIEW_DATE = "09-09-2026"
INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
WARN = RGBColor(0xC0, 0x00, 0x00)
HEAD_BG = RGBColor(0x1F, 0x4E, 0x79)
ROW_BG = RGBColor(0xF2, 0xF5, 0xF9)


# --------------------------------------------------------------------------- helpers
class Lead(tuple):
    """One bullet whose opening phrase is bold: Lead("Head - ", "the rest of the sentence")."""

    def __new__(cls, head, tail, level=0):
        return super().__new__(cls, (head, tail, level))


def _bullets(paragraph, on):
    """Force a bullet glyph on or off - a plain text box has neither by default."""
    pPr = paragraph._p.get_or_add_pPr()
    for tag in ("buNone", "buChar", "buAutoNum", "buFont"):
        for el in pPr.findall(qn("a:" + tag)):
            pPr.remove(el)
    if on:
        pPr.set("marL", str(Emu(Inches(0.28)).emu))
        pPr.set("indent", str(-Emu(Inches(0.28)).emu))
        font = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        char = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(font)
        pPr.append(char)
    else:
        pPr.set("marL", "0")
        pPr.set("indent", "0")
        pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def set_body(shape, items, size=16, space=8, colour=INK, bullet=None):
    """Fill a content placeholder.

    Items are strings, (text, level), (text, level, bold), or Lead(head, tail, level).
    `bullet` forces bullet glyphs on (True) or off (False); None inherits the placeholder's.
    """
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, Lead):
            head, tail, level = item
            segments = [(head, True), (tail, False)]
        elif isinstance(item, tuple):
            text, level, bold = (item + (0, False))[:3]
            segments = [(text, bold)]
        else:
            text, level, segments = item, 0, [(item, False)]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(space)
        if bullet is not None:
            _bullets(p, bullet)
        for text, bold in segments:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(size - 2 * level)
            r.font.bold = bold
            r.font.color.rgb = colour
    return tf


def place(shape, left, top, width, height):
    """Move/resize a placeholder, in inches."""
    shape.left, shape.top = Inches(left), Inches(top)
    shape.width, shape.height = Inches(width), Inches(height)
    return shape


def set_title(slide, text, size=30):
    """Write a slide title in the template's own style: Times New Roman, bold, 30 pt."""
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 0:
            place(sh, 0.5, 0.74, 11.5, 0.63)
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = text
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.name = "Times New Roman"
            return sh
    return None


def body_placeholder(slide):
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 1:
            return sh
    return None


def drop(shape):
    shape._element.getparent().remove(shape._element)


def add_table(slide, rows, left, top, width, height, size=10, head_size=10,
              col_widths=None, first_col_bold=False):
    n, m = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n, m, Inches(left), Inches(top), Inches(width), Inches(height))
    t = gf.table
    if col_widths:
        for j, w in enumerate(col_widths):
            t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(height / n)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = Inches(0.06)
            c.margin_right = Inches(0.06)
            c.margin_top = Inches(0.02)
            c.margin_bottom = Inches(0.02)
            # A cell may hold several lines (the Pros/Cons column does); each becomes its
            # own paragraph, because a run does not carry line breaks.
            for k, line in enumerate(str(val).split("\n")):
                p = c.text_frame.paragraphs[0] if k == 0 else c.text_frame.add_paragraph()
                r = p.add_run()
                r.text = line
                r.font.size = Pt(head_size if i == 0 else size)
                r.font.bold = (i == 0) or (first_col_bold and j == 0)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
    return t


def add_picture_fit(slide, path: Path, left, top, max_w, max_h):
    if not path.exists():
        return None
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    pw, ph = w * scale, h * scale
    return slide.shapes.add_picture(
        str(path),
        Inches(left + (max_w - pw) / 2),
        Inches(top + (max_h - ph) / 2),
        Inches(pw), Inches(ph),
    )


def note(slide, text, left, top, width, size=11, colour=WARN, bold=True):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = colour
    return tb


def clone_slide_layout(prs, layout_index=1, footer_from=2):
    """Add a slide and copy the template's date/footer/slide-number shapes onto it.

    Slides created from a layout do not inherit the header-footer shapes the template's own
    slides carry, so without this the new slides lose the SCOPE footer and slide number.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[layout_index])
    src = prs.slides[footer_from]
    tree = slide.shapes._spTree
    next_id = 1 + max(
        [int(e.get("id")) for e in tree.iter() if e.tag.endswith("}cNvPr") and e.get("id")]
        or [1]
    )
    for sh in src.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx in (10, 11, 12):
            el = copy.deepcopy(sh._element)
            for cnv in el.iter():
                if cnv.tag.endswith("}cNvPr"):
                    cnv.set("id", str(next_id))
                    next_id += 1
            tree.append(el)
    return slide


def move_slide(prs, old_index, new_index):
    lst = prs.slides._sldIdLst
    ids = list(lst)
    lst.remove(ids[old_index])
    lst.insert(new_index, ids[old_index])


def stamp_dates(prs):
    for slide in prs.slides:
        for sh in slide.shapes:
            if sh.is_placeholder and sh.placeholder_format.idx == 10 and sh.has_text_frame:
                tf = sh.text_frame
                tf.clear()
                r = tf.paragraphs[0].add_run()
                r.text = REVIEW_DATE
                r.font.size = Pt(12)


# =========================================================================== content
TITLE_LINES = [
    "B.Tech. — BCSE497J Project-I",
    "AI-Driven Smart Aquaculture Monitoring and Risk Prediction System",
]
TEAM = [
    ("Sanjeev Karnatapu", "23BAI0003"),
    ("Nishant Ray", "23BAI0065"),
    ("Soham Swarup Dutta", "23BCE0549"),
]
GUIDE = "Dr. Sivaranjani A, Assistant Professor Senior Grade 1, SCOPE"

# The five column headings are the template's own (Paper / Objective / Methodology /
# Pros-Cons / Findings) and the citation numbers run [1]..[14] in order of first appearance,
# matching the reference slides.
LIT_HEAD = ["Paper", "Objective", "Methodology", "Pros / Cons", "Findings"]

LIT_A = [
    LIT_HEAD,
    ["[1] Føre et al. (2018)",
     "Define precision fish farming",
     "Conceptual monitor–model–control framework",
     "+ Framing adopted here\n− No implemented controller",
     "Aquaculture should be run as a closed loop"],
    ["[2] Zhou et al. (2018)",
     "Review intelligent feeding control",
     "Survey of vision- and acoustics-driven feeders",
     "+ Feed rate is actionable\n− Isolated from DO and pH",
     "Feeding drives ammonia, so it belongs in the action set"],
    ["[3] Liu et al. (2013)",
     "Predict pond dissolved oxygen",
     "LS-SVR tuned by improved particle swarm optimisation",
     "+ Uses routine sensors only\n− One site; no transfer",
     "DO is predictable hours ahead of a breach"],
    ["[4] Ta and Wei (2018)",
     "Predict DO in a recirculating system",
     "Convolutional network over sensor windows",
     "+ Beats shallow models\n− Centralised training only",
     "Learned windows capture DO dynamics well"],
    ["[5] Ren et al. (2020)",
     "Improve multi-step DO prediction",
     "Deep belief network with supervised fine-tuning",
     "+ Better multi-step accuracy\n− Forecast never used for control",
     "Depth helps, but the loop is left open"],
    ["[6] Barzegar et al. (2020)",
     "Forecast water-quality variables",
     "Hybrid CNN–LSTM on multivariate series",
     "+ Strongest reported family\n− Absolute targets; no transfer",
     "Recurrent hybrids lead on this task"],
    ["[7] Hu et al. (2019)",
     "Predict mariculture water quality",
     "Correlation-aware multivariate deep LSTM",
     "+ Nearest architecture to ours\n− Single farm; no privacy path",
     "Cross-channel correlation improves accuracy"],
]

LIT_B = [
    LIT_HEAD,
    ["[8] McMahan et al. (2017)",
     "Train models without pooling raw data",
     "FedAvg: average client weights each round",
     "+ Aggregation rule used here\n− Assumes broadly IID clients",
     "Weight averaging can match pooled training"],
    ["[9] Zhao et al. (2018)",
     "Quantify the cost of non-IID clients",
     "Weight-divergence analysis on partitioned benchmarks",
     "+ Motivates leave-sites-out\n− Skew simulated, not real",
     "Accuracy falls sharply as client skew rises"],
    ["[10] Durrant et al. (2022)",
     "Apply cross-silo FL in agri-food",
     "Farms as silos sharing models over a broker",
     "+ Closest sector precedent\n− No closed-loop control",
     "Farms will share models, but not raw data"],
    ["[11] Camacho and Bordons (2007)",
     "Control a plant by optimising over a model",
     "MPC: roll candidates forward, score, apply the first step",
     "+ Planner implemented here\n− Assumes the model is trustworthy",
     "Receding-horizon planning needs a model it can trust"],
    ["[12] Hewing et al. (2020)",
     "Survey learning-based predictive control",
     "Learned dynamics inside an MPC optimiser",
     "+ Predicts our unguarded result\n− Assumes bounded model error",
     "An optimiser exploits wherever its model is wrong"],
    ["[13] Wabersich and Zeilinger (2021)",
     "Guarantee safety for learned control",
     "Predictive safety filter placed outside the policy",
     "+ Guardrail design adopted\n− Needs a trusted backup model",
     "Constraints must sit outside the learned component"],
    ["[14] Ha and Schmidhuber (2018)",
     "Plan inside a learned model of the world",
     "Action-conditioned recurrent dynamics model",
     "+ Rollouts without the real system\n− Simulation only",
     "Action-conditioned models support planning"],
]

# The template asks for bullet points here, not a table.
def modules_bullets(R):
    return [
        Lead("Data acquisition (src/data/) — ",
             "Kaggle pond loader, harmoniser with QC and availability masks, licence manifest. "
             f"Done: {R['n_ponds']} real ponds, {R['n_rows']:,} rows."),
        Lead("Digital twin (src/twin/) — ",
             "Pond dynamics, 8 stress scenarios, risk bands, calibration hooks. Done."),
        Lead("Sensor simulation (src/sensors/) — ",
             "Measurement noise, dropout and injectable probe faults. Done."),
        Lead("Perception (src/perception/) — ",
             "LSTM forecaster, HDBSCAN novelty detection, LightGBM risk classifier, "
             "SHAP attribution. Done."),
        Lead("World model (src/perception/) — ",
             "Action-conditioned dynamics with a federated encoder. Done."),
        Lead("Federated learning (src/federated/) — ",
             "PondClient, FedAvg aggregation, 3 training regimes, optional DP-SGD. "
             "Done; DP implemented but unrun."),
        Lead("Agent (src/agent/) — ",
             "Reactive and predictive planners, guardrail layer, actuators. Done."),
        Lead("Evaluation (src/eval/) — ",
             "Baselines, metrics, paired statistics and the ablation grid. Done."),
        Lead("Operator console (desktop/) — ",
             "Live, Counterfactual, Agent and Scorecard views. Done."),
    ]


READING = {
    "no_lstm_correction": "Planner without the LSTM correction",
    "no_forecast_at_all": "No forecaster loaded at all",
    "weights_independent_local": "Solo-pond forecaster weights",
    "weights_central": "Pooled forecaster weights",
    "no_world_model": "Hand-written physics inside the planner",
}


def tables(R):
    fed = [["Regime", "RMSE — DO", "RMSE — temp", "RMSE — pH"]] + [
        [r[0].replace(" (pools all data; breaks the privacy premise)", " (pools data)")
             .replace(" (FedAvg, no raw data shared)", " (FedAvg)"), *r[1:]]
        for r in R["fed_rows"]]
    safety = [["Controller", "Runs", "Unsafe applied", "Guardrail events"]] + [
        [r[0].replace(" (deliberately unguarded reference)", " (unguarded ref.)")
             .replace(" (the contribution)", " (ours)")
             .replace(" (privileged information)", " (privileged)"), *r[1:]]
        for r in R["safety_rows"]]
    # Scenarios where every controller scores zero carry no information on a slide.
    risk = [["Scenario", "do_nothing", "static", "reactive", "predictive"]] + [
        r for r in R["risk_rows"] if any(float(v) != 0 for v in r[1:])]
    ablation = [["Ablation", "DO risk (h)", "Cost", "Unsafe", "Reading"]]
    for name, do, cost, unsafe, _runs in R["ablation_rows"]:
        if name.startswith("full"):
            reading = "Shipped: federated + forecast + guardrails"
            name = "full (shipped)"
        elif name == "no_guardrails":
            reading = (f"Not a real win — {unsafe} unsafe actions" if int(unsafe) > 0
                       else "No unsafe actions even unguarded")
        else:
            reading = READING.get(name, "")
        ablation.append([name, do, cost, unsafe, reading])
    return fed, safety, risk, ablation


# IEEE style, exactly as the template's sample entries are set out, numbered [1]..[14] to
# match the citation keys in the literature-review tables.
REFS_A = [
    "[1] M. Føre, K. Frank, T. Norton, E. Svendsen, J. A. Alfredsen, T. Dempster, H. Eguiraun, W. "
    "Watson, A. Stahl, L. M. Sunde, C. Schellewald, K. Skøien, M. O. Alver, and D. Berckmans, "
    "\"Precision fish farming: A new framework to improve production in aquaculture,\" Biosystems "
    "Engineering, vol. 173, pp. 176-193, 2018.",
    "[2] C. Zhou, D. Xu, L. Chen, S. Zhang, C. Sun, X. Yang, and Y. Wang, \"Intelligent feeding "
    "control methods in aquaculture with an emphasis on fish: a review,\" Reviews in Aquaculture, "
    "vol. 10, no. 4, pp. 975-993, 2018.",
    "[3] S. Liu, L. Xu, D. Li, Q. Li, Y. Jiang, H. Tai, and L. Zeng, \"Prediction of dissolved "
    "oxygen content in river crab culture based on least squares support vector regression "
    "optimized by improved particle swarm optimization,\" Computers and Electronics in "
    "Agriculture, vol. 95, pp. 82-91, 2013.",
    "[4] X. Ta and Y. Wei, \"Research on a dissolved oxygen prediction method for recirculating "
    "aquaculture systems based on a convolution neural network,\" Computers and Electronics in "
    "Agriculture, vol. 145, pp. 302-310, 2018.",
    "[5] J. Ren, Y. Wang, W. Zhao, and D. Li, \"Research of dissolved oxygen prediction in "
    "recirculating aquaculture systems based on deep belief network,\" Aquacultural Engineering, "
    "vol. 90, Art. no. 102085, 2020.",
    "[6] R. Barzegar, M. T. Aalami, and J. Adamowski, \"Short-term water quality variable "
    "prediction using a hybrid CNN-LSTM deep learning model,\" Stochastic Environmental Research "
    "and Risk Assessment, vol. 34, pp. 415-433, 2020.",
    "[7] Z. Hu, Y. Zhang, Y. Zhao, M. Xie, J. Zhong, Z. Tu, and J. Liu, \"A water quality "
    "prediction method based on the deep LSTM network considering correlation in smart "
    "mariculture,\" Sensors, vol. 19, no. 6, Art. no. 1420, 2019.",
]

REFS_B = [
    "[8] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, \"Communication-efficient "
    "learning of deep networks from decentralized data,\" in Proc. 20th International "
    "Conference on Artificial Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, USA, "
    "2017, pp. 1273-1282.",
    "[9] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, \"Federated learning with "
    "non-IID data,\" arXiv preprint arXiv:1806.00582, 2018.",
    "[10] A. Durrant, M. Markovic, D. Matthews, D. May, J. Enright, and G. Leontidis, \"The role "
    "of cross-silo federated learning in facilitating data sharing in the agri-food sector,\" "
    "Computers and Electronics in Agriculture, vol. 193, Art. no. 106648, 2022.",
    "[11] E. F. Camacho and C. Bordons, Model Predictive Control, 2nd ed. London, U.K.: "
    "Springer-Verlag, 2007.",
    "[12] L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, \"Learning-based model "
    "predictive control: Toward safe learning in control,\" Annual Review of Control, Robotics, "
    "and Autonomous Systems, vol. 3, pp. 269-296, 2020.",
    "[13] K. P. Wabersich and M. N. Zeilinger, \"A predictive safety filter for learning-based "
    "control of constrained nonlinear dynamical systems,\" Automatica, vol. 129, Art. no. 109597, "
    "2021.",
    "[14] D. Ha and J. Schmidhuber, \"Recurrent world models facilitate policy evolution,\" in "
    "Advances in Neural Information Processing Systems 31, Montréal, Canada, 2018, pp. 2450-2462.",
]


# =========================================================================== build
def build(out_dir: Path) -> Path:
    prs = Presentation(str(TEMPLATE))
    s = prs.slides
    R = RS.load()
    D = RS.demo()
    FED_TABLE, SAFETY_TABLE, RISK_TABLE, ABLATION_TABLE = tables(R)
    unsafe_word = "Zero" if R["guarded_unsafe"] == 0 else str(R["guarded_unsafe"])
    grid = f"{R['n_scenarios']} scenarios × {R['n_controllers']} controllers × {R['n_seeds']} seeds"

    # ---------------------------------------------------------------- 1. title
    slide = s[0]
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 0:
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            for i, line in enumerate(TITLE_LINES):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.CENTER
                r = p.add_run()
                r.text = line
                r.font.size = Pt(18 if i == 0 else 25)
                r.font.bold = (i == 1)
        elif sh.name == "object 3":
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            rows = [("Team members:", True)]
            rows += [(f"{n}   ({r})", False) for n, r in TEAM]
            rows += [("", False), ("Faculty guide:", True), (GUIDE, False)]
            for i, (text, bold) in enumerate(rows):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.CENTER
                r = p.add_run()
                r.text = text
                r.font.size = Pt(18)
                r.font.bold = bold

    # ---------------------------------------------------------------- 2. approval
    slide = s[1]
    place(body_placeholder(slide), 0.6, 1.5, 12.1, 1.2)
    set_body(body_placeholder(slide), [
        ("Approval obtained by e-mail from the project guide before 01-09-2026 for the "
         "attendance sheet, the presentation and demonstration, the initial draft of the "
         "project report, and the outcome report.", 0, True),
    ], size=18)
    # The template ships a sample approval mail on this slide; swap it for the real one.
    approval = ASSETS / "guide-approval-mail.jpg"
    for sh in list(slide.shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            drop(sh)
    if approval.exists():
        add_picture_fit(slide, approval, 0.6, 2.7, 12.1, 4.1)
    else:
        note(slide, "ACTION BEFORE THE REVIEW: paste the screenshot of your guide's "
                    "approval e-mail here.", 0.6, 2.5, 12.1, size=15)

    # ---------------------------------------------------------------- 3. aim
    slide = s[2]
    set_body(place(body_placeholder(slide), 0.6, 1.5, 12.1, 5.1), [
        ("To keep an aquaculture pond inside its safe operating bands by acting before a "
         "threshold is crossed — using models trained across many farms without any farm "
         "sharing its raw data, and with hard safety limits placed outside the learned "
         "component so that a wrong model cannot cause harm.", 0, True),
        ("Predict, do not react — a fixed-threshold aerator acts only after the limit is "
         "already crossed.", 0),
        ("Learn across farms, not inside one — a single pond has too little history; pooling "
         "raw data is commercially unacceptable.", 0),
        ("Constrain outside the network — an optimiser will find and exploit wherever its "
         "learned model is wrong.", 0),
        ("Prove the effect — every action is shown against a no-action counterfactual pond "
         "forked from the same seed and the same noise draw.", 0),
    ], size=19, space=10)

    # ---------------------------------------------------------------- 4. abstract
    slide = s[3]
    set_body(place(body_placeholder(slide), 0.6, 1.5, 12.1, 5.1), [
        ("Problem — dissolved oxygen sags before dawn, ammonia builds after over-feeding, an "
         "algal bloom strips oxygen at night; a stocked pond can go from healthy to lethal in "
         "one unattended shift.", 0),
        ("System — a physics-lite pond digital twin that responds to actions; an LSTM "
         "forecaster and an action-conditioned world model pre-trained by FedAvg; a model "
         "predictive planner over 14 candidate actions; a hard guardrail layer on the actuator "
         "path.", 0),
        (f"Data — {R['n_ponds']} real catfish aquaponics ponds in Nsukka, Nigeria (one region), "
         f"{R['n_rows']:,} harmonised rows at 15-minute cadence from low-cost IoT probes; one "
         "pond = one federated client.", 0),
        (f"Result — {unsafe_word.lower()} unsafe actions applied across {grid}; significantly "
         f"lower DO time-out-of-band in {R['wins']} of {R['comparisons']} paired comparisons; "
         f"federated training {'beats' if R['fed_do'] < R['loc_do'] else 'does not beat'} "
         "independent-local on the held-out ponds.", 0),
        (f"Honest counterweight — the agent costs more to operate than a thermostat in "
         f"{R['cost_worse']} of {R['cost_n']} scenarios, and that is measured and reported "
         "rather than omitted.", 0),
    ], size=19, space=9)

    # ---------------------------------------------------------------- 5-6. literature
    # The template's own five headings, two slides, as its note allows.
    LIT_COLS = [2.15, 2.35, 2.75, 2.85, 2.65]
    slide = s[4]
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
    for sh in list(slide.shapes):
        if sh.shape_type == 13:      # the template's "not more than 2 slides" note
            drop(sh)
    set_title(slide, "Literature Review (1/2)")
    add_table(slide, LIT_A, 0.4, 1.42, 12.75, 5.1, size=13, head_size=14,
              col_widths=LIT_COLS)

    lit2 = clone_slide_layout(prs)
    set_title(lit2, "Literature Review (2/2)")
    b = body_placeholder(lit2)
    if b is not None:
        drop(b)
    add_table(lit2, LIT_B, 0.4, 1.42, 12.75, 5.1, size=13, head_size=14,
              col_widths=LIT_COLS)

    # ---------------------------------------------------------------- 7. research gap
    slide = s[5]
    set_body(place(body_placeholder(slide), 0.6, 1.5, 12.1, 5.1), [
        ("G1  Prediction is never carried through to action — water-quality papers stop at an "
         "error metric; none closes the loop onto an actuator and re-measures.", 0),
        ("G2  Federated learning has not been evaluated on an actuation task — held-out "
         "accuracy is reported, but not whether the model is safe to act on.", 0),
        ("G3  Client heterogeneity is manufactured, not inherited — non-IID structure is "
         "usually made by partitioning a benchmark by label, not taken from real farms.", 0),
        ("G4  Safety is argued, not measured — very few papers report how many unsafe commands "
         "the same planner would have issued with the safety layer removed.", 0),
        ("G5  Scripted quality labels make results circular — public datasets label by "
         "threshold, so a model scored against them has learned the threshold.", 0),
    ], size=19, space=12)

    # ---------------------------------------------------------------- 8. objectives
    slide = s[6]
    set_body(place(body_placeholder(slide), 0.6, 1.45, 12.1, 3.3), [
        ("O1  Calibratable pond digital twin, 6 channels, 8 stress scenarios asserted by "
         "tests.", 0),
        ("O2  Real aquaculture-pond sensor data from one region, harmonised at 15-min cadence "
         "with quality control and availability masks.", 0),
        ("O3  Federated LSTM forecaster compared with independent-local on a held-out pond, "
         "under matched work and a shared initialisation.", 0),
        ("O4  Action-conditioned world model, so the planner scores learned trajectories.", 0),
        ("O5  Model predictive agent behind guardrails — zero unsafe actions applied.", 0),
        ("O6  Evaluate vs. 4 baselines × 8 scenarios × 10 seeds with paired Wilcoxon tests, "
         "reporting cost as well as benefit.", 0),
        ("O7  Operator console with counterfactual and SHAP rationale.", 0),
    ], size=17, space=5)
    for sh in list(slide.shapes):
        if sh.shape_type == 13:      # SDG reference charts supplied with the template
            drop(sh)
        elif sh.name == "TextBox 6":
            place(sh, 0.6, 4.95, 12.1, 0.9)   # clear of the objectives list above
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            for i, line in enumerate([
                "Identified Sustainable Development Goals:  SDG 2 (Zero Hunger)  ·  "
                "SDG 9 (Industry, Innovation and Infrastructure)  ·  "
                "SDG 12 (Responsible Consumption and Production)  ·  SDG 14 (Life Below Water)",
                "Outcome targeted:  Conference paper (Scopus-indexed).",
            ]):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                r = p.add_run()
                r.text = line
                r.font.size = Pt(15)
                r.font.bold = (i == 1)
                r.font.color.rgb = ACCENT

    # ---------------------------------------------------------------- 9. architecture
    slide = s[7]
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
    # The figure is nearly square, so it is given the full slide height on the left and the
    # commentary sits in a column beside it rather than underneath.
    add_picture_fit(slide, FIGS / "fig2_architecture.png", 0.2, 1.45, 7.9, 5.1)
    set_body(add_textbox(slide, 8.2, 1.45, 4.9, 5.1), [
        ("Four lanes, and the separation is the design:", 0, True),
        (f"Online, once — the pond dataset is downloaded and harmonised ({R['n_ponds']} ponds "
         "pass QC); the only network step.", 0),
        ("Offline pre-training — one pond = one PondClient; FedAvg receives weight tensors "
         "and a sample count, nothing else.", 0),
        ("Live loop — twin → sensor → perception → MPC planner → guardrails → actuators. "
         "The planner decides slowly; the guardrails run every tick.", 0),
        ("Operator interface — desktop console and evaluation harness.", 0),
        ("", 0),
        ("Guardrails are the only component on the actuator path that cannot be bypassed.",
         0, True),
        ("The shadow twin is forked at t = 0 with the same seed and the same noise draw, so "
         "every divergence is caused by an action.", 0),
    ], size=15, space=7, bullet=True)

    # ---------------------------------------------------------------- 10. functional reqs
    slide = s[8]
    set_body(place(body_placeholder(slide), 0.6, 1.45, 12.1, 5.1), [
        ("FR1–2  Load and harmonise real pond data to one 15-minute schema with QC, an "
         "availability mask and a per-pond licence record.", 0),
        ("FR3–4  Step a pond forward, respond to applied actions, and provide 8 named stress "
         "scenarios whose advertised stress is asserted by tests.", 0),
        ("FR5  Sample true state into a noisy observation with dropout and injectable probe "
         "faults that corrupt the reading, not the water.", 0),
        ("FR6–8  Forecast the 2-hour horizon; flag novelty without consuming any quality "
         "label; classify risk and attribute the score with SHAP.", 0),
        ("FR9  Pre-train forecaster and world model by FedAvg — weight tensors and a sample "
         "count leave a client, nothing else.", 0),
        ("FR10–11  Plan over 14 candidate actions through the world model; enforce caps, rate "
         "limits, contextual blocks, predicted-critical rejection and an emergency override on "
         "every tick.", 0),
        ("FR12–14  Run a shadow pond for the counterfactual; render the console; regenerate "
         "every reported table from outputs/.", 0),
    ], size=18, space=6)

    # ---------------------------------------------------------------- 11. modules
    # Bullet points, as the template asks for - not a table.
    slide = s[9]
    set_body(place(body_placeholder(slide), 0.6, 1.45, 12.1, 5.1), modules_bullets(R),
             size=18, space=7)

    # ---------------------------------------------------------------- 12-14. results
    # On each of the three results slides the commentary is grouped into one bullet block at
    # the top and every table sits underneath it.
    slide = s[10]
    set_title(slide, "Experiments and Results (1/3)")
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
    fed_verb = "beats" if R["fed_do"] < R["loc_do"] else "does not beat"
    cen_rel = ("matches" if R["fed_do"] <= R["cen_do"] else
               f"is within {(R['fed_do'] / R['cen_do'] - 1) * 100:.1f} % of")
    set_body(add_textbox(slide, 0.6, 1.4, 12.1, 2.9), [
        Lead("Data — ",
             f"{R['n_ponds']} real catfish ponds (Nsukka, Nigeria), {R['n_rows']:,} rows at "
             f"15-minute cadence, {R['start']} to {R['end']}. One pond = one federated client."),
        Lead("Protocol — ",
             f"{', '.join(R['train_ponds'])} train; {', '.join(R['holdout_ponds'])} "
             f"{'is' if R['n_holdout'] == 1 else 'are'} held "
             "out and never trained on. Matched total work and a shared initialisation."),
        Lead("Result — ",
             f"federation {fed_verb} independent-local on DO ({RS.f3(R['fed_do'])} vs "
             f"{RS.f3(R['loc_do'])} mg/L) and {cen_rel} the pooled reference. FedAvg ran "
             f"{R['rounds']} rounds, mean local loss {R['loss_first']:.1f} → "
             f"{R['loss_last']:.1f}. Absolute error is high: the probes are noisy."),
        Lead("Data-poor ponds — ",
             f"{', '.join(R['poor_ponds'])} kept {R['poor_frac']:.0%} of their history; "
             f"federation improved DO RMSE for {R['poor_won']} of {R['poor_n']} "
             f"({R['poor_changes']} versus training alone)."),
    ], size=16, space=7, bullet=True)
    add_table(slide, FED_TABLE, 1.4, 4.35, 10.5, 1.9, size=16, head_size=16,
              col_widths=[4.5, 2.0, 2.0, 2.0])

    res2 = clone_slide_layout(prs)
    set_title(res2, "Experiments and Results (2/3)")
    b = body_placeholder(res2)
    if b is not None:
        drop(b)
    omitted = [r[0] for r in R["risk_rows"] if all(float(v) == 0 for v in r[1:])]
    ties = (f"; the {R['tie_n']} ties are in {', '.join(R['tie_scenarios'])}"
            if R["tie_n"] else "")
    set_body(add_textbox(res2, 0.6, 1.4, 12.1, 2.3), [
        Lead("Safety — ",
             f"{unsafe_word.lower()} unsafe actions applied by the guardrailed controllers, "
             f"across {grid} (left table)."),
        Lead("Control — ",
             f"median DO time-out-of-band in hours across {R['n_seeds']} seeds, lower is "
             "better (right table)."
             + (f" {', '.join(omitted)} omitted: every controller scores 0.00 there."
                if omitted else "")),
        Lead("Significance — ",
             "paired Wilcoxon signed-rank across seeds puts the agent significantly ahead "
             f"(p < 0.05) in {R['wins']} of {R['comparisons']} scenario × baseline comparisons"
             + ties + "."),
        Lead("And the cost — ",
             f"the agent is more expensive than the static threshold in {R['cost_worse']} of "
             f"{R['cost_n']} scenarios. It buys preemption, and preemption means acting on "
             "excursions that would sometimes have resolved unaided."),
    ], size=15, space=6, bullet=True)
    add_table(res2, SAFETY_TABLE, 0.5, 3.95, 5.95, 2.4, size=13, head_size=13,
              col_widths=[2.85, 0.7, 1.3, 1.1])
    add_table(res2, RISK_TABLE, 6.7, 3.95, 6.1, 2.65, size=13, head_size=13,
              col_widths=[1.95, 1.15, 0.85, 1.0, 1.15])

    res3 = clone_slide_layout(prs)
    set_title(res3, "Experiments and Results (3/3)")
    b = body_placeholder(res3)
    if b is not None:
        drop(b)
    if R["ab_noguard_unsafe"] > 0:
        g = Lead("Guardrails are load-bearing — ",
                 f"{R['ab_noguard_unsafe']:,} unsafe actions applied with the safety layer "
                 f"removed, {R['ab_full_unsafe']} with it in place. That is why the hard "
                 "constraints live outside the learned component.")
    else:
        g = Lead("Guardrails as backstop — ",
                 "no unsafe actions even with the safety layer removed in this grid.")
    diff = R["ab_full"] - R["ab_nolstm"]
    if diff > 0:
        lstm = Lead("An honest negative — ",
                    f"the LSTM residual correction makes control worse ({R['ab_full']:.2f} h "
                    f"against {R['ab_nolstm']:.2f} h) and ships enabled anyway, with the price "
                    "re-measured on every run rather than tuned away.")
    else:
        lstm = Lead("The LSTM correction — ",
                    f"{R['ab_full']:.2f} h of DO time-out-of-band with it against "
                    f"{R['ab_nolstm']:.2f} h without: "
                    + ("it helps control." if diff < 0 else "no difference."))
    set_body(add_textbox(res3, 0.55, 1.4, 12.2, 2.5), [
        g,
        lstm,
        Lead("Demonstration — ",
             f"{D['scenario']} at seed {D['seed']} runs {D['ticks']} ticks with {D['unsafe']} "
             f"unsafe applied and {D['critical_min']:.0f} min critical: {D['risk_h']:.1f} h "
             f"time-in-risk against {D['shadow_risk_h']:.1f} h for the shadow pond on the "
             "identical noise realisation."),
    ], size=15, space=6, bullet=True)
    add_table(res3, ABLATION_TABLE, 0.45, 3.8, 7.9, 2.45, size=13, head_size=13,
              col_widths=[2.4, 0.95, 0.7, 0.8, 3.05])
    add_picture_fit(res3, SHOTS / "2-counterfactual.png", 8.6, 3.75, 4.3, 2.6)

    # ---------------------------------------------------------------- 15. conclusion
    slide = s[11]
    set_body(place(body_placeholder(slide), 0.6, 1.45, 12.1, 5.1), [
        (f"The system is complete end to end and reproducible offline: {R['n_ponds']} real "
         "fish ponds from one region, models pre-trained by FedAvg, agent evaluated across "
         f"the full grid, {RS.test_count()} automated tests passing.", 0),
        (f"Federated pre-training {'beats' if R['fed_do'] < R['loc_do'] else 'does not beat'} "
         f"independent-local on the held-out ponds ({RS.f3(R['fed_do'])} vs "
         f"{RS.f3(R['loc_do'])} mg/L DO RMSE) — a small study: {R['n_train_ponds']} training "
         f"ponds, {R['n_holdout']} "
         "held out.", 0),
        (f"Safety: {unsafe_word.lower()} unsafe actions applied by the guardrailed agent "
         f"across {grid}"
         + (f"; {R['ab_noguard_unsafe']:,} without the safety layer." if
            R["ab_noguard_unsafe"] > 0 else "."), 0),
        (f"Reported even though it does not flatter the system: the agent costs more than a "
         f"thermostat in {R['cost_worse']} of {R['cost_n']} scenarios.", 0),
        ("Scope honestly stated: the controlled pond is simulated, the real data comes from "
         "noisy low-cost probes, and every claim is relative rather than a real-world "
         "guarantee.", 0),
        ("Project-II: more ponds and cleaner probes, a logged record of real actions so the "
         "world model can learn from real responses, DP-FedAvg with a real ε, and "
         "hardware-in-the-loop validation.", 0, True),
    ], size=19, space=8)

    # ---------------------------------------------------------------- 16-17. references
    slide = s[12]
    set_title(slide, "References (1/2)")
    for sh in slide.shapes:
        if sh.name == "Rectangle 3":
            place(sh, 0.6, 1.42, 12.1, 5.1)
            sh.text_frame.vertical_anchor = MSO_ANCHOR.TOP
            set_body(sh, [(r, 0) for r in REFS_A], size=16, space=8, bullet=False)

    refs2 = clone_slide_layout(prs)
    set_title(refs2, "References (2/2)")
    b = body_placeholder(refs2)
    if b is not None:
        drop(b)
    set_body(add_textbox(refs2, 0.6, 1.42, 12.1, 5.1), [(r, 0) for r in REFS_B],
             size=16, space=8, bullet=False)

    # ---------------------------------------------------------------- order and dates
    n = len(prs.slides._sldIdLst)
    move_slide(prs, n - 4, 5)    # literature review 2/2 -> after slide 5
    move_slide(prs, n - 3, 12)   # results 2/3
    move_slide(prs, n - 2, 13)   # results 3/3
    stamp_dates(prs)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "AquaSentinel-Review2-Presentation.pptx"
    prs.save(path)
    return path


def add_textbox(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tb.text_frame.word_wrap = True
    return tb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/review2")
    args = ap.parse_args()
    path = build(ROOT / args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
