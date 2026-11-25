"""Build the BCSE497J Project-I Review-3 presentation from the supplied VIT template.

    python scripts/make_review3_ppt.py [--out report/review3]

The Review-3 template ("report/review3/Review 3- 28-10- 2026.pptx") has the same slides as the
Review-2 one, plus an "Outcome Achieved" slide and a TRL chart on the Objectives slide. The
helpers and the shared content (literature tables, references, result tables) are imported from
make_review2_ppt.py so the two decks cannot drift apart.

Every number on the results slides comes from outputs/*.csv via results_source.py.
Slots only the team can fill (guide approval screenshot, publication status) are left as red
"ACTION" notes rather than invented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_review2_ppt as M2  # noqa: E402
import results_source as RS  # noqa: E402
from make_review2_ppt import (  # noqa: E402
    ACCENT, WARN, Lead, _bullets, add_picture_fit, add_table, add_textbox, body_placeholder,
    clone_slide_layout, drop, move_slide, note, place, set_body, set_title,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "report" / "review3" / "Review 3- 28-10- 2026.pptx"
OUT_NAME = "AquaSentinel-Review3-Presentation.pptx"
REVIEW_DATE = "28-10-2026"
APPROVAL_DEADLINE = "21-10-2026"
OUTCOME_FORM = "https://forms.gle/B9B71VPy5b9bZ9dJ8"
# Optional: drop the Review-3 approval screenshot here and it is used on slide 2.
APPROVAL_IMG = ROOT / "report" / "review3" / "assets" / "guide-approval-mail.jpg"


def stamp_footer(prs):
    """Review date on every slide, and the right cached slide number.

    Slides cloned from slide 3 carry its slide-number field with the cached text "3"; PowerPoint
    refreshes it on open, but other viewers show the cached value, so set it explicitly.
    """
    for i, slide in enumerate(prs.slides, start=1):
        for sh in slide.shapes:
            if not (sh.is_placeholder and sh.has_text_frame):
                continue
            idx = sh.placeholder_format.idx
            if idx == 10:
                tf = sh.text_frame
                tf.clear()
                r = tf.paragraphs[0].add_run()
                r.text = REVIEW_DATE
                r.font.size = Pt(12)
            elif idx == 12:
                for fld in sh._element.iter(qn("a:fld")):
                    t = fld.find(qn("a:t"))
                    if t is not None:
                        t.text = str(i)


def normalize_titles(prs, left=0.6, top=0.6, width=12.1, height=0.75):
    """Give every content-slide title the same box, padding, indent and font.

    The template's own title shapes ("object 2") carry a 0.53" paragraph indent and top
    padding, while titles on cloned slides have neither, so without this they sit at
    different x and y positions from slide to slide.
    """
    from pptx.util import Inches
    for slide in list(prs.slides)[1:]:          # the cover slide keeps its centred title
        for sh in slide.shapes:
            is_title = sh.name == "object 2" or (
                sh.is_placeholder and sh.placeholder_format.idx == 0)
            if not (is_title and sh.has_text_frame):
                continue
            sh.left, sh.top = Inches(left), Inches(top)
            sh.width, sh.height = Inches(width), Inches(height)
            body = sh.text_frame._txBody.find(qn("a:bodyPr"))
            for k in ("lIns", "tIns", "rIns", "bIns"):
                body.set(k, "0")
            body.set("anchor", "b")
            for child in list(body):
                if child.tag in (qn("a:spAutoFit"), qn("a:normAutofit"), qn("a:noAutofit")):
                    body.remove(child)
            for para in sh.text_frame.paragraphs:
                pPr = para._p.get_or_add_pPr()
                pPr.set("marL", "0")
                pPr.set("indent", "0")
                pPr.set("algn", "l")
                for el in pPr.findall(qn("a:spcBef")):
                    pPr.remove(el)
                for r in para.runs:
                    r.font.size = Pt(30)
                    r.font.bold = True
                    r.font.name = "Times New Roman"
            break


def build(out_dir: Path) -> Path:
    prs = Presentation(str(TEMPLATE))
    s = prs.slides
    R = RS.load()
    D = RS.demo()
    FED_TABLE, SAFETY_TABLE, RISK_TABLE, ABLATION_TABLE = M2.tables(R)
    unsafe_word = "Zero" if R["guarded_unsafe"] == 0 else str(R["guarded_unsafe"])
    grid = f"{R['n_scenarios']} scenarios × {R['n_controllers']} controllers × {R['n_seeds']} seeds"
    fed_verb = "beats" if R["fed_do"] < R["loc_do"] else "does not beat"
    n_tests = RS.test_count()

    # ---------------------------------------------------------------- 1. title
    slide = s[0]
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 0:
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            for i, line in enumerate(M2.TITLE_LINES):
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
            rows += [(f"{n}   ({r})", False) for n, r in M2.TEAM]
            rows += [("", False), ("Faculty guide:", True), (M2.GUIDE, False)]
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
        (f"Approval obtained by e-mail from the project guide before {APPROVAL_DEADLINE} for: "
         "meetings with the guide, the PowerPoint presentation, the project demonstration, "
         "the final project report, the project outcome, and the Turnitin plagiarism and "
         "AI scores.", 0, True),
    ], size=18)
    for sh in list(slide.shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:   # the template's sample mail
            drop(sh)
    if APPROVAL_IMG.exists():
        add_picture_fit(slide, APPROVAL_IMG, 0.6, 2.8, 12.1, 4.0)
    else:
        note(slide, "ACTION BEFORE THE REVIEW: paste the screenshot of Dr. Sivaranjani A's "
                    "Review-3 approval e-mail here (with the Turnitin plagiarism % and "
                    "AI score filled in).", 0.6, 3.2, 12.1, size=16)

    # ---------------------------------------------------------------- 3. aim
    set_body(place(body_placeholder(s[2]), 0.6, 1.5, 12.1, 5.1), [
        ("To keep an aquaculture pond inside its safe operating bands by acting before a "
         "threshold is crossed — using models trained across many farms without any farm "
         "sharing its raw data, and with hard safety limits placed outside the learned "
         "component so that a wrong model cannot cause harm.", 0, True),
        Lead("Predict, do not react — ", "a fixed-threshold aerator acts only after the limit "
             "is already crossed."),
        Lead("Learn across farms, not inside one — ", "a single pond has too little history; "
             "pooling raw data is commercially unacceptable."),
        Lead("Constrain outside the network — ", "an optimiser will find and exploit wherever "
             "its learned model is wrong."),
        Lead("Prove the effect — ", "every action is shown against a no-action counterfactual "
             "pond forked from the same seed and the same noise draw."),
    ], size=19, space=10)

    # ---------------------------------------------------------------- 4. abstract
    set_body(place(body_placeholder(s[3]), 0.6, 1.5, 12.1, 5.1), [
        Lead("Problem — ", "dissolved oxygen sags before dawn, ammonia builds after "
             "over-feeding, an algal bloom strips oxygen at night; a stocked pond can go from "
             "healthy to lethal in one unattended shift."),
        Lead("System — ", "a pond digital twin that responds to actions; an LSTM forecaster and "
             "an action-conditioned world model pre-trained by FedAvg; a model predictive "
             "planner over 14 candidate actions; a hard guardrail layer on the actuator path; "
             "a desktop operator console."),
        Lead("Data — ", f"{R['n_ponds']} real catfish aquaponics ponds in Nsukka, Nigeria, "
             f"{R['n_rows']:,} harmonised rows at 15-minute cadence from low-cost IoT probes; "
             "one pond = one federated client."),
        Lead("Result — ", f"{unsafe_word.lower()} unsafe actions applied across {grid}; "
             f"significantly lower DO time-out-of-band in {R['wins']} of {R['comparisons']} "
             f"paired comparisons; federated training {fed_verb} independent-local on the "
             "held-out ponds."),
        Lead("Trade-off — ", f"the agent costs more to operate than a thermostat in "
             f"{R['cost_worse']} of {R['cost_n']} scenarios; this is measured and reported."),
    ], size=19, space=9)

    # ---------------------------------------------------------------- 5-6. literature
    LIT_COLS = [2.15, 2.35, 2.75, 2.85, 2.65]
    slide = s[4]
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
    for sh in list(slide.shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:   # "not more than 2 slides" note
            drop(sh)
    set_title(slide, "Literature Review (1/2)")
    add_table(slide, M2.LIT_A, 0.4, 1.42, 12.75, 5.1, size=13, head_size=14,
              col_widths=LIT_COLS)

    lit2 = clone_slide_layout(prs)
    set_title(lit2, "Literature Review (2/2)")
    b = body_placeholder(lit2)
    if b is not None:
        drop(b)
    add_table(lit2, M2.LIT_B, 0.4, 1.42, 12.75, 5.1, size=13, head_size=14,
              col_widths=LIT_COLS)

    # ---------------------------------------------------------------- 7. research gap
    set_body(place(body_placeholder(s[5]), 0.6, 1.5, 12.1, 5.1), [
        Lead("G1  Prediction is never carried through to action — ", "water-quality papers "
             "stop at an error metric; none closes the loop onto an actuator and re-measures."),
        Lead("G2  Federated learning has not been evaluated on an actuation task — ",
             "held-out accuracy is reported, but not whether the model is safe to act on."),
        Lead("G3  Client heterogeneity is manufactured, not inherited — ", "non-IID structure "
             "is usually made by partitioning a benchmark by label, not taken from real farms."),
        Lead("G4  Safety is argued, not measured — ", "very few papers report how many unsafe "
             "commands the same planner would have issued with the safety layer removed."),
        Lead("G5  Scripted quality labels make results circular — ", "public datasets label "
             "by threshold, so a model scored against them has learned the threshold."),
    ], size=19, space=12)

    # ---------------------------------------------------------------- 8. objectives
    slide = s[6]
    set_body(place(body_placeholder(slide), 0.6, 1.45, 12.1, 3.2), [
        ("O1  Calibratable pond digital twin, 6 channels, 8 stress scenarios asserted by "
         "tests.", 0),
        ("O2  Real aquaculture-pond sensor data harmonised at 15-min cadence with quality "
         "control and availability masks.", 0),
        ("O3  Federated LSTM forecaster compared with independent-local on held-out ponds.", 0),
        ("O4  Action-conditioned world model, so the planner scores learned trajectories.", 0),
        ("O5  Model predictive agent behind guardrails — zero unsafe actions applied.", 0),
        ("O6  Evaluate vs. 4 baselines × 8 scenarios × 10 seeds with paired Wilcoxon tests.", 0),
        ("O7  Desktop operator console with counterfactual and SHAP rationale.", 0),
    ], size=17, space=4)
    for sh in list(slide.shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:   # SDG and TRL reference charts
            drop(sh)
        elif sh.name == "TextBox 6":
            place(sh, 0.6, 4.75, 12.1, 1.5)
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            lines = [
                ("Identified SDGs:  ", "SDG 2 (Zero Hunger)  ·  SDG 9 (Industry, Innovation "
                 "and Infrastructure)  ·  SDG 12 (Responsible Consumption and Production)  ·  "
                 "SDG 14 (Life Below Water)"),
                ("Technology Readiness Level:  ", "TRL 4 — technology validated in the lab "
                 "(models trained on real pond data, control validated on a simulated pond; "
                 "not yet field-deployed)."),
                ("Outcome targeted:  ", "Conference paper (Scopus-indexed)."),
            ]
            for i, (head, tail) in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.space_after = Pt(3)
                for text, bold in ((head, True), (tail, False)):
                    r = p.add_run()
                    r.text = text
                    r.font.size = Pt(15)
                    r.font.bold = bold
                    r.font.color.rgb = ACCENT

    # ---------------------------------------------------------------- 9. architecture
    slide = s[7]
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
    add_picture_fit(slide, M2.FIGS / "fig2_architecture.png", 0.2, 1.45, 7.9, 5.1)
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
    set_body(place(body_placeholder(s[8]), 0.6, 1.45, 12.1, 5.1), [
        Lead("FR1–2  ", "Load and harmonise real pond data to one 15-minute schema with QC, an "
             "availability mask and a per-pond licence record."),
        Lead("FR3–4  ", "Step a pond forward, respond to applied actions, and provide 8 named "
             "stress scenarios whose advertised stress is asserted by tests."),
        Lead("FR5  ", "Sample true state into a noisy observation with dropout and injectable "
             "probe faults that corrupt the reading, not the water."),
        Lead("FR6–8  ", "Forecast the 2-hour horizon; flag novelty without consuming any "
             "quality label; classify risk and attribute the score with SHAP."),
        Lead("FR9  ", "Pre-train forecaster and world model by FedAvg — weight tensors and a "
             "sample count leave a client, nothing else."),
        Lead("FR10–11  ", "Plan over 14 candidate actions through the world model; enforce "
             "caps, rate limits, contextual blocks, predicted-critical rejection and an "
             "emergency override on every tick."),
        Lead("FR12–14  ", "Run a shadow pond for the counterfactual; render the desktop "
             "console; regenerate every reported table from outputs/."),
    ], size=18, space=6)

    # ---------------------------------------------------------------- 11. modules
    set_body(place(body_placeholder(s[9]), 0.6, 1.45, 12.1, 5.1), M2.modules_bullets(R),
             size=18, space=7)

    # ---------------------------------------------------------------- 12-14. results
    slide = s[10]
    set_title(slide, "Experiments and Results (1/3)")
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
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
        lstm = Lead("A negative result — ",
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
    add_picture_fit(res3, M2.SHOTS / "2-counterfactual.png", 8.6, 3.75, 4.3, 2.6)

    # ---------------------------------------------------------------- 15. conclusion + future
    slide = s[11]
    ph = body_placeholder(slide)
    if ph is not None:
        drop(ph)
    set_body(add_textbox(slide, 0.6, 1.45, 6.2, 5.1), [
        ("Conclusion", 0, True),
        (f"Complete end to end and reproducible offline: {R['n_ponds']} real ponds, models "
         f"pre-trained by FedAvg, full evaluation grid, {n_tests} automated tests passing.", 0),
        (f"Federated pre-training {fed_verb} independent-local on held-out ponds "
         f"({RS.f3(R['fed_do'])} vs {RS.f3(R['loc_do'])} mg/L DO RMSE) — a small study: "
         f"{R['n_train_ponds']} training ponds, {R['n_holdout']} held out.", 0),
        (f"{unsafe_word} unsafe actions applied by the guardrailed agent"
         + (f"; {R['ab_noguard_unsafe']:,} without the safety layer." if
            R["ab_noguard_unsafe"] > 0 else "."), 0),
        (f"The agent costs more than a thermostat in {R['cost_worse']} of {R['cost_n']} "
         "scenarios — the price of acting early.", 0),
        ("Limits: the controlled pond is simulated, the real probes are noisy, and every "
         "claim is relative, not a field guarantee.", 0),
    ], size=16, space=7, bullet=True)
    set_body(add_textbox(slide, 7.1, 1.45, 5.7, 5.1), [
        ("Future Work (Project-II)", 0, True),
        ("More ponds from more regions, with cleaner, calibrated probes.", 0),
        ("Log real actuator actions so the world model learns from real pond responses, "
         "not simulated ones.", 0),
        ("Run DP-FedAvg with a stated privacy budget ε (implemented, not yet run).", 0),
        ("Hardware-in-the-loop trial: real probes and a real aerator driven through the "
         "guardrail layer.", 0),
        ("Tune the cost / risk trade-off with farmers, and fix or drop the LSTM "
         "correction that currently hurts control.", 0),
    ], size=16, space=7, bullet=True)
    for tb in list(slide.shapes)[-2:]:          # the two columns just added
        head = tb.text_frame.paragraphs[0]
        _bullets(head, False)
        for r in head.runs:
            r.font.size = Pt(20)
            r.font.color.rgb = ACCENT

    # ---------------------------------------------------------------- 16. outcome achieved
    slide = s[12]
    set_title(slide, "Outcome Achieved")
    set_body(place(body_placeholder(slide), 0.6, 1.5, 12.1, 3.4), [
        Lead("Targeted outcome — ", "International conference paper (Scopus-indexed)."),
        Lead("Status — ", "[FILL IN: e.g. manuscript submitted to <conference>, <date> / "
             "accepted / under preparation]"),
        Lead("Patent / Start-up — ", "None."),
        Lead("Others — ", f"working desktop prototype (AquaSentinel console), reproducible "
             f"codebase with {n_tests} automated tests, and a generated results report."),
    ], size=19, space=10)
    # Colour the placeholder status line red so it cannot be missed.
    for r in body_placeholder(slide).text_frame.paragraphs[1].runs[1:]:
        r.font.color.rgb = WARN
    note(slide, "ACTION: upload the outcome proof (submission / acceptance mail screenshot) "
                f"at {OUTCOME_FORM} — due 27 Oct 2026 — and paste it on this slide.",
         0.6, 5.1, 12.1, size=15)

    # ---------------------------------------------------------------- 17-18. references
    slide = s[13]
    set_title(slide, "References (1/2)")
    for sh in slide.shapes:
        if sh.name == "Rectangle 3":
            place(sh, 0.6, 1.42, 12.1, 5.1)
            sh.text_frame.vertical_anchor = MSO_ANCHOR.TOP
            set_body(sh, [(r, 0) for r in M2.REFS_A], size=16, space=8, bullet=False)

    refs2 = clone_slide_layout(prs)
    set_title(refs2, "References (2/2)")
    b = body_placeholder(refs2)
    if b is not None:
        drop(b)
    set_body(add_textbox(refs2, 0.6, 1.42, 12.1, 5.1), [(r, 0) for r in M2.REFS_B],
             size=16, space=8, bullet=False)

    # ---------------------------------------------------------------- order and footers
    n = len(prs.slides._sldIdLst)
    move_slide(prs, n - 4, 5)    # literature review 2/2 -> after 1/2
    move_slide(prs, n - 3, 12)   # results 2/3 -> after 1/3
    move_slide(prs, n - 2, 13)   # results 3/3
    stamp_footer(prs)
    normalize_titles(prs)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / OUT_NAME
    prs.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/review3")
    args = ap.parse_args()
    print(f"wrote {build(ROOT / args.out)}")


if __name__ == "__main__":
    main()
