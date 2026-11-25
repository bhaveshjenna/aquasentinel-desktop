# Review-2 deliverables — BCSE497J Project-I

Everything in this directory is **generated**. Rebuild it with:

```bash
python scripts/make_review2_figures.py   # figures/*.png + *.svg   (Gantt, architecture, DFD, use case, class, sequence)
python scripts/make_review2_report.py    # AquaSentinel-Review2-Draft-Report.docx
python scripts/make_review2_ppt.py       # AquaSentinel-Review2-Presentation.pptx
python scripts/make_novelty_pdf.py       # AquaSentinel-Novelty-and-Base-Papers.pdf
```

The PDFs beside them are convenience exports made with Word/PowerPoint; the `.docx` and
`.pptx` are the files to upload to Moodle.

| File | What it is |
|---|---|
| `AquaSentinel-Review2-Draft-Report.docx` | 33-page initial draft report, formatted to `3 BCSE497J Project I Report - Template.docx` (Times New Roman, chapter 14 bold upper, section 13 bold title, sub-section 12 bold italic, body 12 at 1.15 spacing). Chapters 1–5, 12 tables, 8 figures, 58 references + 2 weblinks in IEEE format. References [1]–[14] are the 14 papers that also appear on the slides, in the same order, so a number means the same paper in both documents. |
| `AquaSentinel-Review2-Presentation.pptx` | 17 slides built **on top of** `4 Review 2 PPT Template.pptx`, so the SCOPE master, footer and slide numbering are the university's. Section headings, the five literature-review columns (Paper / Objective / Methodology / Pros-Cons / Findings) and the IEEE reference style are the template's own; citations are numbered [1]-[14] and are the same numbers the report uses. |
| `AquaSentinel-Pipeline-Complete.pdf` | 60-page plain-language pipeline reference. Walks every step from a sensor reading to an equipment command: all tensor shapes, every configuration constant, every safety rule in firing order, 19 inline-SVG diagrams, and 30 viva questions with answers. Opens with a full abbreviations table, and every short form (LSTM, FedAvg, RMSE, HDBSCAN, SHAP, MPC, DO, TAN…) is also spelled out at first use. Constants are read from `configs/*.yaml` and measured numbers from `outputs/*.csv` at build time, so it cannot drift from the code. Built by `scripts/make_pipeline_pdf.py`. |
| `AquaSentinel-Novelty-and-Base-Papers.pdf` | 11-page supplement: the base papers the project implements, the research gap, seven novelty claims each with its evidence file, the full pipeline diagrams and the seven-model inventory. Built by `scripts/make_novelty_pdf.py`. |
| `PIPELINE-EXPLAINED.md` | Plain-language walkthrough of Figure 1 (`fig8_pipeline_training`), band by band, plus the questions a reviewer is likely to ask about it. |
| `figures/` | The structural diagrams, including the two full-pipeline figures (`fig8_pipeline_training`, `fig9_pipeline_inference`). Result figures live in `../figures/`; dashboard screenshots in `../screenshots/`. |
| `assets/vit_logo.jpg` | Cover logo, taken from the supplied report template. |

## Mapping to the Review-2 rubric

| Rubric component | Where it is |
|---|---|
| Literature Survey and Gap Identification | Report §2.1 (5 strands, Table 1) and §2.2 (gaps G1–G5); slides 5–7 |
| Objectives, Methodology and Architecture | Report §2.3–2.5, §4.1–4.2; slides 8–9 |
| Implementation (60–75 %) and Demonstration | Report §4.3 (module status) and §4.4 (results); slides 10–14 |
| Presentation, Draft Report and Outcome | This directory; SDG and outcome line on slide 8 |

## Before you submit — four things only you can do

1. **Slide 2** still carries the template's *sample* approval e-mail. Replace it with the
   screenshot of Dr. Sivaranjani A's actual approval mail, which must be obtained before
   01-09-2026 per the guidelines.
2. **Open the report in Word, press Ctrl+A then F9 once.** The table-of-contents page numbers
   are `PAGEREF` fields; they resolve on that keypress. (They are already correct in the PDF
   export here.)
3. **Close the missing ablation cell, or be ready for the question.** `README.md` and
   `STATUS.md` claim the learned planner applies 1058 unsafe actions unguarded "against 423
   for the physics-driven one". That 423 is **not** in `outputs/ablations.csv` — the grid in
   `src/eval/run_eval.py` has `no_guardrails` (guardrails off, world model on) and
   `no_world_model` (guardrails on, world model off) but no cell with both off. The claim has
   been removed from all three Review-2 deliverables. To restore it, add a
   `{guardrails: False, world_model: False}` variant to the grid and re-run
   `python -m src.eval.run_eval`.
4. **Spot-check the reference list.** The 58 IEEE-format entries in Chapter 5 were written by
   hand, not pulled from a bibliography manager — verify volume, issue, year and page ranges
   against the original venues before submission.

## Numbers

Every quantitative claim in report §4.4 and on slides 12–14 is transcribed from `RESULTS.md`,
which `scripts/build_report.py` generates from `outputs/*.csv`. If a number here ever disagrees
with `RESULTS.md`, `RESULTS.md` is right and this is stale.
