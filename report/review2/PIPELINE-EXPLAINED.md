# The training pipeline, explained

A walkthrough of **Figure 1 — `figures/fig8_pipeline_training.png`**, band by band. Open the
figure alongside this.

The companion diagram, `figures/fig9_pipeline_inference.png`, covers what happens at *run*
time — the perception heads, the control loop and the safety layer. This file is only about
how the two networks get trained.

---

## The one-sentence version

These bands turn raw river measurements into **two trained neural networks**, without any site
ever handing over its data.

---

## Band A — Acquisition

One-time and online, and the only part of the whole project that touches the network. Two
public APIs (USGS NWIS, 28 sites; UK Environment Agency, 14 sites) are pulled, mapped onto a
single channel schema, and cached as parquet: 42 sites, 729,542 rows at 15-minute cadence.

Two things to notice. Every site carries a **per-channel availability mask**, because sites
differ in which probes they own. And **no quality label is read** — public water-quality index
labels are scripted thresholds, so training on them and then reporting accuracy against them
would be circular.

---

## Band B — Windows: chopping series into training examples

A neural network cannot consume "a six-month time series". It needs fixed-size examples.

**Windowing** slides a frame along each site's series: take 36 consecutive readings as *input*,
ask the network to predict the next 24 as the *answer*, slide forward 6 steps, take another.
`stride 6` means the frames overlap rather than sitting edge to edge — more training examples
without perfect redundancy.

> **W and H are counts of steps, not clock time.** `make_windows()` slices by index. On the
> 15-minute real feed, 36 steps in / 24 out is **9 h → 6 h**. Inside the twin, whose step is
> 5 minutes, the same 36/24 is **3 h → 2 h**. The figure says both; older drafts said only the
> twin figure, which was wrong.

*"anchored on DO and temp"* — a frame is discarded unless dissolved oxygen and temperature both
have a genuine reading at the **last input step**. That step is the origin the prediction is
measured from, so if it is missing there is nothing to measure from.

### The grey box beside it is the important one

The network does **not** predict "DO will be 6.2 mg/L". It predicts "DO will *change by* +0.4",
then divides by a fixed constant (2.0 for DO) to keep the numbers near 1. Two reasons:

- A network trained on absolute values learns *"English river in April ≈ 9 mg/L"* — useless on
  a 28 °C tilapia pond. Changes transfer between water bodies; levels do not.
- The constants are **fixed, never fitted**. Fitting them would mean computing a statistic
  across everyone's data and sharing it, which quietly breaks the exact privacy boundary the
  whole project is built on.

### The three orange boxes

The same windows, reshaped for three different consumers:

| Box | Shape | Who consumes it |
|---|---|---|
| Forecast tensor | 36 × 12 | **M1**. The 12 is **6 channel values + 6 mask flags**, the mask saying "this probe actually reported here". Essential: all 42 sites report DO, only 12 report ammonium, 2 report nitrate. |
| Regime features | 16 numbers | **M3** — 4 dimensionless derived quantities × 4 window statistics (mean, std, slope, last) |
| Risk features | current channels + forecast summaries | **M4** |

---

## Band C — Federation 1: the part that gets misread

**These three boxes are not a pipeline.** Nothing flows left to right through them. They are
**three arms of one controlled experiment** — three ways of training the *same* architecture on
the *same* 42 sites, so you can ask which one wins.

**PondClient × 42.** Each real monitoring site is wrapped in an object that owns its own data
and its own copy of the model, and exposes exactly four methods: `local_train`, `get_weights`,
`set_weights`, `evaluate`. The server can ask for weights; there is no method that returns the
raw readings. That is why the isolation is called *structural* — it is enforced by the shape of
the code, not by a promise.

**M1** is the network itself: 36×12 in → LSTM with 64 hidden units, 1 layer → a Linear layer
emitting 24 × 6 = 144 numbers (24 future steps × 6 channels).

### The three arms

- **independent-local** — every site trains alone for 24 epochs and shares nothing. You end up
  with 32 separate models (32 = 42 sites − 10 held out). This is "every farm for itself", and
  it is the baseline federation has to beat.
- **FEDERATED (FedAvg)** — 12 rounds. Each round: every client trains 2 epochs on its own data
  → sends *weights* to the server → the server averages them, weighted by how much data each
  client holds → sends the average back. One global model at the end. **This is the thing being
  tested.**
- **central** — pretend privacy does not matter, pool everyone's windows, train once. Not
  deployable; it is the ceiling you would reach if centralising were allowed.

### "What makes the regimes comparable"

This box is what stops the experiment being rigged:

- **matched work** — 24 epochs = 12 rounds × 2 local epochs, so federated cannot win merely by
  training longer.
- **shared init** — every client starts from identical weights. Averaging differently
  initialised networks is meaningless, because their hidden units are in different orders.
- **10 sites held out** — never trained on by *any* arm, so the question is transfer to a pond
  nobody has seen, not recall of the ones we have. *Stratified by source* means both USGS and
  UK EA appear on both sides of the split.
- **4 data-poor clients keep 8 %** — four sites are deliberately starved of history, because
  the small farm that cannot train alone is the case federation actually exists for.

**DP-FedAvg** is an optional extra privacy layer — clip each client's update, add Gaussian
noise. Implemented, currently switched off.

---

## Band D — Federation 2: why it happens twice

The red box is the *reason*, and it is the key insight of the whole diagram.

M1 can forecast what the water does **on its own**. But the agent needs to answer a different
question: *"what happens if I switch the aerator on?"*

Rivers have no aerator, no feeding and no liming. So those 42 real sites contain **zero
examples of an action followed by a response**. You cannot learn what an aerator does from data
that contains no aerators, and no amount of federation fixes that.

So: build 8 simulated ponds, each physically different (volume ×0.55–1.8, depth ×0.75–1.45,
aeration efficiency, algal productivity, temperature regime), roll them forward under
randomised action sequences, and federate over *those*. Two more ponds are held out.

**M2** takes the observed window **plus 8 numbers describing a candidate action** and predicts
the resulting trajectory. Its encoder is **warm-started from the federated M1** — that is the
dashed arrow running down the right-hand side of the figure. That is the part worth
understanding: the half of the model that understands water comes from real rivers; only the
action-response half is learned in simulation.

The grey strip at the bottom is simply what lands on disk. Those files are what Figure 2 loads
at start-up.

---

## The thread to remember

> Real data teaches the system **what water does**. Simulation teaches it **what actions do**.
> Federation is what lets the first happen without anyone sharing data — and the encoder
> warm-start is what carries the real-data learning into the model that actually picks the
> action.

---

## Questions a reviewer is likely to ask

**"Why not just train one model on everything?"**
That is the `central` arm, and it is in the figure precisely so the comparison is honest. It is
the upper reference, not the deliverable — pooling raw records across farms is the thing the
project assumes is unacceptable.

**"Is 36 steps three hours or nine?"**
Nine on the real 15-minute feed, three inside the 5-minute twin. It is 36 *steps* either way.

**"Why predict changes instead of values?"**
Levels do not transfer from a temperate river to a warm-water pond; shapes and changes do.
And the normalising constants had to be fixed rather than fitted, or the fitting itself would
have leaked client statistics across the federated boundary.

**"Why federate twice?"**
Because the two halves of the problem have different data. What water does can be learned from
real sites; what an aerator does cannot, because the real sites have no aerators.

**"What actually crosses the network during training?"**
Weight tensors and a sample count. Nothing else — the client class exposes no method that
returns raw data.
