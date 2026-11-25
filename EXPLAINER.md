# AquaSentinel, explained from scratch

> **Note (25 October 2026): this document describes the earlier river-data version of the project** (42 USGS / UK Environment Agency sites, an optional LLM planner, the Streamlit dashboard as the main UI). The project now trains on **7 real catfish aquaponics ponds in Nsukka, Nigeria** (one region), the LLM planner has been removed, and the main UI is the desktop console (`python -m desktop`). The concepts below still apply; the numbers, data description and file lists do not. For current results see `RESULTS.md` and `README.md`. The river version is kept in the separate copy of this folder.

*A plain-language walkthrough for someone who has never trained a model.*

This document explains what AquaSentinel does, what every technical term in it means, and
where each idea actually lives in the code. No prior machine-learning knowledge is assumed.
Where a jargon word is unavoidable it is defined the first time it appears, in **bold**.

---

## 1. The problem, in one page

A fish pond can kill its stock overnight.

The mechanism is oxygen. During the day, algae in the water photosynthesise and *produce*
oxygen. At night there is no light, so photosynthesis stops dead — but the fish, the algae and
the bacteria all keep breathing. Oxygen is consumed for twelve hours straight with nothing
replacing it, and the low point comes just before dawn. If it drops far enough, the fish
suffocate. A farmer who checks the pond at 8 a.m. finds out too late.

The fix is not complicated in principle: switch on an aerator (a paddle or blower that mixes
air into the water) *before* the oxygen gets low, not after. The hard part is knowing when.
Oxygen is not the only thing that moves — ammonia from uneaten feed, pH swings from algal
blooms, temperature, nitrite — and they interact. Acting too early wastes electricity; acting
too late kills fish.

**AquaSentinel is a software system that watches a pond, predicts what it is about to do, and
decides what to do about it — with hard safety rules that no part of the AI can overrule.**

Because you cannot ethically run experiments on a real pond full of fish, and because nobody
publishes minute-by-minute data from commercial fish farms, the pond itself is **simulated**.
The simulation is built from published aquaculture chemistry. The *learning* half of the
system, though, is trained on real public water-quality data from 42 real monitoring stations
in American rivers and English waterways. The project is upfront that rivers are not ponds,
and a lot of its design exists specifically to handle that gap.

---

## 2. How the pieces fit together

Here is the whole system in one diagram. Read it top to bottom — that is the order things
happen, five simulated minutes at a time.

```
     ┌──────────────────────────────────────────┐
     │  THE POND (simulated)                    │   "ground truth" — what is
     │  oxygen, temperature, pH, ammonia,       │   really in the water
     │  nitrite, turbidity, algae, fish         │
     └───────────────────┬──────────────────────┘
                         │
                         ▼
     ┌──────────────────────────────────────────┐
     │  THE SENSORS                             │   adds noise, rounding, drift,
     │  turn truth into imperfect readings      │   dropouts and faults
     └───────────────────┬──────────────────────┘
                         │  ← everything below this line ONLY sees readings,
                         │    never the truth. This is deliberate.
                         ▼
     ┌──────────────────────────────────────────┐
     │  PERCEPTION                              │
     │  • forecast the next 2 hours             │
     │  • classify the risk (safe/warn/critical)│
     │  • explain which reading drove that      │
     │  • flag "this looks like nothing I know" │
     └───────────────────┬──────────────────────┘
                         ▼
     ┌──────────────────────────────────────────┐
     │  THE PLANNER                             │   tries 14 possible actions,
     │  picks the best action                   │   simulates each, picks cheapest
     └───────────────────┬──────────────────────┘
                         ▼
     ┌──────────────────────────────────────────┐
     │  THE GUARDRAILS  ← the important bit     │   fixed rules. Can veto or
     │  veto / clamp / override any proposal    │   override ANY proposal.
     └───────────────────┬──────────────────────┘
                         ▼
                  action applied to the pond
```

Two things about this diagram matter more than anything else in the project:

1. **The line after the sensors.** No part of the intelligent system is allowed to peek at the
   pond's true state. It only ever sees noisy, occasionally-missing readings — the same
   position a real farmer with real probes is in. There is a test that fails if this is ever
   violated.

2. **The guardrails are outside the AI, not inside it.** A common design is to teach the model
   "being unsafe is very expensive" and hope it learns to avoid it. This project doesn't do
   that. Safety is a separate, fixed, readable list of rules that runs *after* the model has
   spoken and can throw its answer away. You can read the rules without understanding the
   model at all.

---

## 3. Part one: the simulated pond

*Code: `src/twin/`*

### What "digital twin" means

A **digital twin** is just a computer model of a physical thing, detailed enough that you can
experiment on it instead of the real thing. Here it is a pond: a virtual body of water with
oxygen, temperature, pH, ammonia and algae in it, which evolves realistically over time.

The simulation advances in **5-minute steps**. At each step it works out how fast each quantity
is changing right now, multiplies by 5 minutes, and adds that on. This is the simplest possible
way to simulate something continuous, and it is called **Euler integration**. (`src/twin/pond.py`,
the `step()` function.)

### The chemistry

Each 5-minute step, the code computes the oxygen budget as a sum of six separate effects
(`src/twin/dynamics.py`, `do_rate`):

| Effect | Plain meaning |
|---|---|
| Reaeration | oxygen drifting in across the water surface |
| Photosynthesis | algae making oxygen — **zero at night** |
| Respiration | fish and algae breathing it out |
| BOD decay | bacteria eating waste and consuming oxygen doing it |
| Nitrification demand | bacteria converting ammonia, which also costs oxygen |
| Aeration | the aerator, if it's running |

They are kept as separate numbers rather than one total, so the dashboard can tell you *why*
oxygen fell, not just that it did.

Alongside that the simulation tracks:

- **The nitrogen cascade.** Fish excrete ammonia; bacteria convert ammonia → nitrite → nitrate.
  Ammonia and nitrite are both toxic. There's a subtlety the project handles carefully:
  ammonia exists in two chemical forms, and only one of them (the "un-ionised" form, NH₃) is
  actually poisonous. Which form dominates depends on pH and temperature. So a pond with a
  given ammonia level is harmless at pH 7 and lethal at pH 9. The code computes the toxic
  fraction explicitly, and all the safety thresholds are written against *that*, not against
  total ammonia.

- **pH and buffering.** Water resists pH change if it has enough dissolved carbonate
  ("alkalinity"). When photosynthesis strips CO₂ out of the water, pH rises; respiration pushes
  it back down. A pond whose buffer has collapsed swings much harder for the same biological
  activity — which is exactly what one of the test scenarios simulates.

- **Algae.** Modelled with **logistic growth**: a population grows fast when small, slows as it
  approaches the maximum the pond can support ("carrying capacity"), and dies off from
  overcrowding beyond that. Overcrowding death is what makes a bloom *crash*, and the crash is
  what takes the oxygen with it.

Two standard pieces of chemistry appear throughout:

- **Arrhenius temperature correction.** Biological processes speed up when it's warmer, by a
  roughly fixed factor per degree. Written as `θ^(T−20)`. It's the reason a heatwave is
  dangerous: everything consumes oxygen faster *and* warm water holds less of it.
- **Michaelis–Menten (or "Monod") kinetics.** A process that speeds up as a resource becomes
  available but then plateaus, because something else becomes the bottleneck. Used here for
  nitrification (which needs oxygen) and algal growth (which needs nitrogen).

### The daily rhythm

Light is modelled as **half of a sine wave** between sunrise and sunset, and exactly zero
outside those hours (`light_fraction`). Temperature is a smooth daily cycle peaking mid-afternoon.

That hard zero at night is the single most important line in the simulation. It is what
produces the pre-dawn oxygen sag, and the pre-dawn oxygen sag is the problem the entire project
exists to solve.

### Making it feel real: "coloured" noise

Real water measurements wobble. But *how* they wobble matters.

If you add fresh random noise at every step, the result looks like electrical interference —
jittery, and it forgets where it was a moment ago. Real environmental variation is different:
a cloudy patch, a gust of wind, a thermal layer breaking up. Those things persist for a while
and then fade back toward normal.

That behaviour is captured by an **Ornstein–Uhlenbeck process**: random noise that is pulled
back toward zero over time. It's random, but it has memory. The docstring in the code puts it
neatly: *"White noise on a 5-minute step looks like sensor jitter; OU noise looks like weather."*

### Honest calibration — and what was deliberately *refused*

This is one of the more principled things in the project, and it's worth understanding even if
you skip everything else.

The team had 42 real monitoring stations' worth of data. The obvious move is to fit as much of
the simulation to that real data as possible — more real numbers, more credible model, right?

They didn't do that, and the reasoning is sound. Rivers are not ponds. Specifically:

- **What they *did* transfer:** the *size of the short-term wobble* in each channel. Small
  15-minute-scale fluctuations come from turbulence and instrumentation, and those behave
  similarly in any body of water. So the noise levels in the simulation are measured from real
  data.
- **What they *refused* to transfer:** the daily cycle. A river's daily temperature swing is
  governed by its depth, flow and thermal mass. Measured across the 42 real sites, the median
  daily temperature swing was **0.5 °C**. A 1.5-metre earthen fish pond swings **2–4 °C**.
  Copying the river figure across would have flattened the simulated pond into a lukewarm tank
  and *deleted the dawn oxygen sag* — erasing the exact phenomenon the project is about.

The critical detail: the rejected numbers were not quietly discarded. They are written into the
config file under a heading literally named `fitted_diurnal_not_applied`, with the measured
values, the number of sites each came from, and a note ending *"Rivers are not ponds."* A
reviewer can see exactly what was measured, what was used, and what was declined and why.
Applying them anyway requires explicitly passing `--adopt-diurnal` on the command line.

### The eight stress scenarios

Rather than testing on random ponds, the project defines 8 named situations
(`configs/scenarios.yaml`), each with a fixed random seed so it reproduces exactly:

| Scenario | What goes wrong |
|---|---|
| `healthy_baseline` | Nothing. The agent should mostly leave it alone. |
| `night_do_crash` | Heavy algae + warm cloudy night → oxygen crashes before dawn |
| `overfeeding_ammonia` | Too much feed dumped in → ammonia spike |
| `algal_bloom` | Bloom, huge daytime pH surge, then die-off dragging oxygen down |
| `heatwave` | +5 °C sustained: less oxygen capacity, faster consumption, more toxic ammonia |
| `aerator_failure` | The aerator is commanded but physically doesn't work |
| `ph_swing` | The pH buffer collapses; pH becomes volatile |
| `sensor_fault` | The probes break — the water is fine, the *readings* aren't |

Crucially, the test suite **checks that each scenario actually does what it claims**
(`tests/test_twin.py`). There is a test asserting the DO crash really crosses the critical
threshold, that overfeeding really pushes toxic ammonia past its limit, that the bloom really
both supersaturates by day *and* crashes at night, and so on. A scenario that quietly stopped
being stressful — after a config tweak, say — would fail the build rather than silently making
the results look good.

---

## 4. Part two: the sensors

*Code: `src/sensors/stream.py`*

Between the pond and everything else sits a deliberate layer of degradation. Real probes:

- **round** their readings to a fixed resolution (**quantisation**)
- have a **bias** — a fixed calibration offset, so two probes in the same water disagree. This
  is drawn once per run, not per reading, because that's how real miscalibration behaves
- add random **noise** to every reading
- occasionally **drop out** and return nothing at all

All four are simulated. The result is that the AI never sees clean data.

### "Probe offline" vs "value is zero"

This distinction turns out to matter a lot. If a sensor fails and you write `0` into your data,
a model reasonably concludes the oxygen has vanished and panics. If you instead write "missing"
and pass a separate yes/no flag saying *whether this reading exists*, the model can learn to
handle the two cases differently.

The project does the latter everywhere: every reading is paired with an **availability mask**,
and that mask is fed into the neural network as extra input channels alongside the values
themselves.

### Breaking the sensor, not the water

The `sensor_fault` scenario is handled here rather than in the pond simulation, and that
placement is the whole point. In that scenario the pond is perfectly healthy — it's the *probe*
that drops out, spikes, or freezes on one value.

This creates a genuinely hard problem: the anomaly detector has to distinguish *"the pond is in
trouble"* from *"the instrument is lying"*, using only the instrument. That's the real-world
version of the problem, and the project measures how well it does rather than assuming it works.

---

## 5. Part three: the machine learning

Four models. Here's what each one is, in plain terms, before the details.

| Model | Job | Analogy |
|---|---|---|
| Forecaster (LSTM) | Predict the next 2 hours | A weather forecast for the pond |
| World model (LSTM) | Predict "what if I turn the aerator on?" | A flight simulator for decisions |
| Clustering (HDBSCAN) | Spot situations unlike anything seen before | "This doesn't look like anything I know" |
| Risk model (LightGBM) | Convert a forecast into safe / warning / critical | A traffic light |

### 5a. Forecasting — the LSTM

An **LSTM** ("Long Short-Term Memory" network) is a type of neural network built for sequences.
Unlike a model that looks at one snapshot in isolation, an LSTM reads a stretch of history in
order and carries a running summary forward — so it can pick up on trends and rhythms, not just
current values.

Here it reads **3 hours** of history (36 readings, 5 minutes apart, across 6 channels) and
predicts the **next 2 hours** (24 steps, all 6 channels at once). "Multivariate multi-step"
just means: many channels in, many time-steps out.

Two design decisions carry the whole thing, and both exist because of the rivers-aren't-ponds
problem.

**1. It predicts *change*, not *level*.**

Suppose you train a model on English rivers, where dissolved oxygen sits around 9 mg/L at 12 °C.
Then you point it at a tropical tilapia pond running at 6.5 mg/L and 28 °C. A model that learned
absolute values has learned "the answer is about 9" and is simply wrong everywhere.

But if the model instead predicts *"oxygen will fall by 0.4 mg/L over the next hour"* — a
**delta** — then what it has actually learned is the *shape* of a daily oxygen cycle. Shapes
transfer between water bodies. Levels don't.

**2. The scaling numbers are fixed constants, not fitted to the data.**

Neural networks work best when their inputs are roughly the same size, so raw values are
normally divided by something to bring them into a common range. The usual approach is to
compute that divisor from your training data (e.g. the standard deviation).

This project hard-codes the divisors in a config file instead, for two reasons:

- In the federated setup (next section), computing a divisor from pooled data would require each
  site's statistics to leave that site — which is precisely the privacy boundary the whole
  design exists to protect.
- It keeps the three training approaches strictly comparable: they all use identical
  preprocessing, so any difference in results comes from the training method and not from an
  accidental difference in data preparation.

**Measuring whether it's any good — the persistence baseline.**

A forecast is only impressive relative to the dumbest possible alternative. For time series that
alternative is **persistence**: "whatever it is now, it'll be the same in two hours." Simple, and
surprisingly hard to beat over short horizons.

The project scores itself against persistence rather than reporting an error figure in
isolation. There's a neat consequence of the delta design here: because the model predicts
*change*, the persistence baseline is literally "predict zero", so the comparison falls out of
the same numbers with no extra machinery.

### 5b. The world model — "what happens if I do X?"

The forecaster has a hard limitation: it was trained on rivers, and **rivers have no aerators**.
There is no aerator in any of the training data, so you cannot ask it "what happens if I switch
the aerator on?" It has no idea what an aerator is.

So there's a second network for that: same architecture, but it takes the recent history **plus
a proposed action** and predicts the resulting trajectory. This is called an
**action-conditioned world model** — a learned simulator of consequences.

Two details worth understanding:

- **Where the action gets injected.** The action is combined with the network's output *after*
  it has read the history, not mixed into the history itself. The reasoning is in the code
  comment: the window is the past, and the action hasn't happened yet. Feeding it into the past
  would teach the model that the action was already in effect during the observations it's
  looking at.

- **Transfer learning via warm-starting.** This model has to be trained on the *simulation*,
  since only the simulation contains aerators. But the project didn't want to abandon the real
  data entirely. So the network's front half — the part that reads and summarises the recent
  history, called the **encoder** — is *initialised with the weights from the real-data
  forecaster* rather than starting from random. That's **transfer learning**: reusing what one
  model learned as the starting point for another. The result is that the representation the
  final decision is computed from still traces back to 42 real monitoring stations, even though
  the action-response part is necessarily simulated. The project states this openly rather than
  implying the whole thing is real-data-trained.

### 5c. Clustering — and a negative result reported honestly

**Clustering** means finding natural groupings in data without anyone labelling them first.
**HDBSCAN** is a clustering algorithm with a useful property: it can decline to assign a point
to any group, marking it as *noise* instead. That "none of the above" label is exactly what you
want for anomaly detection.

The original goal was a *taxonomy of pond states* — a set of named operating regimes ("healthy
morning", "oxygen-stressed night") that the dashboard could display.

**The data doesn't support one, and the project says so.**

They swept the algorithm's settings across a wide grid — cluster size 40 to 800, several
sensitivity settings, both selection methods, with and without dimensionality reduction, on
simulation data alone and on simulation plus real data pooled. Every configuration gave the same
answer: one big blob holding ~80% of the data, a few small satellites, and a cluster-quality
score of essentially zero.

The measured result, in `outputs/regime_metrics.csv`:

- 2 clusters, 21.8% of windows marked as noise
- **DBCV = 0.0000015** — DBCV is a score for "are these clusters genuinely separated?", and
  0 means "no"
- **Silhouette = 0.52** — a *different* cluster-quality score, which looks healthy

That disagreement is itself the finding, and the code says so explicitly: silhouette flatters a
single blob arbitrarily cut in two, whereas DBCV doesn't. Reporting only the flattering number
would have been the dishonest option. The full sweep is saved to `outputs/regime_sweep.csv` so
the claim rests on evidence rather than assertion.

The honest interpretation: pond conditions form a **continuum**, not discrete states. A pond
doesn't jump between modes; it slides around smoothly.

So the model is used for what it *does* support — the **anomaly flag**. "This window is unlike
anything in training" is a real, useful signal even when the groupings aren't meaningful. Its
measured performance on the injected sensor faults, across 10 runs: **80% of fault events
detected, with an 11% false-alarm rate**. That's reported as-is, including the false alarms.

There's a nice supporting idea here too. To cluster a British river and a tropical pond in the
same space, the raw numbers are useless — they *are* the difference. So the features are made
**dimensionless** first:

- instead of "oxygen = 9 mg/L", use **oxygen ÷ the maximum the water can hold at this
  temperature**. A river at 9 mg/L and 12 °C and a pond at 6.5 mg/L and 28 °C are both about
  *85% saturated* — and that's the physically meaningful similarity.
- instead of absolute temperature, use temperature *minus the window average* — capturing the
  daily shape rather than the climate.

The code even quantifies the trade-off: with raw levels, 38% of windows get flagged as
anomalous; with the derived ones, 21%.

### 5d. The risk classifier, and explaining it

**LightGBM** is a **gradient boosting** model — a large collection of small decision trees,
each one trained to correct the mistakes of the previous ones. It's not a neural network; it's
the workhorse people reach for on table-shaped data, and it's fast and interpretable.

Its job: take the current readings plus a summary of the forecast, and output **safe / warning
/ critical**.

**What it's actually learning, stated honestly.** The correct answer it's trained against isn't
some deep learned notion of water quality. It's a *rule* — the published safety thresholds from
the aquaculture literature, applied to what the pond genuinely did over the following two hours.

So the model isn't learning chemistry. It's learning to **anticipate a known rule from an
imperfect forecast**. The code's own docstring says exactly this.

Which raises the fair question: why model a rule at all, rather than just applying it? Because
the rule needs to know the future, the future comes from a forecaster that makes mistakes, and
a hundreds-of-trees ensemble reading the *whole* forecast summary degrades far more gracefully
under forecast error than a hard threshold on one predicted number does.

One important guard: the correct answers come from what the pond *actually did*, not from what
the forecaster predicted. Training a model against its own predictions would be circular.

Measured on a held-out scenario (`outputs/risk_metrics.csv`): **97.6% accurate, and it catches
100% of genuinely critical periods** at 89% precision — meaning it never missed a critical
period, at the cost of some false alarms. For a safety system, that's the right side to err on,
and the model is explicitly configured (`class_weight="balanced"`) to prioritise it, because
critical events are rare and rare things are easy for a model to ignore.

**SHAP: getting an explanation out.**

A boosted-tree model with hundreds of trees is not something a human can read. **SHAP** is a
technique that answers, for one specific prediction: *how much did each input push this answer
up or down?* It's borrowed from game theory — it treats the inputs as players in a team and
fairly divides up the credit.

For tree models there's an exact version (`TreeExplainer`) — no approximation, and fast enough
to run on every single tick of the dashboard.

One subtle choice: the explanation is always computed for the **critical** class specifically,
even when the model's answer is "safe". The reasoning, from the code: if you explain whichever
class the model currently favours, the *question being answered changes* as the prediction
moves, and the explanation becomes incomparable over time. "What is currently pushing this pond
toward critical?" is a stable question with a stable answer, and it's the one an operator
actually wants.

The output is a human sentence like *"risk raised by forecast min DO and un-ionised NH₃ now"*,
which appears both on the dashboard and in the agent's decision log — so the operator sees the
same explanation the agent reasoned with, not a different story invented afterwards.

---

## 6. Part four: federated learning

*Code: `src/federated/`*

### The idea

Imagine 42 fish farms. Each has data. Pooling it all would train a better model than any farm
could train alone — but no farm wants to hand its operational data to a central server, or to
a competitor.

**Federated learning** solves this. Instead of moving the data to the model, you move the model
to the data:

```
  1. Server sends the current model to every farm.
  2. Each farm trains it a little, on its own data, locally. Data never leaves.
  3. Each farm sends back only the updated model — the numbers inside the network.
  4. Server averages all the updates into a new shared model.
  5. Repeat.
```

Step 4 is called **FedAvg** (federated averaging). The average is weighted by how much data
each farm had, so a farm with ten times the data has ten times the influence. Here it runs for
12 rounds, 2 local training passes per round.

### The isolation is structural, not a promise

A **client** (one farm) is a Python object that exposes exactly four operations to the outside
world: train locally, give me your model, take this model, evaluate yourself. There is no method
that hands out data, and the server code only ever receives model weights and a row count. The
privacy property isn't documentation — it's enforced by the shape of the code.

### The experiment: three ways to train

To find out whether federation actually helps, the project runs three approaches over the
*identical* set of farms:

| Approach | What it is | Verdict |
|---|---|---|
| **Independent-local** | Each farm trains alone, shares nothing | The bar federation must beat |
| **Federated** | FedAvg — models shared, data never | The thing being tested |
| **Central** | All data pooled into one model | Best case, but breaks the privacy promise |

Central is included as a reference point, clearly labelled as violating the premise. It's
reported, never recommended.

**Two fairness details that are easy to get wrong and are handled correctly here:**

1. **Matched total compute.** Independent-local gets 24 training passes; federated gets 12
   rounds × 2 passes = 24. Otherwise a win could just mean one method trained longer.

2. **Shared starting point.** Every farm begins from the *same* randomly-initialised model.
   This sounds pedantic but it's essential: a neural network's internal units have no fixed
   identity, so two networks trained from different random starts can be equally good while
   having their internals arranged completely differently. Averaging them together mixes
   unrelated parts and produces nonsense. The code comment states this reason directly.

### Non-IID data — inherited, not faked

**IID** ("independent and identically distributed") is the assumption that all your data comes
from the same underlying source. Federated learning's central difficulty is that real-world
data is **non-IID**: every participant's data is different in systematic ways, so their model
updates pull in different directions and the average can be worse than the parts.

Papers usually simulate this by artificially splitting a clean dataset. This project doesn't
need to. Its 42 clients are real stations across twelve US states and England, differing in
climate, season, catchment geology, and even **which instruments they carry** — some measure
ammonia, most don't. The messiness is inherited from geography, and the data-cleaning stage
deliberately *preserves* missing data rather than filling it in, precisely because that
missingness is part of what makes clients genuinely different.

### Testing on farms nobody trained on

The 42 sites are split: **32 for training, 10 held out entirely**. No approach trains on the
held-out ten. So the question being answered is *"does this work on a pond we've never seen?"*
rather than *"did it memorise the ponds we have?"*

The split is **stratified by source** — both US and UK stations appear on both sides — so the
result isn't an accident of one country landing entirely in one group.

One more piece of care: "independent-local" has no single model to test (there are 32 of them),
so it's scored as the **average across all 32 solo models**. Reporting only its best-performing
member would be cherry-picking.

### The results

Prediction error on the 10 unseen sites, oxygen channel — lower is better
(`outputs/federated_summary.csv`):

| Approach | Error (mg/L) |
|---|---|
| Central (pooled — privacy-violating) | 0.338 |
| **Federated** | **0.342** |
| Independent-local (alone) | 0.421 |

Federated is **19% better than training alone**, and lands within about **1%** of pooling all
the raw data together. That is the result the whole federated section exists to produce: near-pooled
accuracy with no data ever leaving a site.

### The data-poor study — where federation actually matters

Averaged results can hide the interesting part. Federation's real promise is for the
participant who *can't* train a good model alone — a farm that just installed sensors, or one
whose probes were broken for months.

So four clients are deliberately reduced to **8% of their history**, and then scored on their
own held-out data (`outputs/federated_data_poor.csv`):

| What they use | Error (mg/L) |
|---|---|
| The model they trained alone on their thin slice | 0.835 |
| **The federated model they helped build** | **0.753** |
| The pooled model | 0.664 |

A farm with almost no data gets a **10% better model** by participating — without sharing a
single row.

(Small but telling detail: the code splits off the held-out sites *before* choosing which
clients are data-poor. The other order would let a designated client land in the held-out group,
where it never trains and so never appears in this table — quietly scoring the headline claim
on fewer clients than were actually designated.)

### Differential privacy — built, and honestly labelled

There's a subtle attack on federated learning: model updates aren't raw data, but with effort
you can sometimes reconstruct information about the data from them.

**Differential privacy** defends against this by (a) limiting how much any single participant's
update can influence the result, and (b) adding a carefully calibrated amount of random noise.
The trade-off is measured by a number called **epsilon (ε)** — lower means more private, at some
cost in accuracy.

Both mechanisms are implemented here. What's notable is how ε is reported. Computing ε correctly
requires a specialist tool (an **accountant**). If that library isn't installed, the code does
*not* guess a plausible-looking number. It writes the literal string:

> `UNACCOUNTED (install dp-accounting for an RDP epsilon)`

A made-up privacy guarantee is worse than admitting you don't have one. The feature currently
ships **switched off** (`enabled: false`), and the reported results were produced without it —
which is stated rather than glossed over.

---

## 7. Part five: making decisions

*Code: `src/agent/`*

### Model Predictive Control

**MPC** is a control technique that's genuinely intuitive:

> For each action you could take, simulate forward and see what happens. Score each outcome.
> Do whichever scores best. Then, a few minutes later, throw that plan away and redo it with
> fresh information.

It's how a chess engine works, and how you plan a driving manoeuvre. The re-planning is
important — you commit to the *next* action, not the whole plan.

Here, every 15 minutes the agent simulates 2 hours ahead for each candidate action, and scores
each outcome by:

```
cost =  1.0 × (hours spent in warning or critical)
     +  3.0 × (hours spent in critical)          ← critical hurts three times as much
     + 0.06 × (electricity, chemicals, water)
     + 0.20 × (how volatile the result is)       ← stability is worth something
     + 0.35 × (lost growth from withholding feed)
```

Then it picks the lowest. That last term matters more than it looks: without a penalty for not
feeding, the mathematically optimal strategy is to *starve the fish* — they consume less oxygen
that way, so every safety metric improves. The feed penalty is what stops the agent solving the
problem by killing the business.

### 14 actions, not 108

The full space of combinations — 3 aerator levels × 3 water-exchange options × 3 feeding options
× lime on/off × salt on/off — is 108 possibilities. Simulating all of them every 15 minutes
would be slow, and most are incoherent (adding lime *and* replacing the water simultaneously) or
obviously dominated.

So a hand-picked list of **14** covers every lever and every escalation level. The stated reason
isn't just speed: *you can read the list*. Whatever the agent does, you can check it against 14
lines and confirm it was an option someone deliberately allowed. That property is worth more
than the marginal gain from searching all 108.

### The agent has its own beliefs, and they're wrong

Here's a design decision that's easy to miss and important.

For MPC to work, the agent needs something to simulate with. The lazy option is to let it
simulate using the real pond's internals — but then the agent would implicitly *know* the true
algae concentration, fish biomass and waste load, which no real farmer can measure.

Instead, the agent gets its **own separate copy** of the pond model, starting from generic
assumptions, which it nudges toward reality using only what the sensors report. The nudging is a
**complementary filter** — a simple blend, 65% "what my model predicted" + 35% "what the probe
just said". Enough to stay roughly aligned, not enough to be handed the answer.

The consequence is deliberate: the agent's beliefs about the invisible parts of the pond are
genuinely uncertain and genuinely somewhat wrong. This is called **model mismatch**, and it's
the honest situation. A controller that only works when its internal model is perfect isn't a
useful result. (A test verifies the agent's internal model can't see the scenario's schedule of
upcoming events either — no peeking at the future.)

### Blending the learned model with the physics model

The forecaster (trained on real rivers) and the internal pond model (hand-written physics)
disagree. Rather than choosing one, the agent uses the *difference between them* as a
correction:

```
correction = (LSTM's do-nothing forecast) − (physics model's do-nothing forecast)
```

That difference is a rough estimate of the physics model's systematic bias, learned from real
data. It's halved (damped) and added to every simulated outcome. This is called **residual
correction**.

**And here is the most unusual thing in the entire project.**

It makes the controller *worse*. Measurably. From `outputs/ablations.csv`, the median hours
spent with oxygen out of range:

| Configuration | Hours out of range |
|---|---|
| **`full` — what actually ships** | **10.00** |
| `no_lstm_correction` — the same agent with this switched off | **6.79** |

Turning the feature off improves the controller by about a third. A network trained on temperate
British and American rivers, applied as a correction to a tropical fish pond, injects more bias
than it removes — even halved.

It ships **on anyway**, and the config file explains the choice in full: the learned forecaster
is the component the entire federated half of the project exists to produce, and keeping it in
the decision path is what makes this a *learned* controller rather than classical control with
a neural network bolted on decoratively. The cost is measured on every single evaluation run
and published in the ablation table, and flipping one config flag gives you the better-scoring
version.

Whether you agree with the trade-off or not, it's documented rather than buried — and the
alternative (quietly disabling it and still calling the system "AI-driven") would have been the
dishonest option.

### The guardrails

*Code: `src/agent/guardrails.py`*

This is what the project describes as its actual contribution. Not that a model can predict
water chemistry — that no proposal, from any source, reaches the pond without passing a fixed,
readable checklist.

The rules run in a strict order, and each can modify or reject the proposal:

1. **Dosing caps** — no more than 2 lime doses a day, 6 hours apart; 1 salt dose; water exchange
   limited by hours per day. *(You can't keep adding chemicals to water that hasn't finished
   reacting.)*
2. **Rate limits** — the aerator can't jump instantly between settings.
3. **Contextual blocks** — never add lime to already-alkaline water. Never feed when oxygen is
   low or ammonia is high. Never do a large water exchange if the replacement water is colder
   (thermal shock) or isn't known to be better.
4. **Predicted-critical rejection** — simulate the proposal; if it would *cause* a critical
   excursion, discard it and do nothing instead. *(With a sensible exception: if the pond was
   already heading critical anyway, the action isn't to blame and isn't rejected.)*
5. **Emergency override** — if oxygen is critical now, or predicted to be, force the aerator to
   maximum and stop feeding.

The ordering is deliberate. The emergency override runs **last** so it beats every restriction
above it, including the rate limit — as the code puts it, *"when the pond is drowning, nothing
outranks air."*

**Fast and slow.** Planning is expensive, so it runs every 15 minutes. But a pond can go
critical in far less than that. So the system is split: the **planner** runs every 15 minutes,
while a **fast safety layer** — rules 3 and 5 — runs on *every* 5-minute reading against
whatever action is currently held. This slow-planner/fast-safety arrangement is the standard
pattern in industrial control, called **supervisory control**.

**Allowing for lying instruments.** Here's a subtle problem: the guardrails only see the *noisy
probe reading*, but safety is judged on the pond's *true state*. If the probe reads 2.6 mg/L
when the truth is 2.4, a threshold at 2.5 fires too late.

So each safety threshold is checked against the reading shifted in the *less safe* direction by
about **three standard deviations of the sensor noise**. Thresholds fire slightly early rather
than slightly late. (This is why the config sets a margin of 0.5 mg/L against a sensor noise
level of 0.15.)

**Does it work?** From `outputs/safety.csv`, across the full grid of 8 scenarios × 10 random
seeds — 80 runs per controller:

| Controller | Unsafe actions applied |
|---|---|
| Predictive agent | **0** |
| Reactive agent | **0** |
| Static threshold baseline | **0** |
| Oracle | **0** |
| Do-nothing *(guardrails deliberately disabled)* | 6,171 |

Zero, everywhere the guardrails are enabled. The do-nothing row is the control: guardrails are
switched off for it on purpose, because it exists as a reference trajectory showing what happens
with no intervention at all — letting the safety layer rescue it would destroy the comparison it
exists to provide.

### The counterfactual — proving the agent caused the improvement

How do you prove the agent helped, rather than that it happened to be a good night?

You run **two identical ponds**. Same starting conditions, same scenario, same random seed —
and critically, the random number generator is copied *by internal state*, so both ponds
experience the *exact same* sequence of random fluctuations. One has the agent; the other does
nothing.

Any difference between them is caused by the agent's actions and nothing else. This is a
**counterfactual** — the "what would have happened otherwise" that observational data can never
give you. It's the dashboard's headline visual and the core of the evaluation.

Two tests keep it honest: one checks that the two ponds *do* diverge when actions are taken, and
another checks that a do-nothing agent matches its own shadow *exactly* — confirming the
comparison machinery adds no spurious difference of its own.

### The optional LLM planner

There's an optional mode where a large language model makes the decision. It's constrained in
three ways:

- It gets the **same 14-item menu** and returns an *index* — it cannot invent an actuator
  command, only choose from the reviewed list.
- Its choice passes through the **identical guardrails** as everything else.
- **Any** failure — no model running, no network, malformed reply, out-of-range index — falls
  back to the normal planner rather than doing something unreviewed. Six separate tests cover
  these failure paths.

It also defaults to a *locally-run* model rather than a cloud API, because the rest of the
project works entirely offline and the config notes that switching to a hosted provider would
send pond state to a third party on every decision.

The framing is worth noting: the LLM's job is *judgement over an already-safe menu*, not
invention. That's about the only architecture in which an LLM belongs anywhere near a physical
actuator.

---

## 8. Part six: how the results are measured

*Code: `src/eval/`*

### Never report a bare average

If the agent averages 8 hours of low oxygen and the baseline averages 12, is the agent better?
You can't tell. Maybe the agent won by a mile in two runs and lost slightly in the other eight.

The project's rule, enforced in code: **every comparison is a statistical test, and `p` and `n`
are always reported.**

The test used is the **paired Wilcoxon signed-rank test**. Two parts to that:

- **Paired** — each random seed produces the *identical* conditions for every controller, so you
  can compare them head-to-head, run by run, instead of comparing two clouds of numbers. Pairing
  is far more sensitive, and here the pairing is genuinely valid rather than assumed.
- **Signed-rank** — it doesn't assume the results follow a bell curve. Simulation outputs often
  don't, and a test that assumes it can be badly misleading.

A **p-value** is the probability of seeing a difference this large by pure chance if there were
really no difference. Small p means the effect is probably real. It says nothing about whether
the effect is *large* — which is why the code also reports an effect size.

And there's a refusal built in: **with fewer than 6 seeds, the test does not run.** It returns
the observed difference with an explicit note saying there were too few samples to test. It will
not print a number that looks like evidence but isn't. Summaries elsewhere report the **median
and interquartile range** rather than the mean, so a single outlier can't quietly move the
headline.

**The headline result**, from `outputs/agent_stats.csv` — predictive agent vs. the standard
industry baseline, on hours with oxygen out of range, 10 seeds per scenario:

| Scenario | Agent | Baseline | Improvement | p |
|---|---|---|---|---|
| `ph_swing` | 12.6 h | 42.3 h | **−29.9 h** | 0.002 |
| `algal_bloom` | 19.7 h | 42.6 h | **−23.1 h** | 0.002 |
| `overfeeding_ammonia` | 1.8 h | 11.6 h | **−9.7 h** | 0.002 |
| `heatwave` | 9.5 h | 18.9 h | **−9.1 h** | 0.002 |
| `aerator_failure` | 13.0 h | 18.6 h | **−5.7 h** | 0.002 |
| `night_do_crash` | 13.3 h | 17.2 h | **−4.0 h** | 0.002 |
| `healthy_baseline` | 0 h | 0 h | tie | — |
| `sensor_fault` | 0 h | 0 h | tie | — |

Wins everywhere there was something to win, at consistent statistical significance — and
correctly reports *no difference* in the two scenarios where the pond was never actually in
danger, rather than manufacturing an improvement.

### The baseline ladder

Comparing against a deliberately terrible baseline proves nothing. This project defines five
controllers on a ladder, each labelled with exactly **what information it is allowed to see**:

| Controller | What it sees | Role |
|---|---|---|
| `do_nothing` | nothing | The floor. What happens with no intervention. |
| `static_threshold` | current oxygen only | **The real bar** — the actual industry standard: oxygen below 4 mg/L, aerator on. Nothing else. |
| `reactive` | current readings + forecast | Sensible rules across all channels |
| `predictive` | everything + simulation | The thing being evaluated |
| `oracle` | the pond's *hidden* internals | Reference only — cheats, not deployable |

The `static_threshold` row is the honest comparison, and the project is explicit that this is
the bar to clear. It's what an actual fish farm actually runs today.

**The oracle is handled unusually well.** It's given information no real farm can measure —
actual algae concentration, fish biomass, accumulated waste. You'd expect it to be presented as
the theoretical ceiling. Instead the code repeatedly insists it is **not** an upper bound: it's
a hand-written rule that happens to see everything, and it's imperfect. The proof offered is
concrete — in the `ph_swing` scenario the predictive agent *beats* the oracle on oxygen, because
the oracle's rule has no provision for over-oxygenation. Calling it a ceiling would have made
the agent's result look better by comparison, and the project explicitly declines to.

### The ablation grid

An **ablation** removes one component and re-measures, to find out what each part actually
contributes. Seven variants are run (`outputs/ablations.csv`), median hours with oxygen out of
range:

| Variant | Result | Reading |
|---|---|---|
| `no_lstm_correction` | **6.79** | Best. Learned correction off. |
| `no_forecast_at_all` | 6.79 | Same — confirms the correction is the mechanism |
| `weights_independent_local` | 8.83 | |
| `no_guardrails` | 9.38 | Slightly "better" — safety costs a little performance, as expected |
| `no_world_model` | 9.75 | The learned dynamics model also costs a little |
| **`full` (ships)** | **10.00** | |
| `weights_central` | 11.75 | Worst |

Publishing a table in which your shipped configuration is *not* the best row is unusual, and it
is the table that makes the rest of the project's claims credible.

### Reproducibility

- Every random process is seeded, so any run can be reproduced exactly.
- CPU only — no GPU needed, runs on a laptop. Numerical libraries are pinned to a single thread
  so results don't shift with machine load.
- Fully offline after one initial data download; everything else reads from local cache.
- The one place perfect determinism isn't achievable — certain neural-network operations have no
  deterministic implementation — is noted in a code comment rather than papered over.

### Every number is generated, none typed

`RESULTS.md` and the PDF report are **rendered from the CSV output files** by scripts. No figure
in any document was typed by hand.

The reason is that hand-maintained tables drift: you re-run an experiment, the numbers change,
and the document quietly keeps the old ones. Here, if a results file is missing, the report
prints *the command that generates it* rather than inventing a placeholder.

---

## 9. Part seven: the data pipeline

*Code: `src/data/`*

### Where the data comes from

Two public sources, both free and requiring no registration:

- **USGS NWIS** — the United States Geological Survey's water monitoring network
- **UK Environment Agency** — the English equivalent

**42 stations**, 28 US and 14 UK, roughly 730,000 rows total, six months at 15-minute intervals.
Downloaded once and cached locally as files, after which the entire project works offline. The
download is *idempotent* — re-running it never re-downloads a station it already has.

Sites are **discovered from each source's own catalogue** rather than being a hard-coded list,
so the pipeline still works as stations come and go. Discovery deliberately spans twelve US
states for geographic diversity.

### Cleaning it up

Real monitoring data is messy in specific, recognisable ways, and the cleaning stage
(`src/data/harmonize.py`) handles each:

- **Different formats** — every source names its parameters differently. All are mapped onto one
  common vocabulary, then reshaped from one-row-per-measurement into one-row-per-timestamp and
  resampled onto a common 15-minute grid.
- **Impossible values** — pH of 47, negative oxygen. Dropped.
- **Exact zeros** — an oxygen reading of precisely `0.0` almost always means "below what the
  instrument can detect", not "there is no oxygen". Treated as missing, not believed.
- **Flatlines** — a probe reporting the identical value for hours is broken, not calm. Those
  runs are masked out. The code comment: *"A probe stuck on one value for hours is broken, not
  calm."*
- **Gaps** — short gaps are filled from the previous reading, but only up to 4 steps.

The **crucial ordering detail**: the record of *which values are real observations* is taken
**before** any gap-filling. So downstream code can always distinguish a genuine measurement from
a filled-in guess — and, as covered earlier, that distinction is fed directly to the neural
network.

Missing data is otherwise **preserved rather than repaired**, because it is exactly what makes
the federated clients genuinely different from one another.

### Refusing to use the easy labels

There's an optional third data source (Kaggle aquaculture datasets), disabled by default. It
contains a rule the project enforces regardless of configuration, and it's worth quoting the
reasoning:

Public water-quality datasets routinely include a "quality" or "potable" column. That column is
almost always produced by applying a threshold rule to the very columns sitting next to it. If
you train a model to predict it and then report 98% accuracy, you have measured nothing except
your model's ability to rediscover arithmetic. It looks like a strong result and is entirely
circular.

So the code **refuses to consume those labels**. The configuration option exists, in the words
of the docstring, *"only so it can be seen to be `false`"* — and setting it to true raises an
error rather than proceeding.

---

## 10. Part eight: the dashboard

*Code: `dashboard/`*

A **Streamlit** web app (Streamlit turns Python scripts into web pages) with **Plotly** charts,
running entirely on a laptop with no internet connection. Six panels:

| Panel | Shows |
|---|---|
| Timeseries | Live channels with safe/warning/critical bands shaded behind them |
| Risk & SHAP | Current risk level plus which readings are driving it |
| Regime | Which cluster the pond resembles, and the anomaly flag |
| Counterfactual | The agent's pond against the untouched shadow pond, side by side |
| Agent | Every decision, the action taken, and the plain-English reason |
| Scorecard | Summary metrics for the run |

**Graceful degradation** is the design principle: if the trained models aren't present, the app
doesn't crash or show empty charts. Each model is loaded independently, and any that's missing
is noted in the sidebar while the app falls back to the plain rule-based thresholds. You get a
less capable dashboard, and you're told exactly what's missing and which command produces it.

Two supporting patterns worth naming:

- **Configuration-driven.** Every number that could reasonably be changed — chemistry
  coefficients, safety thresholds, model sizes, cost weights — lives in a YAML text file under
  `configs/`, not scattered through the code. You can retune the system for a different species
  without touching Python.
- **Dependency injection.** The agent doesn't go and fetch its own perception models; they're
  handed to it when it's created. Sounds like jargon, but the practical effect is that you can
  hand it the full stack, a partial stack, or nothing at all, and the same code runs — which is
  what makes the ablation experiments possible in the first place. And when a perception
  component fails, the error is *recorded* rather than swallowed, because (per the code comment)
  a component that quietly returns nothing looks identical to one that had nothing to say.

---

## 11. What to take away

Most of the individual techniques here are standard. What distinguishes the project is a
consistent refusal to make itself look better than it is:

- The negative clustering result is **published**, with the sweep that establishes it, and the
  flattering metric is explained away rather than quoted.
- The shipped configuration is **not the best row** in its own ablation table, and the config
  file explains why it ships that way anyway.
- The all-knowing oracle is **explicitly denied** the status of an upper bound, with a concrete
  counter-example.
- Calibration values that were measured and then **deliberately not used** are recorded in the
  config with the reasoning attached.
- The privacy parameter is reported as **`UNACCOUNTED`** rather than guessed.
- Dataset labels that would have produced an easy 98% are **refused as circular**.
- Every reported number is **generated from a file**, never typed into a document.
- The scenarios are **tested to actually be as hard as they claim**.

And the architectural point underneath all of it: the machine learning proposes, and a small,
fixed, human-readable set of rules disposes. Over 320 evaluation runs with the guardrails
enabled, zero unsafe actions reached the pond.

---

## Appendix A: glossary

| Term | Meaning |
|---|---|
| **Ablation** | Removing one component to measure what it contributed |
| **Anomaly detection** | Flagging data unlike anything seen in training |
| **Arrhenius correction** | Standard formula for how biological rates speed up with temperature |
| **Baseline** | A simple alternative your method must beat to be interesting |
| **Clustering** | Finding natural groups in data with no labels provided |
| **Counterfactual** | What would have happened if you'd done nothing |
| **DBCV** | A score for whether clusters are genuinely separated (≈0 means no) |
| **Delta target** | Predicting *change* rather than *absolute value* |
| **Determinism** | Same inputs always give the same outputs; needed for reproducibility |
| **Differential privacy** | Adding calibrated noise so individual contributions can't be recovered |
| **Digital twin** | A simulation of a physical system detailed enough to experiment on |
| **Epsilon (ε)** | The privacy budget — lower is more private |
| **Euler integration** | Simulating by taking small time steps: rate × time, repeated |
| **FedAvg** | Federated averaging — combining model updates from many participants |
| **Federated learning** | Training a shared model without centralising anyone's data |
| **Gradient boosting** | Many small decision trees, each fixing the last one's mistakes |
| **Guardrail** | A hard rule enforced outside the model, which the model cannot override |
| **HDBSCAN** | A clustering algorithm that can label points as "none of the above" |
| **IID / non-IID** | Whether all participants' data comes from the same distribution |
| **LightGBM** | A fast, widely-used gradient boosting implementation |
| **LSTM** | A neural network for sequences that carries a running memory |
| **Michaelis–Menten / Monod** | A rate that rises with a resource then plateaus |
| **MPC** | Model Predictive Control — simulate options, pick the best, re-plan often |
| **Multivariate** | Handling several measured quantities at once |
| **Ornstein–Uhlenbeck** | Random noise that drifts but is pulled back toward normal |
| **Overfitting** | Memorising your training data instead of learning a general pattern |
| **p-value** | Chance of seeing a difference this big if there were really no difference |
| **Persistence baseline** | "It'll be the same as it is now" — the dumbest forecast |
| **RMSE** | Root mean squared error — typical size of a prediction mistake |
| **Seed** | A number that makes randomness reproducible |
| **SHAP** | A method for attributing one prediction to its individual inputs |
| **Silhouette** | Another cluster-quality score; can flatter poorly separated clusters |
| **Supervisory control** | A slow planner paired with a fast safety layer |
| **Transfer learning** | Reusing what one model learned as the starting point for another |
| **Wilcoxon signed-rank** | A paired statistical test that assumes no particular distribution |

---

## Appendix B: running it

```bash
pip install -r requirements.txt

python scripts/collect_data.py          # one-time, needs internet
python scripts/calibrate_twin.py        # fit the simulation's noise to real data
python scripts/pretrain_federated.py    # train the forecaster, clustering, risk model
python scripts/pretrain_world_model.py  # train the action-conditioned model
python -m src.eval.run_eval             # regenerate every results table
python scripts/build_report.py          # render RESULTS.md
streamlit run dashboard/app.py          # the live dashboard
pytest                                  # the test suite
```

Everything after the first line runs offline.

> **Note on this copy of the repository:** `data_cache/` is currently empty — the downloaded
> station data isn't included (it's large, and regenerable). The trained models in `models/` and
> the results in `outputs/` *are* included, so the dashboard and the reports work as-is. Anything
> that reads the raw station data — re-running `pretrain_federated.py`, or the forecast-quality
> section of the evaluation — needs `python scripts/collect_data.py` run first.

### Where to look in the code

| Layer | Directory |
|---|---|
| Pond simulation | `src/twin/` |
| Sensor simulation | `src/sensors/` |
| Machine learning models | `src/perception/` |
| Federated learning | `src/federated/` |
| Decision-making & safety | `src/agent/` |
| Evaluation & statistics | `src/eval/` |
| Data pipeline | `src/data/` |
| Dashboard | `dashboard/` |
| All tunable settings | `configs/` |
| Generated results | `outputs/` |
