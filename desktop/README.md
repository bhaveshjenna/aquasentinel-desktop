# AquaSentinel — native desktop client

A PySide6 + pyqtgraph front end for the same `src/` simulation the Streamlit dashboard drives.
No browser, no local web server, no websocket — one Python process drawing through Qt.

```
pip install -r requirements-desktop.txt
python -m desktop          # or: run_desktop.bat
```

## Why this exists

The browser, not the science, was the bottleneck. Measured on this machine:

| | cost per playback frame |
|---|---|
| `AgentRun.tick()` — twin + LSTM + regimes + risk + SHAP | **5.05 ms** (~198 ticks/s headless) |
| Building + serialising the six Plotly figures | **60.4 ms**, 67 KB of JSON |

At the Streamlit app's `MAX_FPS = 4` that was ~24% of a core spent producing JSON and
268 KB/s pushed down a websocket, before Chromium re-rendered nine charts. The simulation was
about 8% of the total cost.

Wrapping the Streamlit app in Electron or pywebview would not have helped: same Chromium, same
Plotly, same JSON — only the tab chrome removed. The cost had to be deleted, not rehoused.

Measured here, offscreen (software raster; the real windowed path is faster):

| view | render |
|---|---|
| Live (6 charts + regime + risk + scorecard) | 9.8 ms/frame |
| Counterfactual | 9.2 ms/frame |
| Agent | 1.7 ms/frame |
| Scorecard | 1.1 ms/frame |

Playback holds within 2% of the requested tick rate at both 8 and 32 ticks/s.

## What makes it cheap

- **Nothing is rebuilt.** Every curve, band, bar and axis is constructed once. A frame is
  `setData(x, y)` on items that already exist, so there is no figure construction, no JSON,
  no DOM, and no component re-keying.
- **Incremental buffers** (`engine.Columns`). `RunResult.frame()` rebuilds a pandas DataFrame
  from every state in the run on each call, so in the browser version the cost of drawing a
  frame grew with the length of the run. Here each tick appends to preallocated numpy arrays
  and the panels read views. Drawing frame 2000 costs what drawing frame 10 cost.
- **A real clock.** Playback is a `QTimer`; advancing the pond and repainting are separate
  steps, so a slow paint slows the paint rather than the simulation. Streamlit re-executed
  `app.py` top to bottom every frame.
- **Only the visible view is drawn.** A hidden panel costs nothing.
- **`metrics()` is throttled** to 0.4 s, because it walks every state in the run.
- **A frame with no new tick is skipped**, so at low speeds the app repaints only when the
  pond actually changed.

## Layout

```
desktop/
  __main__.py       entry point (python -m desktop)
  engine.py         RunEngine: AgentRun ownership, playback clock, display buffers
  main_window.py    sidebar, header readouts, view switching
  theme.py          colour vocabulary + Qt stylesheet
  plotting.py       chart primitives (persistent curves, risk bands, y-range)
  widgets.py        cards, metric tiles, risk gauge, legend strip
  panels/           timeseries · regime · risk · counterfactual · agent · scorecard
```

`src/`, `configs/` and `models/` are untouched and unduplicated — this package is a view layer
and only a view layer. The Streamlit app in `dashboard/` still runs, unchanged, from the same
tree.

## Deliberate differences from the Streamlit dashboard

- **One legend per chart grid, not per chart.** pyqtgraph's legend sits inside the view box,
  where on six stacked charts it covered the top-right of every one of them — exactly where a
  DO trace peaks. The colour vocabulary is shared across the grid, so one strip above it says
  the same thing without occluding data.
- **Live charts do not respond to the mouse.** They follow a moving 24 h window, so a user pan
  would be overwritten by the next frame. The counterfactual charts show settled history and
  do allow zoom and pan.
- **Keyboard control**: `Space` play/pause, `S` step one decision interval, `R` reset.
- **The risk gauge is a QPainter arc** rather than a Plotly `go.Indicator`.

Everything else — the risk bands, the colour vocabulary, the metric definitions, the degraded
paths when `models/` is incomplete, the "the pond is a simulation" provenance notes — is the
same, and reads from the same `configs/`.

## Tests

```
python -m pytest tests/test_desktop.py -q
```

Offscreen, so they run with no display. They cover the buffers staying in step with the run
they mirror, capacity growth, reset, the playback pacing (including the rounding bug where
`max(1, round(elapsed * speed))` per frame makes every speed below the frame rate play back
*at* the frame rate), the no-models degraded path, and that every view paints.

## Packaging

Shipping a standalone `.exe` is a separate problem: PyInstaller with torch + shap + lightgbm
lands around 1.5 GB. For a demo, `run_desktop.bat` is the saner answer.
