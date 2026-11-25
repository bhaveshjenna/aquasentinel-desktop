"""AquaSentinel native desktop client.

A PySide6 + pyqtgraph front end for the same `src/` simulation the Streamlit dashboard
drives. It exists because the browser, not the science, was the bottleneck: one tick of
the twin + LSTM + regime + risk + SHAP stack costs ~5 ms, while building and serialising
the six Plotly figures for one frame cost ~60 ms and 67 KB of JSON before Chromium had
drawn anything.

Nothing in `src/`, `configs/` or `models/` is modified or duplicated - this package is a
view layer and only a view layer.
"""
