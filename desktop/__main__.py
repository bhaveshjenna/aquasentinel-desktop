"""Entry point:  python -m desktop

Fully offline, exactly like the Streamlit dashboard: the twin, the pretrained weights and the
cached real-data manifest all live on disk and nothing here touches the network. If ``models/``
is empty the app still runs, with the perception layer degraded to the rule-based risk function
and no forecast overlay, and says so in the sidebar rather than failing.
"""

from __future__ import annotations

import os
import sys
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.theme import STYLESHEET, configure_pyqtgraph  # noqa: E402


def main() -> int:
    # window is built.
    configure_pyqtgraph()

    app = QApplication(sys.argv)
    app.setApplicationName("Safe Predictive Control and Federated Forecasting for Aquaculture Water Quality Simulation")
    app.setOrganizationName("AquaSentinel")
    app.setStyleSheet(STYLESHEET)

    from desktop.main_window import MainWindow

    window = MainWindow()
    screen = QGuiApplication.primaryScreen()
    if screen:
        avail = screen.availableGeometry()
        window.setGeometry(avail.x() + 10, avail.y() + 10, max(1140, avail.width() - 20), max(590, avail.height() - 20))
    window.showNormal()
    window.showMaximized()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
