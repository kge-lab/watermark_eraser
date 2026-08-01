from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gemini_watermark_eraser.ui import MainWindow  # noqa: E402


def test_main_window_has_only_watermark_workflow_controls() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.windowTitle() == "제미나이 워터마크 지우개"
        assert window.start_button.text() == "워터마크 제거"
        assert not window.start_button.isEnabled()
        assert window.table.columnCount() == 4
    finally:
        window.close()
        app.processEvents()
