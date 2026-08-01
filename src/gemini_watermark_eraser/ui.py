from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .media import SUPPORTED_EXTENSIONS, next_output_path
from .models import JobStatus
from .processor import VideoProcessor


class BatchWorker(QObject):
    job_progress = Signal(int, str, int, str)
    job_finished = Signal(int, str, str, str)
    batch_finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self.paths = paths
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            processor = VideoProcessor()
        except Exception as exc:
            for row in range(len(self.paths)):
                self.job_finished.emit(row, JobStatus.FAILED.value, "", str(exc))
            self.batch_finished.emit()
            return

        for row, path in enumerate(self.paths):
            if self.cancel_event.is_set():
                self.job_finished.emit(row, JobStatus.CANCELLED.value, "", "작업이 취소되었습니다.")
                continue

            def report(status: JobStatus, fraction: float, message: str, row_index: int = row) -> None:
                self.job_progress.emit(row_index, status.value, int(round(fraction * 100)), message)

            result = processor.process(path, progress=report, cancel_event=self.cancel_event)
            output = "" if result.output_path is None else str(result.output_path)
            self.job_finished.emit(row, result.status.value, output, result.message)
        self.batch_finished.emit()

    @Slot()
    def cancel(self) -> None:
        self.cancel_event.set()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("제미나이 워터마크 지우개")
        self.resize(820, 500)
        self.setAcceptDrops(True)
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._paths: list[Path] = []
        self._outputs: dict[int, Path] = {}

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("MP4 또는 MOV 영상을 여기에 끌어놓으세요")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600; padding: 16px;")
        layout.addWidget(title)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["영상", "상태", "진행률", "결과"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("영상 추가")
        self.remove_button = QPushButton("선택 삭제")
        self.open_button = QPushButton("결과 폴더 열기")
        self.cancel_button = QPushButton("취소")
        self.start_button = QPushButton("워터마크 제거")
        self.start_button.setDefault(True)
        self.start_button.setStyleSheet("font-weight: 600; padding: 8px 18px;")
        self.cancel_button.setEnabled(False)
        self.open_button.setEnabled(False)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.open_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)

        self.status_label = QLabel("원본 파일은 변경하지 않습니다.")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        self.setCentralWidget(root)
        self.add_button.clicked.connect(self._choose_files)
        self.remove_button.clicked.connect(self._remove_selected)
        self.start_button.clicked.connect(self._start_batch)
        self.cancel_button.clicked.connect(self._cancel_batch)
        self.open_button.clicked.connect(self._open_output_folder)
        self._update_controls()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if any(Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS for url in urls):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        self._add_paths([Path(url.toLocalFile()) for url in event.mimeData().urls()])
        event.acceptProposedAction()

    @Slot()
    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "영상 선택",
            "",
            "영상 파일 (*.mp4 *.mov);;모든 파일 (*.*)",
        )
        self._add_paths([Path(path) for path in files])

    def _add_paths(self, paths: list[Path]) -> None:
        known = {path.resolve() for path in self._paths}
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file() or resolved.suffix.lower() not in SUPPORTED_EXTENSIONS or resolved in known:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._paths.append(resolved)
            known.add(resolved)
            self.table.setItem(row, 0, QTableWidgetItem(resolved.name))
            self.table.item(row, 0).setToolTip(str(resolved))
            self.table.setItem(row, 1, QTableWidgetItem(JobStatus.QUEUED.value))
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setTextVisible(True)
            self.table.setCellWidget(row, 2, progress)
            target = next_output_path(resolved)
            self.table.setItem(row, 3, QTableWidgetItem(target.name))
            self.table.item(row, 3).setToolTip(str(target))
        self._update_controls()

    @Slot()
    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            self._paths.pop(row)
        self._outputs.clear()
        self._update_controls()

    @Slot()
    def _start_batch(self) -> None:
        if not self._paths or self._thread is not None:
            return
        self._outputs.clear()
        for row in range(self.table.rowCount()):
            self.table.item(row, 1).setText(JobStatus.QUEUED.value)
            progress = self.table.cellWidget(row, 2)
            if isinstance(progress, QProgressBar):
                progress.setValue(0)

        self._thread = QThread(self)
        self._worker = BatchWorker(list(self._paths))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.job_progress.connect(self._on_job_progress)
        self._worker.job_finished.connect(self._on_job_finished)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.batch_finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._set_running(True)
        self.status_label.setText(f"{len(self._paths)}개 영상을 순서대로 처리합니다.")
        self._thread.start()

    @Slot(int, str, int, str)
    def _on_job_progress(self, row: int, status: str, percent: int, message: str) -> None:
        if row >= self.table.rowCount():
            return
        self.table.item(row, 1).setText(status)
        progress = self.table.cellWidget(row, 2)
        if isinstance(progress, QProgressBar):
            progress.setValue(percent)
            progress.setFormat(f"{percent}%")
        self.status_label.setText(message)

    @Slot(int, str, str, str)
    def _on_job_finished(self, row: int, status: str, output: str, message: str) -> None:
        if row >= self.table.rowCount():
            return
        self.table.item(row, 1).setText(status)
        progress = self.table.cellWidget(row, 2)
        if isinstance(progress, QProgressBar) and status == JobStatus.COMPLETED.value:
            progress.setValue(100)
        if output:
            output_path = Path(output)
            self._outputs[row] = output_path
            self.table.item(row, 3).setText(output_path.name)
            self.table.item(row, 3).setToolTip(output)
        elif message:
            self.table.item(row, 3).setText(message)
            self.table.item(row, 3).setToolTip(message)

    @Slot()
    def _on_batch_finished(self) -> None:
        completed = len(self._outputs)
        self.status_label.setText(f"완료: {completed}개 / 전체: {len(self._paths)}개")
        self.open_button.setEnabled(bool(self._outputs))

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._set_running(False)

    @Slot()
    def _cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("현재 작업을 안전하게 취소하는 중입니다.")

    @Slot()
    def _open_output_folder(self) -> None:
        if not self._outputs:
            return
        folder = next(reversed(self._outputs.values())).parent
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(folder)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _set_running(self, running: bool) -> None:
        self.add_button.setEnabled(not running)
        self.remove_button.setEnabled(not running)
        self.start_button.setEnabled(not running and bool(self._paths))
        self.cancel_button.setEnabled(running)
        self.table.setAcceptDrops(not running)

    def _update_controls(self) -> None:
        running = self._thread is not None
        self.start_button.setEnabled(bool(self._paths) and not running)
        self.remove_button.setEnabled(bool(self._paths) and not running)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._thread is None:
            event.accept()
            return
        answer = QMessageBox.question(self, "작업 취소", "처리 중인 작업을 취소하고 종료할까요?")
        if answer == QMessageBox.StandardButton.Yes:
            if self._worker is not None:
                self._worker.cancel()
            event.ignore()
            self._thread.finished.connect(self.close)
        else:
            event.ignore()


def run_app() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("제미나이 워터마크 지우개")
    window = MainWindow()
    window.show()
    return app.exec()
