from __future__ import annotations

from pathlib import Path
import threading

from gemini_watermark_eraser.models import JobStatus
from gemini_watermark_eraser.processor import VideoProcessor


def test_pre_cancelled_job_preserves_source_and_writes_nothing(tmp_path: Path) -> None:
    source = Path("sample/sample_1.mp4")
    output = tmp_path / "cancelled.mp4"
    cancel = threading.Event()
    cancel.set()
    result = VideoProcessor().process(source, output_path=output, cancel_event=cancel)
    assert result.status == JobStatus.CANCELLED
    assert not output.exists()
    assert source.exists()


def test_processor_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep")
    result = VideoProcessor().process(Path("sample/sample_1.mp4"), output_path=output)
    assert result.status == JobStatus.FAILED
    assert "덮어쓰지" in result.message
    assert output.read_bytes() == b"keep"
