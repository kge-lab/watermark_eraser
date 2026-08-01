from __future__ import annotations

from pathlib import Path

import pytest

from gemini_watermark_eraser.media import next_output_path, probe_media
from gemini_watermark_eraser.models import ProcessingError


def test_probe_sample_media() -> None:
    media = probe_media(Path("sample/sample_1.mp4"))
    assert (media.width, media.height) == (1280, 720)
    assert media.fps == pytest.approx(24.0)
    assert media.duration == pytest.approx(10.0, abs=0.1)


def test_output_names_never_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    assert next_output_path(source).name == "clip_clean.mp4"
    (tmp_path / "clip_clean.mp4").touch()
    assert next_output_path(source).name == "clip_clean_2.mp4"


def test_probe_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "clip.webm"
    source.touch()
    with pytest.raises(ProcessingError, match="MP4 또는 MOV"):
        probe_media(source)
