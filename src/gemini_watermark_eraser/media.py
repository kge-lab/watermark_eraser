from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

from .models import MediaInfo, ProcessingError

SUPPORTED_EXTENSIONS = {".mp4", ".mov"}
COPYABLE_MP4_AUDIO = {"aac", "alac", "mp3"}


def find_ffmpeg() -> Path:
    override = os.environ.get("WATERMARK_ERASER_FFMPEG")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    executable_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    roots = [Path(sys.executable).resolve().parent, Path(__file__).resolve().parent]
    for root in roots:
        for candidate in (root / executable_name, root / "resources" / executable_name):
            if candidate.is_file():
                return candidate

    try:
        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if candidate.is_file():
            return candidate.resolve()
    except Exception:
        pass

    on_path = shutil.which("ffmpeg")
    if on_path:
        return Path(on_path).resolve()
    raise ProcessingError("FFmpeg 실행 파일을 찾을 수 없습니다. 앱을 다시 설치해 주세요.")


def probe_media(path: Path) -> MediaInfo:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ProcessingError("영상 파일을 찾을 수 없습니다.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ProcessingError("MP4 또는 MOV 영상만 지원합니다.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ProcessingError("영상을 열 수 없습니다. 파일 또는 코덱을 확인해 주세요.")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()

    if width < 64 or height < 64 or not (0.1 < fps < 240) or frame_count < 1:
        raise ProcessingError("영상의 해상도, 프레임률 또는 길이를 읽을 수 없습니다.")
    duration = frame_count / fps
    if duration > 600.5:
        raise ProcessingError("10분 이하의 영상만 지원합니다.")
    return MediaInfo(path, width, height, fps, frame_count, duration)


def next_output_path(input_path: Path) -> Path:
    base = input_path.with_name(f"{input_path.stem}_clean.mp4")
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = input_path.with_name(f"{input_path.stem}_clean_{index}.mp4")
        if not candidate.exists():
            return candidate
        index += 1


def partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.partial.mp4")


def audio_codec(path: Path, ffmpeg_path: Path | None = None) -> str | None:
    ffmpeg_path = ffmpeg_path or find_ffmpeg()
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-i",
        str(path),
        "-t",
        "0",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    matches = re.findall(r"Audio:\s*([^,\s]+)", completed.stderr, flags=re.IGNORECASE)
    return matches[0].lower() if matches else None
