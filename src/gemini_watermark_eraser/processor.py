from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path
import os
import shutil
import subprocess
import sys
import threading

import cv2

from .detector import detect_logo
from .media import COPYABLE_MP4_AUDIO, audio_codec, find_ffmpeg, next_output_path, partial_output_path, probe_media
from .models import (
    JobResult,
    JobStatus,
    ProcessingCancelled,
    ProcessingError,
)
from .restorer import FramePatch, PreparedFrame, TemporalLogoRestorer

ProgressCallback = Callable[[JobStatus, float, str], None]


def _noop_progress(status: JobStatus, fraction: float, message: str) -> None:
    del status, fraction, message


class VideoProcessor:
    def __init__(self, ffmpeg_path: Path | None = None) -> None:
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg()

    def _encoder_command(
        self,
        *,
        input_path: Path,
        partial_path: Path,
        width: int,
        height: int,
        fps: float,
    ) -> list[str]:
        codec = audio_codec(input_path, self.ffmpeg_path)
        audio_args = ["-c:a", "copy"] if codec in COPYABLE_MP4_AUDIO else ["-c:a", "aac", "-b:a", "192k"]
        return [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map_metadata",
            "1",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            *audio_args,
            "-movflags",
            "+faststart",
            "-shortest",
            str(partial_path),
        ]

    def process(
        self,
        input_path: Path,
        *,
        output_path: Path | None = None,
        progress: ProgressCallback = _noop_progress,
        cancel_event: threading.Event | None = None,
    ) -> JobResult:
        input_path = input_path.expanduser().resolve()
        explicit_output = output_path is not None
        output_path = output_path or next_output_path(input_path)
        output_path = output_path.expanduser().resolve()
        partial_path = partial_output_path(output_path)
        cancel_event = cancel_event or threading.Event()

        def cancelled() -> bool:
            return cancel_event.is_set()

        process: subprocess.Popen[bytes] | None = None
        capture: cv2.VideoCapture | None = None
        try:
            if output_path == input_path:
                raise ProcessingError("원본 영상과 같은 경로에는 저장할 수 없습니다.")
            if explicit_output and output_path.exists():
                raise ProcessingError("출력 파일이 이미 존재합니다. 원본과 기존 결과는 덮어쓰지 않습니다.")
            media = probe_media(input_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(output_path.parent).free
            required_bytes = max(512 * 1024 * 1024, int(input_path.stat().st_size * 2.5))
            if free_bytes < required_bytes:
                raise ProcessingError("출력 폴더의 디스크 공간이 부족합니다.")
            progress(JobStatus.DETECTING, 0.0, "제미나이 로고를 찾는 중")
            detection = detect_logo(media, cancelled=cancelled)
            if cancelled():
                raise ProcessingCancelled("작업이 취소되었습니다.")

            restorer = TemporalLogoRestorer(media.width, media.height, detection)
            capture = cv2.VideoCapture(str(input_path))
            if not capture.isOpened():
                raise ProcessingError("영상 디코더를 시작할 수 없습니다.")

            partial_path.unlink(missing_ok=True)
            process = subprocess.Popen(
                self._encoder_command(
                    input_path=input_path,
                    partial_path=partial_path,
                    width=media.width,
                    height=media.height,
                    fps=media.fps,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            if process.stdin is None:
                raise ProcessingError("영상 인코더 입력을 열 수 없습니다.")

            future: deque[PreparedFrame] = deque()
            past: deque[FramePatch] = deque(maxlen=restorer.temporal_radius)
            next_index = 0

            def read_next() -> bool:
                nonlocal next_index
                assert capture is not None
                ok, frame = capture.read()
                if not ok or frame is None:
                    return False
                future.append(restorer.prepare_frame(next_index, frame))
                next_index += 1
                return True

            for _ in range(restorer.temporal_radius + 1):
                if not read_next():
                    break

            processed_count = 0
            while future:
                if cancelled():
                    raise ProcessingCancelled("작업이 취소되었습니다.")
                prepared = future.popleft()
                read_next()
                cleaned = restorer.restore(
                    prepared,
                    past=past,
                    future=future,
                )
                try:
                    process.stdin.write(cleaned.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    stderr = b"" if process.stderr is None else process.stderr.read()
                    detail = stderr.decode("utf-8", errors="replace").strip()
                    raise ProcessingError(f"영상 인코더가 중단되었습니다. {detail}") from exc
                past.append(restorer.make_patch(prepared))
                processed_count += 1
                progress(
                    JobStatus.RESTORING,
                    min(0.99, processed_count / max(1, media.frame_count)),
                    f"{processed_count:,} / {media.frame_count:,} 프레임",
                )

            progress(JobStatus.ENCODING, 0.99, "영상과 오디오를 마무리하는 중")
            process.stdin.close()
            return_code = process.wait()
            stderr = b"" if process.stderr is None else process.stderr.read()
            process.stdin = None
            if return_code != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise ProcessingError(f"출력 영상을 저장하지 못했습니다. {detail}")
            if not partial_path.is_file() or partial_path.stat().st_size == 0:
                raise ProcessingError("출력 영상이 생성되지 않았습니다.")
            os.replace(partial_path, output_path)
            progress(JobStatus.COMPLETED, 1.0, "완료")
            return JobResult(input_path, output_path, JobStatus.COMPLETED, "완료")
        except ProcessingCancelled as exc:
            return JobResult(input_path, None, JobStatus.CANCELLED, str(exc))
        except ProcessingError as exc:
            return JobResult(input_path, None, JobStatus.FAILED, str(exc))
        except Exception as exc:  # Keep one damaged file from stopping a batch.
            return JobResult(input_path, None, JobStatus.FAILED, f"예상하지 못한 오류: {exc}")
        finally:
            if capture is not None:
                capture.release()
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            partial_path.unlink(missing_ok=True)
