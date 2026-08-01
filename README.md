# 제미나이 워터마크 지우개

Gemini에서 생성한 MP4/MOV 영상의 오른쪽 하단에 표시되는 제미나이 로고만 자동으로 지우는 오프라인 데스크톱 앱입니다. 영상은 외부로 전송되지 않으며 원본 파일을 덮어쓰지 않습니다.

## 개발 환경에서 실행

Python 3.11 또는 3.12와 `uv`가 필요합니다.

```powershell
uv sync --extra dev --extra build
uv run python -m gemini_watermark_eraser
```

앱 창에 MP4 또는 MOV 파일을 여러 개 끌어놓고 **워터마크 제거**를 누릅니다. 결과는 원본과 같은 폴더의 `원본명_clean.mp4`에 저장됩니다.

## 지원 범위

- Windows 10/11 x64
- Apple Silicon macOS 13 이상
- MP4/MOV, 최대 10분
- 가로·세로 720p/1080p 제미나이 생성 영상

제미나이 로고를 확실히 찾지 못하면 영상은 변경하지 않고 해당 파일을 실패 처리합니다. 범용 워터마크 제거, 수동 마스크, 자르기, 색보정 등의 편집 기능은 포함하지 않습니다.

## 테스트

```powershell
uv run pytest
```

## 배포본 만들기

Windows portable ZIP은 다음 명령으로 생성합니다.

```powershell
./packaging/build_windows.ps1
```

Apple Silicon macOS 앱은 GitHub Actions의 **Build desktop apps** 워크플로를 수동 실행하면 `GeminiWatermarkEraser-macos-arm64.dmg` 아티팩트로 생성됩니다. Mac에서 직접 만들 때는 다음 명령을 사용합니다.

```bash
bash packaging/build_macos.sh
```

## macOS 개인용 앱 실행

배포되는 macOS 앱은 Developer ID 서명과 Apple 공증을 하지 않은 개인용 빌드입니다. 최초 실행 시 Finder에서 앱을 Control-클릭한 뒤 **열기**를 선택하거나, 시스템 설정의 **개인정보 보호 및 보안**에서 실행을 허용해야 합니다.

## 라이선스

프로젝트 코드는 MIT 라이선스입니다. 패키지에는 Qt for Python, OpenCV, FFmpeg 등 별도 라이선스의 구성요소가 포함되며 자세한 내용은 `THIRD_PARTY_NOTICES.md`를 확인하세요.
