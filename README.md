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
- Intel macOS 13 이상 (x86_64)
- Apple Silicon macOS 13 이상 (arm64)
- MP4/MOV, 최대 10분
- 가로·세로 720p/1080p 제미나이 생성 영상

제미나이 로고를 확실히 찾지 못하면 영상은 변경하지 않고 해당 파일을 실패 처리합니다. 범용 워터마크 제거, 수동 마스크, 자르기, 색보정 등의 편집 기능은 포함하지 않습니다.

워터마크가 덮은 원본 픽셀은 영상 파일에 남아 있지 않으므로, 앱은 앞뒤 프레임과 주변의 깨끗한 질감을 이용해 가려진 영역을 추정합니다. 움직임이 거의 없거나 고유한 물체가 완전히 가려진 장면에서는 픽셀 단위 원상복원을 보장할 수 없습니다.

지원되는 제미나이 프로필에서는 보정된 알파 맵으로 로고 아래의 미세 질감을 역산합니다. 전체 역산 결과를 그대로 쓰면 어두운 별 모양 윤곽이 생길 수 있으므로, 앱은 로고 경계를 제외한 내부 디테일만 제한적으로 결합합니다. 밝기 편향·클리핑·선명도 검사를 통과하지 못한 프레임은 기존 시간축 복원 결과를 그대로 사용합니다.

## 테스트

```powershell
uv run pytest
```

## 배포본 만들기

Windows portable ZIP은 다음 명령으로 생성합니다.

```powershell
./packaging/build_windows.ps1
```

macOS DMG는 실행 중인 Mac의 아키텍처를 자동 감지해 생성합니다. Apple Silicon에서는 `GeminiWatermarkEraser-macos-arm64.dmg`, Intel에서는 `GeminiWatermarkEraser-macos-x86_64.dmg`가 만들어집니다. GitHub Actions의 **Build desktop apps** 워크플로도 두 아키텍처를 각각 빌드합니다. Mac에서 직접 만들 때는 다음 명령을 사용합니다.

```bash
bash packaging/build_macos.sh
```

GitHub Actions에서 `release-*` 태그를 푸시하면 Windows ZIP과 두 macOS DMG를 모두 빌드한 뒤 같은 GitHub Release에 자동으로 첨부합니다. `workflow_dispatch`로 빌드만 실행해 테스트할 수도 있습니다.

## 공개 배포본 테스트

GitHub의 [Releases](https://github.com/kge-lab/watermark_eraser/releases)에서 운영체제에 맞는 파일을 내려받습니다.

- Windows: `GeminiWatermarkEraser-windows-x64.zip`을 새 폴더에 압축 해제한 뒤 `GeminiWatermarkEraser.exe`를 실행합니다. SmartScreen이 표시되면 릴리스의 SHA-256을 확인한 뒤 **추가 정보 → 실행**을 선택합니다.
- Apple Silicon Mac: `GeminiWatermarkEraser-macos-arm64.dmg`
- Intel Mac: `GeminiWatermarkEraser-macos-x86_64.dmg`

Windows와 macOS 모두 원본 영상의 복사본을 추가한 뒤 **워터마크 제거**를 누르면 같은 폴더에 `_clean.mp4` 결과가 생성됩니다.

## macOS 개인용 앱 실행

배포되는 macOS 앱은 Developer ID 서명과 Apple 공증을 하지 않은 개인용 빌드입니다. DMG를 열고 앱을 응용 프로그램 폴더로 복사한 다음, 최초 실행 시 Finder에서 앱을 Control-클릭한 뒤 **열기**를 선택하거나 시스템 설정의 **개인정보 보호 및 보안**에서 실행을 허용해야 합니다.

## 라이선스

프로젝트 코드는 MIT 라이선스입니다. 패키지에는 Qt for Python, OpenCV, FFmpeg 등 별도 라이선스의 구성요소가 포함되며 자세한 내용은 `THIRD_PARTY_NOTICES.md`를 확인하세요.
