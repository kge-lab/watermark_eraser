#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

uv sync --extra dev --extra build
uv run pyside6-deploy -c packaging/pysidedeploy.macos.spec -f

app_path="$(find "$project_root/dist" -maxdepth 2 -type d -name '*.app' -print -quit)"
if [[ -z "$app_path" ]]; then
  echo "macOS application bundle was not produced." >&2
  exit 1
fi

codesign --force --deep --sign - "$app_path"
dmg_path="$project_root/dist/GeminiWatermarkEraser-macos-arm64.dmg"
rm -f "$dmg_path"
hdiutil create -volname "GeminiWatermarkEraser" -srcfolder "$app_path" -ov -format UDZO "$dmg_path"
echo "$dmg_path"
