#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

mac_arch="${MACOS_TARGET_ARCH:-$(uname -m)}"
case "$mac_arch" in
  arm64|x86_64) ;;
  *)
    echo "Unsupported macOS architecture: $mac_arch (expected arm64 or x86_64)." >&2
    exit 1
    ;;
esac

dist_directory="$project_root/dist"
mkdir -p "$dist_directory"

# pyside6-deploy rewrites its spec with local absolute paths. Restore the
# tracked spec on every exit so a local build never dirties the checkout.
spec_path="$project_root/packaging/pysidedeploy.macos.spec"
spec_backup="$(mktemp "${TMPDIR:-/tmp}/gemini-watermark-spec.XXXXXX")"
cp "$spec_path" "$spec_backup"
metadata_manifest=""
license_manifest=""
cleanup() {
  cp "$spec_backup" "$spec_path" 2>/dev/null || true
  rm -f "$spec_backup" "$metadata_manifest" "$license_manifest"
}
trap cleanup EXIT

# Avoid selecting a stale bundle when a developer rebuilds in the same tree.
for existing_app in "$dist_directory"/GeminiWatermarkEraser*.app; do
  [[ -d "$existing_app" ]] || continue
  rm -rf -- "$existing_app"
done

uv sync --extra dev --extra build
uv run pyside6-deploy -c "$spec_path" -f

app_path="$(find "$dist_directory" -type d -name 'GeminiWatermarkEraser*.app' -print -quit)"
if [[ -z "$app_path" ]]; then
  echo "macOS application bundle was not produced." >&2
  exit 1
fi

project_notice_files=(
  "$project_root/LICENSE"
  "$project_root/THIRD_PARTY_NOTICES.md"
)
for notice_file in "${project_notice_files[@]}"; do
  if [[ ! -f "$notice_file" ]]; then
    echo "Required project notice file is missing: $notice_file" >&2
    exit 1
  fi
done

site_packages="$(uv run python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if [[ ! -d "$site_packages" ]]; then
  echo "Installed Python site-packages directory does not exist: $site_packages" >&2
  exit 1
fi

resolved_app_path="$(cd "$app_path" && pwd -P)"
contents_directory="$app_path/Contents"
if [[ ! -d "$contents_directory" ]]; then
  echo "macOS application bundle has no Contents directory: $contents_directory" >&2
  exit 1
fi
resolved_contents_directory="$(cd "$contents_directory" && pwd -P)"
if [[ "$resolved_contents_directory" != "$resolved_app_path/Contents" ]]; then
  echo "Refusing to create a licenses directory through a symbolic link outside the application bundle." >&2
  exit 1
fi

resources_candidate="$contents_directory/Resources"
if [[ -L "$resources_candidate" ]]; then
  echo "Refusing to create a licenses directory through a symbolic link: $resources_candidate" >&2
  exit 1
fi
mkdir -p "$resources_candidate"
resources_directory="$(cd "$resources_candidate" && pwd -P)"
if [[ "$resources_directory" != "$resolved_app_path/Contents/Resources" ]]; then
  echo "Refusing to replace a licenses directory through a symbolic link outside the application bundle." >&2
  exit 1
fi

licenses_directory="$resources_directory/licenses"
if [[ -L "$licenses_directory" ]]; then
  echo "Refusing to replace a licenses directory that is a symbolic link: $licenses_directory" >&2
  exit 1
fi
if [[ -e "$licenses_directory" ]]; then
  rm -rf -- "$licenses_directory"
fi
mkdir "$licenses_directory"

cp "$project_root/LICENSE" "$licenses_directory/LICENSE"
cp "$project_root/THIRD_PARTY_NOTICES.md" "$licenses_directory/THIRD_PARTY_NOTICES.md"

dependency_labels=(
  "PySide6-Essentials"
  "opencv-python-headless"
  "NumPy"
  "imageio-ffmpeg/FFmpeg"
)
dependency_patterns=(
  'pyside6[-_]essentials-*.dist-info'
  'opencv[-_]python[-_]headless-*.dist-info'
  'numpy-*.dist-info'
  'imageio[-_]ffmpeg-*.dist-info'
)
matched_metadata_counts=(0 0 0 0)

# BSD find on macOS does not implement GNU -mindepth/-maxdepth. Build the
# manifest with shell globs, then use only portable find features below.
metadata_manifest="$(mktemp "${TMPDIR:-/tmp}/gemini-watermark-metadata.XXXXXX")"
license_manifest="$(mktemp "${TMPDIR:-/tmp}/gemini-watermark-license-files.XXXXXX")"
for metadata_directory in "$site_packages"/*.dist-info; do
  [[ -d "$metadata_directory" ]] || continue
  printf '%s\0' "$metadata_directory" >> "$metadata_manifest"
done

shopt -s nocasematch
while IFS= read -r -d '' metadata_directory; do
  metadata_name="${metadata_directory##*/}"
  matching_dependency_index=-1
  for ((dependency_index = 0; dependency_index < ${#dependency_patterns[@]}; dependency_index++)); do
    case "$metadata_name" in
      ${dependency_patterns[$dependency_index]})
        matching_dependency_index=$dependency_index
        break
        ;;
    esac
  done
  if ((matching_dependency_index < 0)); then
    continue
  fi

  matched_metadata_counts[$matching_dependency_index]=$((matched_metadata_counts[$matching_dependency_index] + 1))
  copied_license_count=0
  : > "$license_manifest"
  if ! find "$metadata_directory" -type f -print0 >> "$license_manifest"; then
    echo "Could not inspect installed license metadata: $metadata_directory" >&2
    exit 1
  fi

  while IFS= read -r -d '' candidate_file; do
    relative_path="${candidate_file#"$metadata_directory"/}"
    candidate_name="${relative_path##*/}"
    is_license_file=0
    case "$relative_path" in
      license/*|licenses/*|*/license/*|*/licenses/*) is_license_file=1 ;;
    esac
    case "$candidate_name" in
      *license*|*licence*|*copying*|*notice*|*author*|*copyright*|*patent*) is_license_file=1 ;;
    esac
    if ((is_license_file == 0)); then
      continue
    fi

    destination_file="$licenses_directory/$metadata_name/$relative_path"
    mkdir -p "${destination_file%/*}"
    cp "$candidate_file" "$destination_file"
    copied_license_count=$((copied_license_count + 1))
  done < "$license_manifest"

  if ((copied_license_count == 0)); then
    echo "Warning: no license files were found in $metadata_name; continuing without them." >&2
  fi
done < "$metadata_manifest"
shopt -u nocasematch

for ((dependency_index = 0; dependency_index < ${#dependency_labels[@]}; dependency_index++)); do
  if ((matched_metadata_counts[$dependency_index] == 0)); then
    echo "Warning: no installed dist-info directory matched ${dependency_labels[$dependency_index]}; continuing without it." >&2
  fi
done

app_binary="$(find "$app_path/Contents/MacOS" -type f -print -quit)"
if [[ -z "$app_binary" ]]; then
  echo "Could not locate the macOS application executable." >&2
  exit 1
fi
binary_arches="$(lipo -archs "$app_binary")"
case " $binary_arches " in
  *" $mac_arch "*) ;;
  *)
    echo "Application architecture mismatch: expected $mac_arch, got $binary_arches." >&2
    exit 1
    ;;
esac

codesign --force --deep --sign - "$app_path"
dmg_path="$project_root/dist/GeminiWatermarkEraser-macos-$mac_arch.dmg"
rm -f -- "$dmg_path"
hdiutil create -volname "GeminiWatermarkEraser-$mac_arch" -srcfolder "$app_path" -ov -format UDZO "$dmg_path"
echo "$dmg_path"
