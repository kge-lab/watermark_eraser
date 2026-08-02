$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

uv sync --extra dev --extra build
if ($LASTEXITCODE -ne 0) {
    throw "uv sync failed with exit code $LASTEXITCODE."
}

$distDirectory = Join-Path $projectRoot "dist"
if (Test-Path -LiteralPath $distDirectory -PathType Container) {
    foreach ($existingApplicationDirectory in (Get-ChildItem -LiteralPath $distDirectory -Directory -Filter "GeminiWatermarkEraser*.dist")) {
        if (($existingApplicationDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to remove a standalone application directory that is a reparse point: $($existingApplicationDirectory.FullName)"
        }
        Remove-Item -LiteralPath $existingApplicationDirectory.FullName -Recurse -Force
    }
}

$deploySpecPath = Join-Path $projectRoot "packaging\pysidedeploy.windows.spec"
$deploySpecBytes = [IO.File]::ReadAllBytes($deploySpecPath)
$deployExitCode = 1
try {
    uv run pyside6-deploy -c $deploySpecPath -f
    $deployExitCode = $LASTEXITCODE
} finally {
    # pyside6-deploy rewrites the spec with machine-specific absolute paths.
    # Restore the portable, tracked form even when compilation fails.
    [IO.File]::WriteAllBytes($deploySpecPath, $deploySpecBytes)
}
if ($deployExitCode -ne 0) {
    throw "Windows application build failed with exit code $deployExitCode."
}

$applicationDirectory = Get-ChildItem -LiteralPath $distDirectory -Directory -Filter "*.dist" | Select-Object -First 1
if ($null -eq $applicationDirectory) {
    throw "Windows standalone application directory was not produced."
}

$projectNoticeFiles = @(
    (Join-Path $projectRoot "LICENSE"),
    (Join-Path $projectRoot "THIRD_PARTY_NOTICES.md")
)
foreach ($noticeFile in $projectNoticeFiles) {
    if (-not (Test-Path -LiteralPath $noticeFile -PathType Leaf)) {
        throw "Required project notice file is missing: $noticeFile"
    }
}

$ffmpegComplianceDirectory = Join-Path $projectRoot "third_party\ffmpeg\windows"
$ffmpegComplianceFiles = @(
    (Join-Path $ffmpegComplianceDirectory "LICENSE"),
    (Join-Path $ffmpegComplianceDirectory "README.txt"),
    (Join-Path $ffmpegComplianceDirectory "SOURCE.md")
)
foreach ($complianceFile in $ffmpegComplianceFiles) {
    if (-not (Test-Path -LiteralPath $complianceFile -PathType Leaf)) {
        throw "Required FFmpeg compliance file is missing: $complianceFile"
    }
}

$sitePackagesOutput = & uv run python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not locate the installed Python site-packages directory."
}
$sitePackages = ($sitePackagesOutput | Select-Object -Last 1).Trim()
if (-not (Test-Path -LiteralPath $sitePackages -PathType Container)) {
    throw "Installed Python site-packages directory does not exist: $sitePackages"
}

if (($applicationDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Refusing to modify a standalone application directory that is a reparse point: $($applicationDirectory.FullName)"
}
$applicationPath = [IO.Path]::GetFullPath($applicationDirectory.FullName).TrimEnd([IO.Path]::DirectorySeparatorChar)
$licensesDirectory = [IO.Path]::GetFullPath((Join-Path $applicationPath "licenses"))
$expectedLicensesDirectory = $applicationPath + [IO.Path]::DirectorySeparatorChar + "licenses"
if (-not $licensesDirectory.Equals($expectedLicensesDirectory, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to replace a licenses directory outside the standalone application directory."
}
if (Test-Path -LiteralPath $licensesDirectory) {
    $existingLicensesDirectory = Get-Item -LiteralPath $licensesDirectory -Force
    if (($existingLicensesDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to replace a licenses directory that is a reparse point: $licensesDirectory"
    }
    Remove-Item -LiteralPath $licensesDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $licensesDirectory | Out-Null

Copy-Item -LiteralPath $projectNoticeFiles[0] -Destination (Join-Path $licensesDirectory "LICENSE")
Copy-Item -LiteralPath $projectNoticeFiles[1] -Destination (Join-Path $licensesDirectory "THIRD_PARTY_NOTICES.md")

$bundledFfmpegExecutables = @(
    Get-ChildItem -LiteralPath $applicationPath -Recurse -File -Filter "ffmpeg-win-x86_64-v7.1.exe"
)
if ($bundledFfmpegExecutables.Count -ne 1) {
    throw "Expected exactly one bundled FFmpeg 7.1 executable, found $($bundledFfmpegExecutables.Count)."
}
$bundledFfmpeg = $bundledFfmpegExecutables[0]
$expectedFfmpegSha256 = "2CE797A0F88D7F067180338FB227F7B1928EA727BD9A4D7A1D022F7C52AF71A3"
$actualFfmpegSha256 = (Get-FileHash -LiteralPath $bundledFfmpeg.FullName -Algorithm SHA256).Hash
if ($actualFfmpegSha256 -ne $expectedFfmpegSha256) {
    throw "Bundled FFmpeg binary does not match the audited GPLv3 build. Expected $expectedFfmpegSha256, got $actualFfmpegSha256."
}

$ffmpegLicensesDirectory = Join-Path $licensesDirectory "ffmpeg\windows"
New-Item -ItemType Directory -Path $ffmpegLicensesDirectory -Force | Out-Null
foreach ($complianceFile in $ffmpegComplianceFiles) {
    Copy-Item -LiteralPath $complianceFile -Destination (Join-Path $ffmpegLicensesDirectory (Split-Path -Leaf $complianceFile))
}

$dependencyDistributions = @(
    @{ Label = "PySide6-Essentials"; Pattern = '^pyside6[-_]essentials-.+\.dist-info$' },
    @{ Label = "opencv-python-headless"; Pattern = '^opencv[-_]python[-_]headless-.+\.dist-info$' },
    @{ Label = "NumPy"; Pattern = '^numpy-.+\.dist-info$' },
    @{ Label = "imageio-ffmpeg/FFmpeg"; Pattern = '^imageio[-_]ffmpeg-.+\.dist-info$' }
)
$installedMetadataDirectories = @(Get-ChildItem -LiteralPath $sitePackages -Directory -Filter "*.dist-info")

foreach ($distribution in $dependencyDistributions) {
    $matchingMetadataDirectories = @(
        $installedMetadataDirectories |
            Where-Object { $_.Name -match $distribution.Pattern } |
            Sort-Object -Property Name
    )
    if ($matchingMetadataDirectories.Count -eq 0) {
        Write-Warning "No installed dist-info directory matched $($distribution.Label); continuing without it."
        continue
    }

    foreach ($metadataDirectory in $matchingMetadataDirectories) {
        $metadataPrefix = $metadataDirectory.FullName.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        $copiedLicenseCount = 0

        foreach ($candidateFile in (Get-ChildItem -LiteralPath $metadataDirectory.FullName -Recurse -File | Sort-Object -Property FullName)) {
            if (-not $candidateFile.FullName.StartsWith($metadataPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to copy a file outside its dist-info directory: $($candidateFile.FullName)"
            }
            $relativePath = $candidateFile.FullName.Substring($metadataPrefix.Length)
            $isLicenseFile =
                $relativePath -match '(^|[\\/])licenses?([\\/]|$)' -or
                $candidateFile.Name -match '(license|licence|copying|notice|authors?|copyright|patents?)'
            if (-not $isLicenseFile) {
                continue
            }

            $destinationFile = Join-Path (Join-Path $licensesDirectory $metadataDirectory.Name) $relativePath
            $destinationParent = Split-Path -Parent $destinationFile
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
            Copy-Item -LiteralPath $candidateFile.FullName -Destination $destinationFile
            $copiedLicenseCount++
        }

        if ($copiedLicenseCount -eq 0) {
            Write-Warning "No license files were found in $($metadataDirectory.Name); continuing without them."
        }
    }
}

$archivePath = Join-Path $distDirectory "GeminiWatermarkEraser-windows-x64.zip"
$resolvedDist = (Resolve-Path -LiteralPath $distDirectory).Path.TrimEnd([IO.Path]::DirectorySeparatorChar)
$resolvedArchiveParent = (Resolve-Path -LiteralPath (Split-Path -Parent $archivePath)).Path.TrimEnd([IO.Path]::DirectorySeparatorChar)
if ($resolvedArchiveParent -ne $resolvedDist) {
    throw "Refusing to replace an archive outside the project dist directory."
}
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Compress-Archive -Path (Join-Path $applicationDirectory.FullName "*") -DestinationPath $archivePath -CompressionLevel Optimal
Write-Output $archivePath
