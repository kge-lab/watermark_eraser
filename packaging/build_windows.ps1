$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

uv sync --extra dev --extra build
uv run pyside6-deploy -c packaging/pysidedeploy.windows.spec -f

$distDirectory = Join-Path $projectRoot "dist"
$applicationDirectory = Get-ChildItem -LiteralPath $distDirectory -Directory -Filter "*.dist" | Select-Object -First 1
if ($null -eq $applicationDirectory) {
    throw "Windows standalone application directory was not produced."
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
