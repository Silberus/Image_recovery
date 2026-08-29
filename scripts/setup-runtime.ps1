param(
  [ValidateSet('core','extended','neural')]
  [string]$Profile = 'extended',
  [string]$Runtime = '.runtime'
)

$ErrorActionPreference = 'Stop'
$pluginRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $pluginRoot $Runtime
if (-not (Test-Path -LiteralPath $runtimePath)) {
  py -3.12 -m venv $runtimePath
}
$pythonExe = Join-Path $runtimePath 'Scripts\python.exe'
& $pythonExe -m pip install --upgrade pip
$requirements = Join-Path $pluginRoot ("requirements-{0}.txt" -f $Profile)
& $pythonExe -m pip install -r $requirements
& $pythonExe (Join-Path $pluginRoot 'skills\evidence-media-restoration\scripts\evidence_media_tool.py') selftest (Join-Path $pluginRoot '.selftest') --config (Join-Path $pluginRoot 'skills\evidence-media-restoration\assets\profiles\hmi-screen.yaml')
Write-Host "Runtime ready: $runtimePath"

