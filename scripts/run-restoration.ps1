param(
  [Parameter(Mandatory=$true)][string]$InputPath,
  [Parameter(Mandatory=$true)][string]$OutputPath,
  [string]$Profile = 'hmi-screen'
)

$ErrorActionPreference = 'Stop'
$pluginRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $pluginRoot '.runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw 'Runtime is absent. Run scripts\setup-runtime.ps1 first.'
}
$tool = Join-Path $pluginRoot 'skills\evidence-media-restoration\scripts\evidence_media_tool.py'
$config = Join-Path $pluginRoot ("skills\evidence-media-restoration\assets\profiles\{0}.yaml" -f $Profile)
& $pythonExe $tool restore $InputPath $OutputPath --config $config

