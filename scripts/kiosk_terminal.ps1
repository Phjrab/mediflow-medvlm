# Use the selected Python/Conda environment. GPU commands require Linux/WSL.
$ErrorActionPreference = "Stop"
$kioskPython = if ($env:MEDVLM_PYTHON) { $env:MEDVLM_PYTHON } else { "python" }
& $kioskPython (Join-Path $PSScriptRoot "kiosk_terminal.py") @args
exit $LASTEXITCODE
