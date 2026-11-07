# Windows PowerShell 5.1 and PowerShell 7: arguments pass through unchanged.
$ErrorActionPreference = 'Stop'
$PythonExecutable = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $PythonExecutable (Join-Path $PSScriptRoot 'run_raw_native_benchmark.py') @args
exit $LASTEXITCODE
