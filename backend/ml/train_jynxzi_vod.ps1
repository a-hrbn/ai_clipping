param(
    [string]$DateFrom = "",
    [string]$DateTo = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

if (-not $DateFrom) {
    $DateFrom = (Get-Date).AddDays(-30).ToString("yyyy-MM-dd")
}
if (-not $DateTo) {
    $DateTo = (Get-Date).ToString("yyyy-MM-dd")
}

$LabelsPath = Join-Path $BackendDir "ml\artifacts\jynxzi_twitch_clip_labels.jsonl"
$DatasetPath = Join-Path $BackendDir "ml\artifacts\jynxzi_vod_training_rows.jsonl"
$ShortFormPath = Join-Path $BackendDir "ml\artifacts\real_short_form_training_rows.jsonl"
$ArtifactDir = Join-Path $BackendDir "ml\artifacts"
$WorkDir = Join-Path $ArtifactDir "work\jynxzi"
$LogDir = Join-Path $ArtifactDir "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLogPath = Join-Path $LogDir ("train_jynxzi_vod_{0}.log" -f $Timestamp)
$LatestLogPath = Join-Path $LogDir "train_jynxzi_vod_latest.log"

Set-Location $BackendDir
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
} else {
    $PythonBin = "py"
}

function Write-RunLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $Line
    Add-Content -Path $RunLogPath -Value $Line
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-RunLog ("Running: {0} {1}" -f $PythonBin, ($Arguments -join " "))

    $allArgs = New-Object System.Collections.Generic.List[string]
    if ($PythonBin -eq "py") {
        [void]$allArgs.Add("-3")
    }
    foreach ($arg in $Arguments) {
        [void]$allArgs.Add($arg)
    }
    $argumentString = [string]::Join(" ", ($allArgs | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }))

    $stdoutPath = Join-Path $LogDir ("stdout_{0}.log" -f ([guid]::NewGuid().ToString("N")))
    $stderrPath = Join-Path $LogDir ("stderr_{0}.log" -f ([guid]::NewGuid().ToString("N")))

    try {
        $process = Start-Process `
            -FilePath $PythonBin `
            -ArgumentList $argumentString `
            -WorkingDirectory $BackendDir `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -NoNewWindow `
            -PassThru `
            -Wait
    } finally {
        foreach ($path in @($stdoutPath, $stderrPath)) {
            if (Test-Path $path) {
                Get-Content $path | ForEach-Object {
                    Write-Host $_
                    Add-Content -Path $RunLogPath -Value $_
                }
            }
        }
        Remove-Item $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
    }

    if ($process.ExitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1} {2}" -f $process.ExitCode, $PythonBin, ($Arguments -join " "))
    }
}

Set-Content -Path $RunLogPath -Value ("[{0}] Training run started" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

try {
    Write-RunLog ("Log file: {0}" -f $RunLogPath)
    Write-RunLog ("Date range: {0} to {1}" -f $DateFrom, $DateTo)

    Write-RunLog "Collecting Twitch clip labels for jynxzi..."
    Invoke-PythonStep @(
        "ml\collect_twitch_clip_labels.py",
        "--broadcaster", "jynxzi",
        "--date-from", $DateFrom,
        "--date-to", $DateTo,
        "--output", $LabelsPath
    )

    Write-RunLog "Training VOD-first ranker for jynxzi..."
    Invoke-PythonStep @(
        "ml\train_vod_ranker.py",
        "--labels", $LabelsPath,
        "--profile", "jynxzi",
        "--dataset-out", $DatasetPath,
        "--artifact-dir", $ArtifactDir,
        "--work-dir", $WorkDir,
        "--include-short-form", $ShortFormPath
    )

    Write-RunLog "Evaluating resulting dataset..."
    Invoke-PythonStep @(
        "ml\evaluate_twitch_ranker.py",
        "--dataset", $DatasetPath
    )

    Write-RunLog "Done."
    Write-RunLog ("Labels:   {0}" -f $LabelsPath)
    Write-RunLog ("Dataset:  {0}" -f $DatasetPath)
    Write-RunLog ("Artifact: {0}" -f (Join-Path $ArtifactDir "jynxzi_ranker.json"))
}
catch {
    Write-RunLog ("Run failed: {0}" -f $_.Exception.Message)
    throw
}
finally {
    Add-Content -Path $RunLogPath -Value ("[{0}] Training run finished" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Copy-Item -Path $RunLogPath -Destination $LatestLogPath -Force
}
