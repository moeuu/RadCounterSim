[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRepository,

    [Parameter(Mandatory = $true)]
    [string]$OutputStep,

    [switch]$Force,
    [switch]$KeepSolidWorksOpen
)

$ErrorActionPreference = "Stop"
$sourceRoot = (Resolve-Path -LiteralPath $SourceRepository).Path
$assemblyPath = Join-Path $sourceRoot "Building.SLDASM"
if (-not (Test-Path -LiteralPath $assemblyPath -PathType Leaf)) {
    throw "Building.SLDASM was not found below $sourceRoot"
}

$outputPath = [System.IO.Path]::GetFullPath($OutputStep)
if ([System.IO.Path]::GetExtension($outputPath).ToLowerInvariant() -notin @(".step", ".stp")) {
    throw "OutputStep must end in .step or .stp"
}
if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
    throw "Output already exists: $outputPath. Pass -Force to replace it."
}
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($outputPath)) | Out-Null

$solidWorks = $null
$model = $null
$openErrors = 0
$openWarnings = 0
$saveErrors = 0
$saveWarnings = 0
try {
    $solidWorks = New-Object -ComObject SldWorks.Application
    $solidWorks.Visible = $false
    $model = $solidWorks.OpenDoc6(
        $assemblyPath,
        2,
        1,
        "",
        [ref]$openErrors,
        [ref]$openWarnings
    )
    if ($null -eq $model) {
        throw "SolidWorks could not open the assembly (errors=$openErrors warnings=$openWarnings)"
    }
    $model.ForceRebuild3($false) | Out-Null
    $saved = $model.Extension.SaveAs(
        $outputPath,
        0,
        1,
        $null,
        [ref]$saveErrors,
        [ref]$saveWarnings
    )
    if (-not $saved -or $saveErrors -ne 0 -or -not (Test-Path -LiteralPath $outputPath)) {
        throw "STEP export failed (saved=$saved errors=$saveErrors warnings=$saveWarnings)"
    }

    $sourceRevision = $null
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $sourceRevision = (& git -C $sourceRoot rev-parse HEAD 2>$null)
    }
    $provenance = [ordered]@{
        schema_version = 1
        source_repository = "https://github.com/Qualot/fukushima_daiichi_solidworks"
        source_revision = $sourceRevision
        source_assembly = $assemblyPath
        source_license = "CC BY 4.0"
        source_author_attribution = "Qualot/fukushima_daiichi_solidworks contributors"
        exported_step = $outputPath
        exported_step_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
        solidworks_revision = $solidWorks.RevisionNumber()
        open_warnings = $openWarnings
        save_warnings = $saveWarnings
    }
    $provenancePath = "$outputPath.provenance.json"
    $provenance | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $provenancePath -Encoding utf8
    $provenance | ConvertTo-Json -Depth 4
}
finally {
    if ($null -ne $solidWorks -and $null -ne $model) {
        $solidWorks.CloseDoc($model.GetTitle())
    }
    if ($null -ne $solidWorks -and -not $KeepSolidWorksOpen) {
        $solidWorks.ExitApp()
    }
    if ($null -ne $model) {
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($model) | Out-Null
    }
    if ($null -ne $solidWorks) {
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($solidWorks) | Out-Null
    }
}
