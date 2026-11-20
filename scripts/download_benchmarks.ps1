#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Plan', 'Download', 'Verify')]
    [string]$Action = 'Plan',
    [ValidateSet('crvd', 'real-rawvsr')]
    [string]$Dataset = 'crvd',
    [ValidateSet('test', 'train', 'indoor', 'outdoor')]
    [string]$Split = 'test',
    [int[]]$Scenes,
    [string]$DataRoot,
    [string]$Python,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Find-Python {
    param([string]$ExplicitPython)

    $commands = @()
    if ($ExplicitPython) {
        $commands = @(Get-Command $ExplicitPython -CommandType Application -ErrorAction SilentlyContinue)
    } else {
        foreach ($name in @('python', 'python3', 'py')) {
            $commands += @(Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue)
        }
    }
    foreach ($command in $commands) {
        $exe = $command.Source
        if (-not $ExplicitPython -and $exe -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]') {
            continue
        }
        $prefix = @()
        if ([System.IO.Path]::GetFileNameWithoutExtension($exe) -eq 'py') {
            $prefix = @('-3')
        }
        try {
            $version = & $exe @prefix -c 'import sys; print(sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 10) else 3)' 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = $exe; Prefix = $prefix; Version = ($version -join '') }
            }
        } catch {
            # Try the next installed interpreter.
        }
    }
    throw 'Python 3.10+ was not found. Use -Python C:\path\to\python.exe. Download additionally needs gdown from requirements-download.txt; install it separately when authorized. Verify uses only the Python standard library. Plan works without Python.'
}

try {
    if ($Json -and $Action -ne 'Plan') {
        throw '-Json is supported only with -Action Plan.'
    }
    $repoRoot = Split-Path -Parent $PSScriptRoot
    if (-not $DataRoot) {
        $DataRoot = Join-Path $repoRoot 'data'
    }
    $DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
    $manifestPath = Join-Path $repoRoot 'benchmarks/sources.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema_version -ne 1) {
        throw 'Unsupported benchmarks/sources.json schema_version.'
    }
    $source = $manifest.datasets.PSObject.Properties[$Dataset].Value

    if ($Dataset -eq 'real-rawvsr') {
        if ($PSBoundParameters.ContainsKey('Scenes') -or $Split -ne 'test') {
            throw 'Real-RawVSR does not support -Scenes or a split other than test; its exact evaluation selection is unverified.'
        }
        if ($Action -ne 'Plan') {
            throw ('Real-RawVSR requires manual access through the official Baidu provider: {0} (code: {1}). {2}' -f $source.download_url, $source.access_code, $source.note)
        }
        $plan = [ordered]@{
            dataset = $Dataset
            mode = 'manual'
            archive_count = $null
            scenes = @()
            selectedfiles = @()
            paths = @()
            destination = Join-Path $DataRoot 'manual/real-rawvsr'
            license = $source.license
            license_url = $source.license_url
            download_url = $source.download_url
            access_code = $source.access_code
            note = $source.note
        }
    } else {
        switch ($Split) {
            'test' { $baseScenes = @(7..11) }
            'train' { $baseScenes = @(1..6) }
            'indoor' { $baseScenes = @(1..11) }
            'outdoor' { $baseScenes = @(1..10) }
        }
        if ($PSBoundParameters.ContainsKey('Scenes')) {
            if (@($Scenes).Count -eq 0) {
                throw 'Scenes must contain at least one scene number.'
            }
            if (@($Scenes | Sort-Object -Unique).Count -ne @($Scenes).Count) {
                throw 'Scenes must not contain duplicates.'
            }
            foreach ($scene in $Scenes) {
                if ($scene -notin $baseScenes) {
                    throw ('Scene {0} is outside split {1}; allowed: {2}.' -f $scene, $Split, ($baseScenes -join ', '))
                }
            }
            $selectedScenes = @($Scenes | Sort-Object)
        } else {
            $selectedScenes = $baseScenes
        }

        $collections = @('indoor_raw_gt', 'indoor_raw_noisy')
        if ($Split -eq 'outdoor') {
            $collections = @('outdoor_raw_noisy')
        }
        $files = @($source.files | Where-Object {
            $_.collection -in $collections -and $_.scene -in $selectedScenes
        } | Sort-Object scene, collection)
        if (@($files.id | Sort-Object -Unique).Count -ne $files.Count -or
            @($files.name | Sort-Object -Unique).Count -ne $files.Count) {
            throw 'The selected manifest contains duplicate file IDs or names.'
        }
        foreach ($scene in $selectedScenes) {
            $sceneFiles = @($files | Where-Object { $_.scene -eq $scene })
            if ($sceneFiles.Count -ne $collections.Count -or
                @($sceneFiles.collection | Sort-Object -Unique).Count -ne $collections.Count) {
                throw ('Manifest must contain exactly {0} archive(s) for scene {1} in split {2}.' -f $collections.Count, $scene, $Split)
            }
        }
        $paths = @()
        foreach ($file in $files) {
            $expectedName = '{0}_scene{1}.zip' -f $file.collection, $file.scene
            if ($file.name -ne $expectedName -or $file.id -notmatch '^[A-Za-z0-9_-]{25,}$') {
                throw ('Invalid archive name or Drive ID in manifest: {0}.' -f $file.name)
            }
            $archiveFolder = Join-Path $DataRoot ('archives/crvd/{0}' -f $file.collection)
            $paths += Join-Path $archiveFolder $file.name
        }
        $plan = [ordered]@{
            dataset = $Dataset
            split = $Split
            archive_count = $files.Count
            scenes = @($selectedScenes)
            selectedfiles = @($files)
            paths = @($paths)
            data_root = $DataRoot
            license = $source.license
            license_url = $source.license_url
            note = $source.note
        }
    }

    if ($Action -eq 'Plan') {
        if ($Json) {
            $plan | ConvertTo-Json -Depth 8
        } elseif ($Dataset -eq 'real-rawvsr') {
            Write-Output ('Real-RawVSR: manual download {0} (code: {1})' -f $plan.download_url, $plan.access_code)
            Write-Output ('License: {0}' -f $plan.license)
            Write-Output ('Destination: {0}' -f $plan.destination)
            Write-Output $plan.note
        } else {
            Write-Output ('CRVD {0}: {1} archive(s); scenes {2}' -f $Split, $plan.archive_count, ($plan.scenes -join ', '))
            Write-Output ('License: {0}' -f $plan.license)
            Write-Output ('Data root: {0}' -f $DataRoot)
            Write-Output $plan.paths
            Write-Output 'Plan only: no network requests, directory creation, or downloads.'
        }
        exit 0
    }

    $interpreter = Find-Python -ExplicitPython $Python
    $cli = Join-Path $PSScriptRoot 'download_benchmarks.py'
    if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
        throw ('Python CLI is missing: {0}' -f $cli)
    }
    $cliArgs = @($cli, $Action.ToLowerInvariant(), '--dataset', $Dataset,
        '--split', $Split, '--data-root', $DataRoot, '--scenes')
    $cliArgs += @($selectedScenes | ForEach-Object { [string]$_ })
    if ($Json) {
        $cliArgs += '--json'
    }
    $pythonCommand = $interpreter.Command
    $pythonArgs = @($interpreter.Prefix) + $cliArgs
    & $pythonCommand @pythonArgs
    exit $LASTEXITCODE
} catch {
    Write-Error $_
    exit 1
}
