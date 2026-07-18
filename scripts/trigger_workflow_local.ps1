param(
    [Parameter(Mandatory=$true)]
    [string]$Workflow
)

# Generalized version of trigger_scan_local.ps1 - same reasoning (GitHub's
# schedule: cron confirmed throttled, workflow_dispatch fires reliably),
# parameterized so one script + 5 separate Scheduled Tasks (one per
# workflow, each at its own matching daily/weekly time) covers Audit,
# Trade Diagnostic, VWAP Investigation, Backtest, and Realistic Backtest,
# instead of duplicating this file 5 times. scan.yml keeps its own
# dedicated trigger_scan_local.ps1 (5-minute repeat, already deployed) -
# left alone rather than migrated, since it's already working.
#
# Token read from the same local User-scope GITHUB_DISPATCH_TOKEN env var
# trigger_scan_local.ps1 uses - set once, shared by all trigger scripts.

$ErrorActionPreference = "Stop"

$token = [System.Environment]::GetEnvironmentVariable("GITHUB_DISPATCH_TOKEN", "User")
if (-not $token) {
    Write-Error "GITHUB_DISPATCH_TOKEN is not set. See scripts/trigger_scan_local.ps1's header for setup."
    exit 1
}

$owner = "SHALITH22"
$repo = "ashen-crypto-scanne"

$headers = @{
    Authorization = "Bearer $token"
    Accept        = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}
$body = @{ ref = "master" } | ConvertTo-Json

try {
    Invoke-RestMethod -Method Post `
        -Uri "https://api.github.com/repos/$owner/$repo/actions/workflows/$Workflow/dispatches" `
        -Headers $headers -Body $body -ContentType "application/json" | Out-Null
    Write-Output "$(Get-Date -Format o)  $Workflow dispatched OK"
} catch {
    Write-Output "$(Get-Date -Format o)  $Workflow dispatch FAILED: $($_.Exception.Message)"
}
