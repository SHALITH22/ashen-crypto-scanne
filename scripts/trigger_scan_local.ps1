# Fires scan.yml's workflow_dispatch trigger via GitHub's API - the SAME
# call the "Run workflow" button makes, which has fired instantly, 100%
# of the time, all day, unlike scan.yml's own `schedule:` cron (confirmed
# throttled to an effective ~55-90 minute cadence regardless of
# configured interval, tracked down over several hours - see the
# project's memory for the full investigation).
#
# Registered as a Windows Scheduled Task repeating every 5 minutes -
# gives real, precise timing while this PC is on, falling back to
# scan.yml's own (slower, throttled) schedule when it's off, rather than
# stopping entirely. Not a fragile replacement for the GitHub schedule,
# a supplement to it.
#
# The token is read from a LOCAL environment variable only
# ($env:GITHUB_DISPATCH_TOKEN, set once at the User level so it survives
# reboots - see setup_local_trigger.ps1) - never embedded in this file,
# never committed to git.

$ErrorActionPreference = "Stop"

$token = [System.Environment]::GetEnvironmentVariable("GITHUB_DISPATCH_TOKEN", "User")
if (-not $token) {
    Write-Error "GITHUB_DISPATCH_TOKEN is not set. Run setup_local_trigger.ps1 first."
    exit 1
}

$owner = "SHALITH22"
$repo = "ashen-crypto-scanne"
$workflow = "scan.yml"

$headers = @{
    Authorization = "Bearer $token"
    Accept        = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}
$body = @{ ref = "master" } | ConvertTo-Json

try {
    Invoke-RestMethod -Method Post `
        -Uri "https://api.github.com/repos/$owner/$repo/actions/workflows/$workflow/dispatches" `
        -Headers $headers -Body $body -ContentType "application/json" | Out-Null
    Write-Output "$(Get-Date -Format o)  scan.yml dispatched OK"
} catch {
    Write-Output "$(Get-Date -Format o)  scan.yml dispatch FAILED: $($_.Exception.Message)"
}
