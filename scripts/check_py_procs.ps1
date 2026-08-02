$ps = Get-Process python -ErrorAction SilentlyContinue
foreach ($p in $ps) {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
        Write-Host "$($p.Id): $cmd"
    } catch {
        Write-Host "$($p.Id): <no access>"
    }
}
