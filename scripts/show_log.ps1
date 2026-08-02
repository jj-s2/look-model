$logPath = 'c:\Users\John\Desktop\look model\experiments\logs\import_verbose.log'
if (Test-Path $logPath) {
    $f = Get-Item $logPath
    Write-Host "Size: $($f.Length) bytes, Modified: $($f.LastWriteTime)"
    Write-Host "=== Last 60 lines ==="
    Get-Content $logPath -Tail 60
} else {
    Write-Host "log not created"
}
