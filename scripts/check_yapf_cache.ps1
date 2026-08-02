$cacheDir = Join-Path $env:LOCALAPPDATA 'Google\YAPF\0.43.0'
Write-Host "Cache dir: $cacheDir"
if (Test-Path $cacheDir) {
    Write-Host "Exists. Contents:"
    Get-ChildItem $cacheDir | Select-Object Name, Length, LastWriteTime | Format-Table
} else {
    Write-Host "not exists"
    # 尝试创建
    try {
        New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        Write-Host "Created successfully"
    } catch {
        Write-Host "Create FAILED: $_"
    }
}
