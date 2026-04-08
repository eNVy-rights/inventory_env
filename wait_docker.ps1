$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$maxWait = 120
$start = Get-Date

while (((Get-Date) - $start).TotalSeconds -lt $maxWait) {
    $result = & $docker ps 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker is ready!"
        break
    }
    $elapsed = [int](((Get-Date) - $start).TotalSeconds)
    Write-Host "Still waiting... ($elapsed s)"
    Start-Sleep 5
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker daemon did not start within $maxWait seconds"
}
