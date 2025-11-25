Param(
    [string]$ImageName = "chromadb-gpu:latest",
    [string]$ContainerName = "chromadb-gpu-dev",
    [switch]$Rebuild
)

Write-Host "Building Docker image '$ImageName' (CUDA + Torch GPU)..."
if ($Rebuild) { docker build --no-cache -t $ImageName . } else { docker build -t $ImageName . }

if ($LASTEXITCODE -ne 0) { Write-Error "Image build failed."; exit 1 }

# Remove existing container if present
$existing = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
if ($existing) {
    Write-Host "Removing existing container '$ContainerName'..."
    docker rm -f $ContainerName | Out-Null
}

# Resolve workspace path for volume mount
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
Write-Host "Starting container '$ContainerName' with workspace mount: $RepoRoot"

docker run --gpus all -d `
  --name $ContainerName `
  -v "${RepoRoot}:/workspace" `
  -w /workspace `
  -p 8000:8000 `
  $ImageName tail -f /dev/null

if ($LASTEXITCODE -ne 0) { Write-Error "Container start failed."; exit 1 }

Write-Host "Container started. To attach in VS Code:"
Write-Host "  1. Install 'Dev Containers' extension."
Write-Host "  2. Open Command Palette -> 'Dev Containers: Attach to Running Container'"
Write-Host "  3. Select '$ContainerName'"

Write-Host "Verify GPU inside container (optional): docker exec -it $ContainerName python -c 'import torch; print(torch.cuda.is_available())'"
