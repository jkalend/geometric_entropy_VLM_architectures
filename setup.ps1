# Setup script for geometric-entropy project
# Uses PyTorch with CUDA 13.0 (cu130) for RTX 4090

$ErrorActionPreference = "Stop"

Write-Host "Creating venv..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Activating venv and installing PyTorch (cu130)..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Setup complete." -ForegroundColor Green
