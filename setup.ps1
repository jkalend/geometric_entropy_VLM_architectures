# Setup script for geometric-entropy project
# Uses PyTorch with CUDA 13.0 (cu130) for RTX 4090

$ErrorActionPreference = "Stop"

Write-Output "Creating venv..."
python -m venv venv

Write-Output "Activating venv and installing PyTorch (cu130)..."
& .\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

Write-Output "Installing remaining dependencies..."
pip install -r requirements.txt

Write-Output "Setup complete."
