# Voice Symptom Intake & Documentation Assistant - Setup Script (Windows)
# This script helps set up the environment and verify requirements

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Voice Symptom Intake & Documentation Assistant" -ForegroundColor Cyan
Write-Host "Setup Script (Windows)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check for Docker
Write-Host "Checking for Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker not found. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "  Visit: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}

# Check for Docker Compose
Write-Host "Checking for Docker Compose..." -ForegroundColor Yellow
try {
    docker-compose --version | Out-Null
    Write-Host "✓ Docker Compose found" -ForegroundColor Green
} catch {
    try {
        docker compose version | Out-Null
        Write-Host "✓ Docker Compose found" -ForegroundColor Green
    } catch {
        Write-Host "✗ Docker Compose not found." -ForegroundColor Red
        exit 1
    }
}

# Check for .env file
Write-Host ""
Write-Host "Checking for .env file..." -ForegroundColor Yellow
if (Test-Path .env) {
    Write-Host "✓ .env file found" -ForegroundColor Green
} else {
    Write-Host "! .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "✓ Created .env file" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Please edit .env and add your Hugging Face token!" -ForegroundColor Yellow
    Write-Host "   1. Get token from: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
    Write-Host "   2. Accept terms for: google/medasr and google/medgemma-1.5-4b-it" -ForegroundColor Cyan
    Write-Host "   3. Add token to .env file: HF_TOKEN=your_token_here" -ForegroundColor Cyan
    Write-Host ""
}

# Check for GPU (optional)
Write-Host "Checking for NVIDIA GPU (optional)..." -ForegroundColor Yellow
try {
    nvidia-smi | Out-Null
    Write-Host "✓ NVIDIA GPU detected" -ForegroundColor Green
    Write-Host ""
    Write-Host "  You can use GPU mode: docker-compose --profile gpu up" -ForegroundColor Cyan
} catch {
    Write-Host "! No NVIDIA GPU detected" -ForegroundColor Yellow
    Write-Host "  You can still use CPU mode: docker-compose --profile cpu up" -ForegroundColor Cyan
}

# Create directories
Write-Host ""
Write-Host "Creating required directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path test_data | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Make sure HF_TOKEN is set in .env file" -ForegroundColor White
Write-Host "2. Run: docker-compose --profile cpu up --build" -ForegroundColor White
Write-Host "   (or --profile gpu if you have GPU)" -ForegroundColor Gray
Write-Host "3. Open: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "For help, see README.md" -ForegroundColor Cyan
Write-Host ""
