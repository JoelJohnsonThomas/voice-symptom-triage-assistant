#!/bin/bash

# Voice Symptom Intake & Documentation Assistant - Setup Script
# This script helps set up the environment and verify requirements

echo "================================================"
echo "Voice Symptom Intake & Documentation Assistant"
echo "Setup Script"
echo "================================================"
echo ""

# Check for Docker
echo "Checking for Docker..."
if command -v docker &> /dev/null; then
    echo "✓ Docker found: $(docker --version)"
else
    echo "✗ Docker not found. Please install Docker first."
    echo "  Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for Docker Compose
echo "Checking for Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "✓ Docker Compose found"
else
    echo "✗ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Check for .env file
echo ""
echo "Checking for .env file..."
if [ -f .env ]; then
    echo "✓ .env file found"
else
    echo "! .env file not found. Creating from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your Hugging Face token!"
    echo "   1. Get token from: https://huggingface.co/settings/tokens"
    echo "   2. Accept terms for: google/medasr and google/medgemma-1.5-4b-it"
    echo "   3. Add token to .env file: HF_TOKEN=your_token_here"
    echo ""
fi

# Check for GPU (optional)
echo "Checking for NVIDIA GPU (optional)..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
    echo "  You can use GPU mode: docker-compose --profile gpu up"
else
    echo "! No NVIDIA GPU detected"
    echo "  You can still use CPU mode: docker-compose --profile cpu up"
fi

# Create directories
echo ""
echo "Creating required directories..."
mkdir -p models test_data

echo "✓ Directories created"

# Summary
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Make sure HF_TOKEN is set in .env file"
echo "2. Run: docker-compose --profile cpu up --build"
echo "   (or --profile gpu if you have GPU)"
echo "3. Open: http://localhost:8000"
echo ""
echo "For help, see README.md"
echo ""
