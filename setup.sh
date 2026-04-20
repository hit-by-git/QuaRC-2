#!/bin/bash
# Quick setup and run script for QuaRC
# This script installs dependencies and runs the experiments

set -e

echo "=================================="
echo "QuaRC Setup and Execution Script"
echo "=================================="

# Check Python version
echo "Checking Python version..."
python --version

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo ""
echo "=================================="
echo "Setup completed successfully!"
echo "=================================="
echo ""
echo "To run training on CIFAR-100 with MobileNetV2 (2-bit, 1% coreset):"
echo "  python main.py"
echo ""
echo "To reproduce all paper results:"
echo "  python run_experiments.py"
echo ""
echo "=================================="
