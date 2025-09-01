#!/bin/bash
# install_system.sh
# Vehicle License Plate Recognition (VLPR) System Installation Script
# Compatible with Ubuntu/Debian-based systems including Raspberry Pi OS

set -e  # Exit on any error

echo "🚗 Setting up Vehicle License Plate Recognition (VLPR) System"
echo "================================================================"

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# Detect system architecture
ARCH=$(uname -m)
print_status "Detected architecture: $ARCH"

# Check for supported OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    print_status "Detected OS: $NAME $VERSION"
else
    print_warning "Cannot detect OS. Proceeding with generic installation..."
fi

# --- SYSTEM UPDATE ---
print_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# --- SYSTEM DEPENDENCIES ---
print_step "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libatlas-base-dev libhdf5-dev libopencv-dev
sudo apt install -y libgl1-mesa-glx libglib2.0-0 ffmpeg libsm6 libxext6
sudo apt install -y libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module
sudo apt install -y git curl wget

# Additional dependencies for video processing
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base
sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
sudo apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-libav

print_status "System dependencies installed successfully!"

# --- CREATE PYTHON VIRTUAL ENVIRONMENT ---
print_step "Creating Python virtual environment..."
if [ -d "venv_vlpr" ]; then
    print_warning "Virtual environment already exists. Removing old one..."
    rm -rf venv_vlpr
fi

python3 -m venv venv_vlpr
source venv_vlpr/bin/activate
print_status "Virtual environment created and activated!"

# --- UPGRADE PIP ---
print_step "Upgrading pip..."
pip install --upgrade pip

# --- PYTHON PACKAGES ---
print_step "Installing Python packages..."

# Detect if we're on Raspberry Pi or other ARM devices
if [[ "$ARCH" == "armv7l" ]] || [[ "$ARCH" == "aarch64" ]]; then
    print_status "Installing packages optimized for ARM architecture..."
    # ARM-specific PyTorch installation
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # Use headless OpenCV for better compatibility
    pip install opencv-python-headless==4.8.1.78
else
    print_status "Installing packages for x86_64 architecture..."
    # Standard PyTorch installation for x86_64
    pip install torch torchvision torchaudio
    # Full OpenCV with GUI support
    pip install opencv-python
fi

# Core YOLO and ML packages
pip install ultralytics
pip install numpy
pip install Pillow
pip install onnxruntime

# Optional but recommended packages
pip install easyocr  # Alternative OCR engine
pip install flask    # For web streaming
pip install requests # For API calls

# Development and testing tools
pip install pytest
pip install black    # Code formatter
pip install flake8   # Linter

print_status "Python packages installed successfully!"

# --- SYSTEM VERIFICATION ---
print_step "Verifying installation..."

# Test imports
python3 -c "
import sys
try:
    from ultralytics import YOLO
    import cv2
    import torch
    import numpy as np
    print('✅ All core packages imported successfully!')
    print(f'📦 OpenCV version: {cv2.__version__}')
    print(f'🔥 PyTorch version: {torch.__version__}')
    print(f'🎯 CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || {
    print_error "Package verification failed!"
    exit 1
}

# --- SETUP COMPLETION ---
print_status "Installation completed successfully! 🎉"
echo ""
echo "================================================================"
echo "🚗 Vehicle License Plate Recognition System Ready!"
echo "================================================================"
echo ""
echo "📋 Next Steps:"
echo "1. Activate the environment: source venv_vlpr/bin/activate"
echo "2. Verify the setup: python system_verification.py"
echo "3. Run detection: python main.py"
echo ""
echo "📁 Important Files:"
echo "• main.py                   - Main detection script"
echo "• system_verification.py   - System verification script"
echo "• models/                   - YOLO model files directory"
echo "• tests/                    - Test video and image files"
echo ""
echo "🔗 For more information, visit:"
echo "   https://github.com/GaruVA/Vehicle-License-Plate-Recognition"
echo ""
echo "⚠️  Remember to:"
echo "• Place your trained models in the models/ directory"
echo "• Update camera URLs in the scripts as needed"
echo "• Ensure proper lighting for optimal detection"
echo ""
print_status "Happy detecting! 🚗💨"
