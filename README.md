# 🚗 Vehicle License Plate Recognition (VLPR)

A real-time **Vehicle License Plate Recognition** system built with **YOLOv11** and **OpenCV**, supporting both local video files and **IP camera streams** (RTSP/HTTP). Optimized for **Linux**, **Raspberry Pi**, and **embedded systems**.

![License Plate Detection Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

---

## 🌟 Features

- **🎯 Real-time License Plate Detection** - High-accuracy plate detection using YOLOv11
- **🔤 Character Recognition** - OCR for reading license plate text
- **📹 Multi-Source Support** - Local videos, IP cameras (RTSP/HTTP), webcams
- **🖥️ Live Preview** - Real-time detection display with bounding boxes
- **💾 Save Detections** - Capture and save detected plates automatically

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/GaruVA/vehicle-license-plate-recognition.git
cd vehicle-license-plate-recognition
```

### 2. Run Installation Script
```bash
chmod +x install_system.sh
./install_system.sh
```

### 3. Activate Environment
```bash
source venv_vlpr/bin/activate
```

### 4. Verify Installation
```bash
python system_verification.py
```

### 5. Run Detection
```bash
# Run with video file
python main.py
```

---

## 📁 Project Structure

```
vehicle-license-plate-recognition/
├── 📄 README.md                    # This file
├── 🔧 install_system.sh           # Automated installation script
├── 🧪 system_verification.py      # Comprehensive system verification
├── 🎯 main.py                     # Core detection script
├── 📁 models/                     # Detection models
│   ├── 🤖 plate_detection.pt      # YOLOv8 plate detection model
│   ├── 🔤 character_recognition.pt # YOLOv8 character recognition model
│   └── 📁 optimized/              # TensorRT optimized models (optional)
├── 📁 tests/                      # Test files and videos
│   └── 🎬 dges1.mp4               # Sample test video
└── 📁 venv_vlpr/                  # Python virtual environment
```

---

## 🎮 Usage

### Basic Detection
```bash
# Activate environment
source venv_vlpr/bin/activate

# Run detection on video file
python main.py
```

### Controls
- **`q`** - Quit application
- **`s`** - Save current frame and detected plates
- **Mouse** - Click to focus on detection windows

### Configuration
Edit the camera URL in `main.py`:
```python
# IP Camera (RTSP/HTTP)
source = "rtsp://username:password@ip:port/stream"

# Local video file  
source = "tests/dges1.mp4"

# Webcam
source = 0
```

---

## 🔧 Advanced Configuration

### Video Source Options
The system supports multiple input sources:

1. **IP Camera (RTSP)**
   ```python
   source = "rtsp://username:password@192.168.1.100:554/stream"
   ```

2. **HTTP Stream** 
   ```python
   source = "http://192.168.1.100:8080/video"
   ```

3. **Local Video File**
   ```python
   source = "tests/dges1.mp4"
   ```

4. **USB Webcam**
   ```python
   source = 0  # Default camera
   ```

### Performance Optimization
- **Frame Skip**: Adjust `frame_skip` in main.py for better performance
- **Resolution**: Lower resolution improves speed
- **GPU**: CUDA acceleration automatically detected if available

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 18.04+, Debian 10+, Raspberry Pi OS
- **Python**: 3.8+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: 4 cores (ARM64 supported)

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 8GB+
- **Camera**: IP camera with RTSP/HTTP stream
- **Network**: Stable connection for IP cameras

### Supported Platforms
- ✅ **Ubuntu/Debian** (x86_64)
- ✅ **Raspberry Pi** (ARM64/ARMv7)
- ✅ **NVIDIA Jetson** (ARM64 + CUDA)
- ✅ **Linux Mint, Pop!_OS** (Ubuntu-based)

---

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall packages
source venv_vlpr/bin/activate
pip install --upgrade ultralytics opencv-python torch
```

**2. Camera Connection Issues**
```bash
# Test camera connectivity
python ip_camera_test.py

# Check firewall settings
sudo ufw allow from [camera_ip]
```

**3. Performance Issues**
```bash
# Monitor system resources
python test_setup.py

# Close unnecessary applications
# Use TensorRT optimization for GPU acceleration
```

**4. Model Loading Errors**
- Ensure model files are in `models/` directory
- Check file permissions: `chmod 644 models/*.pt`
- Verify model file integrity

### Getting Help
- 📖 **Wiki**: [GitHub Wiki](https://github.com/GaruVA/vehicle-license-plate-recognition/wiki)
- 🐛 **Issues**: [Report Bug](https://github.com/GaruVA/vehicle-license-plate-recognition/issues)
- 💬 **Discussions**: [Community Forum](https://github.com/GaruVA/vehicle-license-plate-recognition/discussions)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **YOLOv11**: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **OpenCV**: [Apache 2.0](https://opencv.org/license/)
- **PyTorch**: [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/master/LICENSE)

---

