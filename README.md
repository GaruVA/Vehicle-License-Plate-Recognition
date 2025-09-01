# Vehicle License Plate Recognition

An advanced Sri Lankan license plate recognition system using YOLOv11 with intelligent character recognition, multi-frame tracking, and robust validation.

## 🚗 Features

### Core Detection
- **YOLOv11-based plate detection** with high accuracy
- **Custom character recognition** with proper class mapping
- **Smart character ordering** for both single-row and two-row plates
- **Multi-frame consensus tracking** for improved accuracy

### Sri Lankan Plate Support
- **Strict format validation** for Sri Lankan plates (XX-0000/XXX-0000)
- **Intelligent OCR corrections** based on character position
- **Dash positioning fixes** for misplaced characters
- **Comprehensive prefix validation** with all valid letter combinations

### Advanced Processing
- **Real-time tracking** with IoU-based plate matching
- **Consensus text validation** across multiple frames
- **Enhanced preprocessing** with CLAHE and bilateral filtering
- **Performance optimization** with frame skipping for live streams

## 📁 Project Structure

```
vehicle-license-plate-recognition/
├── main.py                   # Main Detection system
├── install_system.sh         # System setup script
├── system_verification.py    # System verification tools
├── models/                   # YOLOv8 model files
│   ├── plate_detection.pt    # Plate detection model
│   ├── character_recognition.pt # Character recognition model
│   └── optimized/           # TensorRT optimized models (optional)
└── tests/                   # Test video files
    ├── dges1.mp4
    └── dges2.mp4
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install system dependencies
sudo ./install_system.sh

# Create virtual environment
python3 -m venv venv_vlpr
source venv_vlpr/bin/activate

# Install Python dependencies
pip install ultralytics opencv-python numpy
```

### Usage

#### Process Video File
```bash
python enhanced_main.py tests/dges1.mp4
```

#### Live Camera Stream
```bash
# Edit CONFIG['VIDEO_SOURCE'] in enhanced_main.py to your camera/RTSP stream
python enhanced_main.py
```

#### Configuration
Edit the `CONFIG` dictionary in `enhanced_main.py`:
```python
CONFIG = {
    'PLATE_MODEL_PATH': "path/to/plate_detection.pt",
    'CHAR_MODEL_PATH': "path/to/character_recognition.pt", 
    'VIDEO_SOURCE': "rtsp://your-camera-stream",
    'CONFIDENCE_THRESHOLD': 0.5,
    'TRACKER_MAX_AGE': 30,
    'TRACKER_MIN_HITS': 3,
    'FRAME_SKIP': 3,  # For performance optimization
}
```

## 🎯 Detection Results

### Validation Examples
- ✅ `CBN-5808` - Valid 3-letter format
- ✅ `CBF-8675` - Valid 3-letter format  
- ✅ `KD-4534` - Valid 2-letter format
- ✅ `CAC-7751` - Valid 3-letter format
- ❌ `KIA-737` - Invalid (car brand + 3 digits)
- ❌ `DBT-11` - Invalid (only 2 digits)
- ❌ `CAC-77` - Invalid (only 2 digits)

### Smart Corrections
- `KDA-534` → `KD-4534` (dash positioning fix)
- `K3-4534` → `KE-4534` (positional OCR correction)
- `CBN5808` → `CBN-5808` (automatic formatting)

## 🔧 Key Features Explained

### Multi-Frame Tracking
- Tracks plates across multiple frames using IoU matching
- Builds consensus text from multiple detections
- Maintains confidence scores and hit counts
- Handles temporary occlusions and detection gaps

### Sri Lankan Plate Validation
- **Format Patterns**: Supports XX-0000 and XXX-0000 formats
- **Comprehensive Prefix System**: All valid letter combinations (AA-ZZ, AAA-ZZZ)
- **Invalid Pattern Detection**: Rejects car brands, insufficient digits, etc.
- **Strict Requirements**: Exactly 4 digits required

### Character Recognition Enhancements
- **Class Mapping**: Proper mapping from model class IDs to characters
- **Positional Corrections**: Context-aware OCR error correction
- **Two-Row Support**: Smart ordering for stacked plate layouts
- **Quality Filtering**: Removes obviously incorrect detections

### Performance Features
- **Frame Skipping**: Configurable processing rate for live streams
- **Buffer Management**: Optimized for RTSP streams
- **Individual Plate Windows**: Real-time crop display
- **Debug Output**: Comprehensive logging and statistics

## 🎮 Controls

- **'q'**: Quit the application
- **'s'**: Save current frame and detected plate crops
- **Mouse**: Click to focus on detection windows

## 📊 Output Information

The system provides real-time information including:
- **FPS**: Current processing frame rate
- **Tracks**: Number of active plate tracks
- **Valid**: Number of valid plates detected
- **Confidence**: Detection confidence scores
- **Hit Count**: Frames where each plate was detected
- **Consensus Text**: Most frequent text across frames

## 🔬 Technical Details

### Models Required
- **Plate Detection Model**: YOLOv11 trained for license plate detection
- **Character Recognition Model**: YOLOv11 trained for character classification

### Dependencies
- `ultralytics`: YOLOv11 framework
- `opencv-python`: Computer vision operations
- `numpy`: Numerical computations
- `tempfile`: Temporary file handling
- `re`: Regular expression validation

### Optimization Options
- **TensorRT**: Use optimized models in `models/optimized/`
- **CUDA**: GPU acceleration (if available)
- **Frame Skipping**: Adjust `FRAME_SKIP` for performance vs accuracy

## 📄 License

This project is open source. Please ensure you have proper licensing for the YOLOv11 models used.

