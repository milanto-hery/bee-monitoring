# Bee Monitoring AI 🐝

A professional Python package for monitoring, tracking, and counting honey bees using YOLOv11 and Computer Vision.

![Bee Monitoring Demo](output/bee_monitoring_demo.gif)


## Features
- **Real-time Tracking**: Uses `sv.ByteTrack` for reliable individual bee identification.
- **Automated Counting**: Monitor hive entrance traffic with line-crossing detection.
- **Custom HUD**: Visual feedback with detection logs and performance metrics.
- **Dual Mode**: 
  - **CLI Scripts**: Optimized for performance and production use.
  - **Jupyter Notebook**: Interactive tutorial for learning and experimentation.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training (`train.py`)
Train a new model on your own dataset or using Roboflow.
```bash
python train.py --roboflow_key YOUR_API_KEY --epochs 50 --model_size n
```

### 2. Detection & Monitoring

#### Image Detection (`detect_image.py`)
Run inference on a single image.
```bash
python detect_image.py --source path/to/image.jpg --weights yolo11n.pt
```

#### Video Monitoring (`detect_video.py`)
Run real-time tracking and counting on a video file or webcam.
```bash
python detect_video.py --weights yolo11n.pt --source path/to/video.mp4 --show
```
*Press `q` to exit the video preview.*

### 3. Tutorial (`demo.ipynb`)
Explore the logic interactively by opening the `demo.ipynb` notebook.

## Key Classes
- **Bees**: Standard workers.
- **Pollen Bees**: Foragers with pollen sacks.
- **Drones**: Larger male bees.
- **Queens**: The hive's mother (triggers special alerts in `detect.py`).

## Project Structure
- `train.py`: Training script with CLI arguments.
- `detect_image.py`: Static image detection script.
- `detect_video.py`: Real-time tracking and counting module with `BeeMonitor` class.
- `demo.ipynb`: Interactive tutorial.
- `requirements.txt`: Project dependencies.
- `.gitignore`: Clean repository configuration.
