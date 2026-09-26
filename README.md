# 🚦 Road Intelligence & Traffic Monitoring System

AI-powered computer-vision traffic monitoring system built with Python, OpenCV, YOLO and ByteTrack.

## Features

- YOLO vehicle detection
- Car / motorcycle / bus / truck classification
- ByteTrack multi-object tracking
- Persistent vehicle IDs
- Current vehicle count
- Line-crossing vehicle counting
- Perspective-aware speed estimation
- EMA speed smoothing
- Direction detection with multi-frame history
- Traffic congestion scoring
- Streamlit live dashboard
- Vehicle distribution analytics
- Speed / vehicle / congestion charts

## Architecture

```text
Video
  ↓
OpenCV
  ↓
YOLO Detection
  ↓
ByteTrack Tracking
  ↓
Vehicle Classification + IDs
  ├── Current Vehicle Count
  ├── Line Crossing
  ├── Perspective Calibration
  │      ↓
  │   Real-world speed
  │      ↓
  │   EMA smoothing
  ├── Direction Detection
  └── Congestion Analysis
          ↓
    Streamlit Dashboard
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install ultralytics opencv-python streamlit pandas numpy
```

Put these files in the project folder:

```text
road-intelligence/
├── demo.mp4
├── yolo11n.pt
├── calibrate.py
├── calibration.json
├── dashboard.py
└── README.md
```

## Calibration

Run:

```bash
python calibrate.py
```

Click four road corners in this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

The selected quadrilateral should represent a region of road for which you know the real-world width and length.

Edit these values in `calibrate.py` before saving:

```python
ROAD_WIDTH_METERS = 10.0
ROAD_LENGTH_METERS = 30.0
```

Then run:

```bash
streamlit run dashboard.py
```

## Important limitation

Speed accuracy depends on calibration quality. The system should be validated against known road distances or reference measurements before using speed values operationally.

## Technologies

Python, OpenCV, Ultralytics YOLO, ByteTrack, NumPy, Pandas, Streamlit.
