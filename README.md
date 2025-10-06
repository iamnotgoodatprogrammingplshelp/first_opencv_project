# Emotion & Motion Detection System

A comprehensive OpenCV-based system for real-time emotion detection, motion tracking, and activity logging with database storage.

## Project Structure

```
emotion-motion-detector/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── camera.py           # Camera capture and management
│   ├── face_detector.py    # Face detection using Haar cascades
│   ├── emotion_detector.py # Emotion recognition
│   ├── motion_detector.py  # Motion and action detection
│   ├── database.py         # Database operations
│   └── utils.py            # Helper utilities
├── models/                 # Pre-trained models directory
│   └── download_models.py
├── tests/
│   ├── __init__.py
│   ├── test_database.py
│   ├── test_detectors.py
│   └── fixtures/
├── data/
│   └── detections.db       # SQLite database
├── config/
│   └── config.yaml         # Configuration file
├── scripts/
│   ├── run_detector.py     # Main detection script
│   ├── view_database.py    # Database viewer
│   └── export_data.py      # Export to CSV/JSON
├── notebooks/
│   └── analysis.ipynb      # Data analysis notebook
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── README.md
└── .env.example
```

## Features

- **Real-time Face Detection**: Using OpenCV Haar Cascades
- **Emotion Recognition**: Detect 7 emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- **Motion Detection**: Track movement and classify actions (walking, running, sitting, standing, jumping)
- **Database Storage**: SQLite database for storing all detections with timestamps
- **Video Recording**: Optional recording of detection sessions
- **Analytics Dashboard**: Visualize emotion and motion patterns over time
- **REST API**: Query detection data programmatically
- **Privacy Mode**: Blur faces while detecting emotions
