#!/bin/bash

echo "🐕 Starting Dog Detector Web App..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    # Start PulseAudio for sound in Docker
    pulseaudio --start --verbose
fi

# Download YOLO model if not present
if [ ! -f "yolov8n.pt" ]; then
    echo "📥 Downloading YOLO model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# Start the application
python app.py