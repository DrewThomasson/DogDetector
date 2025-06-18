from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import os
import sys
import time
import base64
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebDogDetector:
    def __init__(self):
        # Load YOLO model
        try:
            baked_model_path = '/app/yolov8n.pt'
            if os.path.exists(baked_model_path):
                self.model = YOLO(baked_model_path)
                print(f"Using baked-in YOLO model: {baked_model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                print("YOLO model downloaded and loaded successfully!")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            sys.exit(1)
        
        # Detection variables
        self.is_running = False
        self.detection_counter = 0
        self.confidence_threshold = 0.5
        self.last_alert_time = 0
        self.alert_cooldown = 2
        
        # Status
        self.status = {
            'running': False,
            'dog_detected': False,
            'detection_count': 0,
            'model_ready': True
        }
    
    def process_frame(self, frame_data):
        """Process a single frame from the browser"""
        try:
            # Decode base64 image
            img_data = base64.b64decode(frame_data.split(',')[1])
            img_array = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Check for dogs (class 16 in COCO dataset)
            dog_detected_this_frame = False
            dog_count = 0
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Class 16 is 'dog' in COCO dataset
                        if class_id == 16 and confidence > self.confidence_threshold:
                            dog_detected_this_frame = True
                            dog_count += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            detections.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'confidence': confidence
                            })
            
            # Handle dog detection alerts
            if dog_detected_this_frame:
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    self.trigger_alert(dog_count)
                    self.last_alert_time = current_time
            
            return {
                'dogs_detected': dog_detected_this_frame,
                'dog_count': dog_count,
                'detections': detections
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def trigger_alert(self, dog_count):
        """Trigger dog detection alert"""
        self.detection_counter += 1
        self.status['dog_detected'] = True
        self.status['detection_count'] = self.detection_counter
        
        if dog_count == 1:
            message = "DOG DETECTED!"
        else:
            message = f"{dog_count} DOGS DETECTED!"
        
        self.log_message(f"{message} (Total: {self.detection_counter})")
        
        # Emit alert to web clients
        socketio.emit('dog_alert', {
            'detected': True,
            'count': dog_count,
            'total': self.detection_counter,
            'message': message
        })
        
        print(f"{message} Found {dog_count} dog(s) in frame!")
    
    def log_message(self, message):
        """Add message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Emit log message to web clients
        socketio.emit('log_message', {'message': log_entry})
        print(message)

# Initialize detector
detector = WebDogDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a frame sent from the browser"""
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data provided'}), 400
    
    result = detector.process_frame(data['frame'])
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to process frame'}), 500

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection mode"""
    data = request.json or {}
    detector.confidence_threshold = data.get('confidence', 0.5)
    detector.alert_cooldown = data.get('cooldown', 2)
    detector.is_running = True
    detector.detection_counter = 0
    detector.status['running'] = True
    
    detector.log_message("Dog detection started! 🐕")
    return jsonify({'success': True, 'message': 'Detection started successfully'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection mode"""
    detector.is_running = False
    detector.status['running'] = False
    detector.status['dog_detected'] = False
    
    detector.log_message(f"Detection stopped! Total dogs detected: {detector.detection_counter}")
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/api/status')
def get_status():
    return jsonify(detector.status)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'confidence' in data:
        detector.confidence_threshold = data['confidence']
    if 'cooldown' in data:
        detector.alert_cooldown = data['cooldown']
    return jsonify({'success': True, 'message': 'Settings updated'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status_update', detector.status)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting Dog Detector Web App...")
    print("Camera access will be handled by your browser")
    print("Access the app at http://localhost:5001")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)