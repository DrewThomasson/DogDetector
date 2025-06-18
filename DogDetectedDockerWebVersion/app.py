from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import threading
import pygame
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

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class WebDogDetector:
    def __init__(self):
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Custom sound file path
        self.sound_file = resource_path("DOGDETECTED.mp3")
        
        # Check if sound file exists
        if os.path.exists(self.sound_file):
            print(f"Custom sound file found: {self.sound_file}")
            try:
                self.dog_sound = pygame.mixer.Sound(self.sound_file)
                print("Sound file loaded successfully!")
            except Exception as e:
                print(f"Error loading sound file: {e}")
                self.dog_sound = None
        else:
            print(f"Sound file not found: {self.sound_file}")
            self.dog_sound = None
        
        # Load YOLO model - check for baked-in model first
        try:
            # Check if model is baked into the image (Docker)
            baked_model_path = '/app/yolov8n.pt'
            local_model_path = resource_path('yolov8n.pt')
            env_model_path = os.environ.get('YOLO_MODEL_PATH')
            
            model_path = None
            
            # Priority order: environment variable, baked-in, local, download
            if env_model_path and os.path.exists(env_model_path):
                model_path = env_model_path
                print(f"Using YOLO model from environment: {model_path}")
            elif os.path.exists(baked_model_path):
                model_path = baked_model_path
                print(f"Using baked-in YOLO model: {model_path}")
            elif os.path.exists(local_model_path):
                model_path = local_model_path
                print(f"Using local YOLO model: {model_path}")
            
            if model_path:
                self.model = YOLO(model_path)
                print("YOLO model loaded successfully from pre-downloaded file!")
            else:
                # Fallback to download if no pre-existing model found
                print("No pre-existing model found, downloading...")
                self.model = YOLO('yolov8n.pt')
                print("YOLO model downloaded and loaded successfully!")
                
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            sys.exit(1)
        
        # Camera variables
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        
        # Detection variables
        self.dog_detected = False
        self.last_alert_time = 0
        self.alert_cooldown = 2
        self.detection_counter = 0
        self.confidence_threshold = 0.5
        self.camera_index = 0
        
        # Audio variables
        self.sound_enabled = True
        self.sound_volume = 0.7
        
        # Status
        self.status = {
            'running': False,
            'dog_detected': False,
            'detection_count': 0,
            'sound_available': self.dog_sound is not None,
            'model_ready': True
        }
    
    def start_detection(self, camera_index=0, confidence=0.5, cooldown=2):
        """Start dog detection"""
        if self.is_running:
            return False, "Detection already running"
        
        self.camera_index = camera_index
        self.confidence_threshold = confidence
        self.alert_cooldown = cooldown
        
        # Try to open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False, f"Could not open camera {camera_index}"
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.detection_counter = 0
        self.status['running'] = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.log_message("Dog detection started! 🐕")
        return True, "Detection started successfully"
    
    def stop_detection(self):
        """Stop dog detection"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.status['running'] = False
        self.status['dog_detected'] = False
        
        self.log_message(f"Detection stopped! Total dogs detected: {self.detection_counter}")
        return True, "Detection stopped"
    
    def detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Check for dogs (class 16 in COCO dataset)
            dog_detected_this_frame = False
            dog_count = 0
            
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
                            
                            # Draw RED bounding box for dogs
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Add label with red background
                            label = f'DOG! {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                                        (x1+label_size[0], y1), (0, 0, 255), -1)
                            cv2.putText(frame, label, (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Handle dog detection
            if dog_detected_this_frame and not self.dog_detected:
                self.dog_detected = True
                current_time = time.time()
                
                if current_time - self.last_alert_time > self.alert_cooldown:
                    self.trigger_alert(dog_count)
                    self.last_alert_time = current_time
                    
            elif not dog_detected_this_frame:
                if self.dog_detected:
                    self.clear_alert()
                self.dog_detected = False
            
            # Add detection info to frame
            if dog_detected_this_frame:
                cv2.putText(frame, f'DOGS DETECTED: {dog_count}', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Encode frame for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit frame to connected clients
            socketio.emit('video_frame', {'frame': frame_base64})
            
            time.sleep(0.033)  # ~30 FPS
        
        # Cleanup
        if self.cap:
            self.cap.release()
    
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
        
        # Play sound if enabled
        if self.sound_enabled:
            self.play_dog_sound()
        
        # Emit alert to web clients
        socketio.emit('dog_alert', {
            'detected': True,
            'count': dog_count,
            'total': self.detection_counter,
            'message': message
        })
        
        print(f"{message} Found {dog_count} dog(s) in frame!")
    
    def clear_alert(self):
        """Clear dog detection alert"""
        self.status['dog_detected'] = False
        
        socketio.emit('dog_alert', {
            'detected': False,
            'count': 0,
            'total': self.detection_counter,
            'message': "No Dogs Detected"
        })
    
    def play_dog_sound(self):
        """Play the custom dog detected sound"""
        if not self.sound_enabled:
            return
            
        try:
            if self.dog_sound:
                self.dog_sound.set_volume(self.sound_volume)
                self.dog_sound.play()
                print(f"Playing DOGDETECTED.mp3 at volume {self.sound_volume:.1f}")
            else:
                self.play_fallback_beep()
        except Exception as e:
            print(f"Error playing sound: {e}")
            self.play_fallback_beep()
    
    def play_fallback_beep(self):
        """Play fallback beep sound"""
        try:
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            sound.set_volume(self.sound_volume)
            sound.play()
        except Exception as e:
            print(f"Could not play fallback sound: {e}")
    
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

@app.route('/api/start', methods=['POST'])
def start_detection():
    data = request.json or {}
    camera_index = data.get('camera_index', 0)
    confidence = data.get('confidence', 0.5)
    cooldown = data.get('cooldown', 2)
    
    success, message = detector.start_detection(camera_index, confidence, cooldown)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    success, message = detector.stop_detection()
    return jsonify({'success': success, 'message': message})

@app.route('/api/status')
def get_status():
    return jsonify(detector.status)

@app.route('/api/test_sound', methods=['POST'])
def test_sound():
    detector.play_dog_sound()
    detector.log_message("Testing dog detection sound...")
    return jsonify({'success': True, 'message': 'Sound test triggered'})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'sound_enabled' in data:
        detector.sound_enabled = data['sound_enabled']
    if 'sound_volume' in data:
        detector.sound_volume = data['sound_volume']
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
    print("Make sure you have a webcam connected!")
    print("Access the app at http://localhost:5001")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)