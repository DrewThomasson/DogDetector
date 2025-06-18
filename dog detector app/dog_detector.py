import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pygame
from ultralytics import YOLO
import os
import sys
from PIL import Image, ImageTk
import time
import platform

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class DogDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dog Detector - Live Camera Feed")
        self.root.geometry("1200x800")
        
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Custom sound file path - use resource_path for bundled resources
        self.sound_file = resource_path("DOGDETECTED.mp3")
        
        # Check if sound file exists
        if os.path.exists(self.sound_file):
            print(f"Custom sound file found: {self.sound_file}")
            try:
                # Test loading the sound file
                self.dog_sound = pygame.mixer.Sound(self.sound_file)
                print("Sound file loaded successfully!")
            except Exception as e:
                print(f"Error loading sound file: {e}")
                self.dog_sound = None
        else:
            print(f"Sound file not found: {self.sound_file}")
            self.dog_sound = None
        
        # Load YOLO model with bundled weights
        try:
            model_path = resource_path('yolov8n.pt')
            print(f"Loading YOLO model from: {model_path}")
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("YOLO model loaded successfully from bundled file!")
            else:
                # Fallback to download if bundled file not found
                print("Bundled model not found, attempting to download...")
                self.model = YOLO('yolov8n.pt')
                print("YOLO model downloaded and loaded successfully!")
                
        except Exception as e:
            error_msg = f"Failed to load YOLO model: {e}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            sys.exit(1)
        
        # Camera variables
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        self.video_frame = None
        
        # Dog detection variables
        self.dog_detected = False
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
        
        # Audio variables
        self.sound_enabled = True
        self.sound_volume = 0.7
        
        # GUI variables
        self.photo = None
        
        # Create GUI
        self.create_gui()
        
        # Start GUI update loop
        self.update_gui()
        
    def create_gui(self):
        # Create main container with two columns
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column - Camera feed
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera feed label
        camera_label = ttk.Label(left_frame, text="📹 Live Camera Feed", font=("Arial", 16, "bold"))
        camera_label.pack(pady=(0, 10))
        
        # Video display
        self.video_label = tk.Label(left_frame, bg="black", text="Camera feed will appear here\nClick 'Start Detection' to begin", 
                                   fg="white", font=("Arial", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right column - Controls and info
        right_frame = ttk.Frame(main_container, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Title
        title_label = ttk.Label(right_frame, text="🐕 Dog Detector", font=("Arial", 20, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Status
        self.status_label = ttk.Label(right_frame, text="Status: Not Running", font=("Arial", 12))
        self.status_label.pack(pady=(0, 10))
        
        # Detection indicator
        self.detection_label = ttk.Label(right_frame, text="No Dogs Detected", 
                                       font=("Arial", 14, "bold"), foreground="green")
        self.detection_label.pack(pady=(0, 20))
        
        # Detection counter
        self.detection_counter = 0
        self.counter_label = ttk.Label(right_frame, text="Dogs Detected: 0", font=("Arial", 12))
        self.counter_label.pack(pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(pady=(0, 20))
        
        # Start button
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                     command=self.start_detection)
        self.start_button.pack(fill=tk.X, pady=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                    command=self.stop_detection, state="disabled")
        self.stop_button.pack(fill=tk.X, pady=(0, 10))
        
        # Test sound button
        self.test_sound_button = ttk.Button(button_frame, text="Test Sound", 
                                          command=self.test_sound)
        self.test_sound_button.pack(fill=tk.X)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(right_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Alert cooldown
        ttk.Label(settings_frame, text="Alert Cooldown (sec):").pack(anchor=tk.W)
        self.cooldown_var = tk.StringVar(value="2")
        cooldown_entry = ttk.Entry(settings_frame, textvariable=self.cooldown_var, width=15)
        cooldown_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Camera selection
        ttk.Label(settings_frame, text="Camera Index:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(settings_frame, textvariable=self.camera_var, width=15)
        camera_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.confidence_var = tk.StringVar(value="0.5")
        confidence_entry = ttk.Entry(settings_frame, textvariable=self.confidence_var, width=15)
        confidence_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Sound settings
        # Sound Enable/Disable
        self.sound_var = tk.BooleanVar(value=True)
        sound_check = ttk.Checkbutton(settings_frame, text="Enable Sound Alerts", 
                                     variable=self.sound_var)
        sound_check.pack(anchor=tk.W, pady=(5, 5))
        
        # Volume control
        ttk.Label(settings_frame, text="Volume:").pack(anchor=tk.W)
        self.volume_var = tk.DoubleVar(value=0.7)
        volume_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                               variable=self.volume_var, orient="horizontal")
        volume_scale.pack(fill=tk.X, pady=(5, 10))
        
        # Sound file status
        sound_status = "✅ Custom sound loaded" if self.dog_sound else "❌ Sound file not found"
        sound_color = "green" if self.dog_sound else "red"
        self.sound_status_label = ttk.Label(settings_frame, text=sound_status, foreground=sound_color)
        self.sound_status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Sound file path display
        path_label = ttk.Label(settings_frame, text="Sound file:", font=("Arial", 8))
        path_label.pack(anchor=tk.W)
        path_display = ttk.Label(settings_frame, text=os.path.basename(self.sound_file), 
                               font=("Arial", 8), foreground="gray")
        path_display.pack(anchor=tk.W, pady=(0, 10))
        
        # Log area
        log_frame = ttk.LabelFrame(right_frame, text="Detection Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(text_frame, height=8, width=35, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def log_message(self, message):
        """Add message to the log"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        print(message)  # Also print to console
        
    def play_dog_sound(self):
        """Play the custom dog detected sound"""
        if not self.sound_var.get():  # Check if sound is enabled
            return
            
        try:
            if self.dog_sound:
                # Set volume
                volume = self.volume_var.get()
                self.dog_sound.set_volume(volume)
                
                # Play the sound
                self.dog_sound.play()
                print(f"Playing DOGDETECTED.mp3 at volume {volume:.1f}")
            else:
                # Fallback beep if custom sound not available
                self.play_fallback_beep()
                
        except Exception as e:
            print(f"Error playing sound: {e}")
            self.play_fallback_beep()
    
    def play_fallback_beep(self):
        """Play fallback beep sound if custom sound fails"""
        try:
            # Create a simple beep sound
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            sound.set_volume(self.volume_var.get())
            sound.play()
        except Exception as e:
            print(f"Could not play fallback sound: {e}")
    
    def test_sound(self):
        """Test the dog detection sound"""
        self.play_dog_sound()
        self.log_message("Testing dog detection sound...")
    
    def start_detection(self):
        """Start the dog detection"""
        try:
            self.alert_cooldown = float(self.cooldown_var.get())
            camera_index = int(self.camera_var.get())
            self.confidence_threshold = float(self.confidence_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for settings")
            return
        
        # Try to open camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.detection_counter = 0
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: Running")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.log_message("Dog detection started! 🐕")
    
    def stop_detection(self):
        """Stop the dog detection"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Not Running")
        self.detection_label.config(text="No Dogs Detected", foreground="green")
        
        # Clear video display
        self.video_label.config(image='', text="Camera feed stopped\nClick 'Start Detection' to begin", 
                               bg="black", fg="white")
        self.photo = None
        
        self.log_message(f"Detection stopped! Total dogs detected: {self.detection_counter}")
    
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
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # RED box
                            
                            # Add label with red background
                            label = f'DOG! {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                                        (x1+label_size[0], y1), (0, 0, 255), -1)  # Red background
                            cv2.putText(frame, label, (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
            
            # Handle dog detection
            if dog_detected_this_frame and not self.dog_detected:
                self.dog_detected = True
                current_time = time.time()
                
                if current_time - self.last_alert_time > self.alert_cooldown:
                    # Update GUI in main thread
                    self.root.after(0, lambda: self.trigger_alert(dog_count))
                    self.last_alert_time = current_time
                    
            elif not dog_detected_this_frame:
                if self.dog_detected:
                    self.root.after(0, self.clear_alert)
                self.dog_detected = False
            
            # Add detection info to frame
            if dog_detected_this_frame:
                cv2.putText(frame, f'DOGS DETECTED: {dog_count}', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store frame for GUI display
            self.video_frame = frame.copy()
        
        # Cleanup
        if self.cap:
            self.cap.release()
        
        # Update GUI
        self.root.after(0, self.stop_detection)
    
    def trigger_alert(self, dog_count):
        """Trigger dog detection alert"""
        self.detection_counter += 1
        self.detection_label.config(text="🚨 DOG DETECTED! 🚨", foreground="red")
        self.counter_label.config(text=f"Dogs Detected: {self.detection_counter}")
        
        # Create alert message
        if dog_count == 1:
            message = "DOG DETECTED!"
        else:
            message = f"{dog_count} DOGS DETECTED!"
        
        self.log_message(f"{message} (Total: {self.detection_counter})")
        
        # Play custom sound
        self.play_dog_sound()
        
        print(f"{message} Found {dog_count} dog(s) in frame!")  # Console output as requested
    
    def clear_alert(self):
        """Clear dog detection alert"""
        self.detection_label.config(text="No Dogs Detected", foreground="green")
    
    def update_gui(self):
        """Update the GUI with the latest video frame"""
        if self.video_frame is not None:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(self.video_frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit in GUI
            height, width = rgb_frame.shape[:2]
            max_width = 640
            max_height = 480
            
            # Calculate scaling factor
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize frame
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=self.photo, text="")
        
        # Schedule next update
        self.root.after(33, self.update_gui)  # ~30 FPS
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        welcome_msg = "Dog Detector ready! Connect a camera and click 'Start Detection'"
        self.log_message(welcome_msg)
        
        if self.dog_sound:
            self.log_message("🔊 Custom DOGDETECTED.mp3 sound loaded!")
        else:
            self.log_message("🔊 Using fallback beep sound (custom MP3 not found)")
            
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    print("Starting Dog Detector with Custom Sound Alert...")
    print("Make sure you have a webcam connected!")
    
    app = DogDetector()
    app.run()