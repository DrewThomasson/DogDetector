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
import sqlite3
from datetime import datetime
import socket
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from collections import defaultdict

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class DatabaseManager:
    def __init__(self, db_path="dog_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database and create tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    detection_type TEXT NOT NULL,
                    dog_count INTEGER DEFAULT 0,
                    person_count INTEGER DEFAULT 0,
                    confidence_dog REAL DEFAULT 0.0,
                    confidence_person REAL DEFAULT 0.0,
                    location TEXT,
                    device_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"Database initialized successfully at: {self.db_path}")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def insert_detection(self, detection_type, dog_count=0, person_count=0, 
                        confidence_dog=0.0, confidence_person=0.0, location="Unknown"):
        """Insert a detection record into the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get device name
            device_name = socket.gethostname()
            
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, detection_type, dog_count, person_count, 
                 confidence_dog, confidence_person, location, device_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                detection_type,
                dog_count,
                person_count,
                confidence_dog,
                confidence_person,
                location,
                device_name
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
            
            print(f"Detection recorded to database: ID {record_id}, Type: {detection_type}")
            return record_id
            
        except Exception as e:
            print(f"Error inserting detection: {e}")
            return None
    
    def get_recent_detections(self, limit=10):
        """Get recent detections from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            records = cursor.fetchall()
            conn.close()
            
            return records
            
        except Exception as e:
            print(f"Error fetching detections: {e}")
            return []
    
    def get_heatmap_data(self, detection_type=None, days_back=30):
        """Get data for time-of-day heatmap"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Base query
            query = '''
                SELECT timestamp, detection_type
                FROM detections 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days_back)
            
            if detection_type:
                query += f" AND detection_type = '{detection_type}'"
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query)
            records = cursor.fetchall()
            conn.close()
            
            return records
            
        except Exception as e:
            print(f"Error fetching heatmap data: {e}")
            return []

class HeatmapWindow:
    def __init__(self, parent, db_manager):
        self.parent = parent
        self.db_manager = db_manager
        self.window = tk.Toplevel(parent)
        self.window.title("Detection Time Heatmaps")
        self.window.geometry("1200x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs for each detection type
        self.create_heatmap_tab("Dog Only", "dog_only")
        self.create_heatmap_tab("Person Only", "person_only")
        self.create_heatmap_tab("Dog & Person", "dog_and_person")
        self.create_heatmap_tab("All Detections", None)
        
        # Controls frame
        controls_frame = ttk.Frame(self.window)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Days selection
        ttk.Label(controls_frame, text="Days to include:").pack(side=tk.LEFT, padx=(0, 5))
        self.days_var = tk.StringVar(value="30")
        days_combo = ttk.Combobox(controls_frame, textvariable=self.days_var, 
                                 values=["7", "14", "30", "60", "90"], width=5)
        days_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh button
        refresh_btn = ttk.Button(controls_frame, text="Refresh Heatmaps", 
                               command=self.refresh_all_heatmaps)
        refresh_btn.pack(side=tk.LEFT, padx=10)
        
        # Initial load
        self.refresh_all_heatmaps()
    
    def create_heatmap_tab(self, tab_name, detection_type):
        """Create a tab with heatmap for specific detection type"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=tab_name)
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 6), dpi=100)
        
        # Store reference to figure and detection type
        setattr(self, f"fig_{detection_type or 'all'}", fig)
        setattr(self, f"detection_type_{detection_type or 'all'}", detection_type)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, tab_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store canvas reference
        setattr(self, f"canvas_{detection_type or 'all'}", canvas)
    
    def create_heatmap(self, detection_type, days_back=30):
        """Create heatmap for specific detection type"""
        try:
            # Get data from database
            records = self.db_manager.get_heatmap_data(detection_type, days_back)
            
            if not records:
                return self.create_empty_heatmap(f"No data for {detection_type or 'all detections'}")
            
            # Process data into hour/day matrix
            hour_day_matrix = np.zeros((7, 24))  # 7 days of week, 24 hours
            hour_counts = defaultdict(int)
            day_counts = defaultdict(int)
            
            for record in records:
                timestamp_str = record[0]
                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    # Fallback parsing
                    dt = datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                
                hour = dt.hour
                day_of_week = dt.weekday()  # 0 = Monday, 6 = Sunday
                
                hour_day_matrix[day_of_week][hour] += 1
                hour_counts[hour] += 1
                day_counts[day_of_week] += 1
            
            return hour_day_matrix, hour_counts, day_counts
            
        except Exception as e:
            print(f"Error creating heatmap data: {e}")
            return self.create_empty_heatmap(f"Error loading data: {e}")
    
    def create_empty_heatmap(self, message):
        """Create empty heatmap with message"""
        return np.zeros((7, 24)), {}, {}
    
    def plot_heatmap(self, detection_type, days_back=30):
        """Plot heatmap for detection type"""
        try:
            # Get figure
            fig = getattr(self, f"fig_{detection_type or 'all'}")
            fig.clear()
            
            # Create heatmap data
            hour_day_matrix, hour_counts, day_counts = self.create_heatmap(detection_type, days_back)
            
            # Create subplots
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], 
                                hspace=0.3, wspace=0.3)
            
            # Main heatmap
            ax_main = fig.add_subplot(gs[0, 0])
            
            # Day labels
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            hour_labels = [f'{h:02d}:00' for h in range(24)]
            
            # Create heatmap
            im = ax_main.imshow(hour_day_matrix, cmap='YlOrRd', aspect='auto', 
                              interpolation='nearest')
            
            # Set labels
            ax_main.set_xticks(range(24))
            ax_main.set_xticklabels([f'{h:02d}' for h in range(24)])
            ax_main.set_yticks(range(7))
            ax_main.set_yticklabels(day_labels)
            ax_main.set_xlabel('Hour of Day')
            ax_main.set_ylabel('Day of Week')
            
            # Title
            title = f"Detection Heatmap - {detection_type.replace('_', ' ').title() if detection_type else 'All Detections'}"
            title += f" (Last {days_back} days)"
            ax_main.set_title(title)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax_main, shrink=0.8)
            cbar.set_label('Detection Count')
            
            # Add text annotations for non-zero values
            for i in range(7):
                for j in range(24):
                    if hour_day_matrix[i, j] > 0:
                        text_color = 'white' if hour_day_matrix[i, j] > hour_day_matrix.max()/2 else 'black'
                        ax_main.text(j, i, f'{int(hour_day_matrix[i, j])}', 
                                   ha='center', va='center', color=text_color, fontsize=8)
            
            # Hour distribution (right subplot)
            ax_hour = fig.add_subplot(gs[0, 1])
            if hour_counts:
                hours = list(range(24))
                counts = [hour_counts.get(h, 0) for h in hours]
                ax_hour.barh(hours, counts, color='orange', alpha=0.7)
                ax_hour.set_ylabel('Hour of Day')
                ax_hour.set_xlabel('Count')
                ax_hour.set_title('Hourly Distribution')
                ax_hour.set_yticks(range(0, 24, 4))
                ax_hour.grid(True, alpha=0.3)
            
            # Day distribution (bottom subplot)
            ax_day = fig.add_subplot(gs[1, 0])
            if day_counts:
                days = list(range(7))
                counts = [day_counts.get(d, 0) for d in days]
                ax_day.bar(days, counts, color='skyblue', alpha=0.7)
                ax_day.set_xlabel('Day of Week')
                ax_day.set_ylabel('Count')
                ax_day.set_title('Daily Distribution')
                ax_day.set_xticks(range(7))
                ax_day.set_xticklabels(day_labels)
                ax_day.grid(True, alpha=0.3)
            
            # Statistics (bottom right)
            ax_stats = fig.add_subplot(gs[1, 1])
            ax_stats.axis('off')
            
            total_detections = int(hour_day_matrix.sum())
            if total_detections > 0:
                peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
                peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else 0
                peak_day_name = day_labels[peak_day]
                
                stats_text = f"""Statistics:
Total: {total_detections}
Peak Hour: {peak_hour:02d}:00
Peak Day: {peak_day_name}
Avg/Day: {total_detections/7:.1f}"""
            else:
                stats_text = "No detections\nin selected period"
            
            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # Refresh canvas
            canvas = getattr(self, f"canvas_{detection_type or 'all'}")
            canvas.draw()
            
        except Exception as e:
            print(f"Error plotting heatmap: {e}")
    
    def refresh_all_heatmaps(self):
        """Refresh all heatmaps"""
        try:
            days_back = int(self.days_var.get())
        except:
            days_back = 30
        
        # Plot each heatmap
        self.plot_heatmap("dog_only", days_back)
        self.plot_heatmap("person_only", days_back)
        self.plot_heatmap("dog_and_person", days_back)
        self.plot_heatmap(None, days_back)  # All detections

class DogDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dog & Person Detector - Live Camera Feed with Database & Heatmaps")
        self.root.geometry("1300x900")
        
        # Initialize database
        self.db_manager = DatabaseManager()
        
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
        
        # Detection variables
        self.dog_detected = False
        self.person_detected = False
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
        self.last_detection_type = None
        
        # Audio variables
        self.sound_enabled = True
        self.sound_volume = 0.7
        
        # GUI variables
        self.photo = None
        
        # Detection counters
        self.dog_only_count = 0
        self.person_only_count = 0
        self.dog_and_person_count = 0
        self.total_detections = 0
        
        # Location setting
        self.location = "Home"  # Default location
        
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
        right_frame = ttk.Frame(main_container, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Title
        title_label = ttk.Label(right_frame, text="🐕👤 Dog & Person Detector", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Status
        self.status_label = ttk.Label(right_frame, text="Status: Not Running", font=("Arial", 12))
        self.status_label.pack(pady=(0, 10))
        
        # Detection indicator
        self.detection_label = ttk.Label(right_frame, text="No Detection", 
                                       font=("Arial", 14, "bold"), foreground="green")
        self.detection_label.pack(pady=(0, 20))
        
        # Detection counters frame
        counters_frame = ttk.LabelFrame(right_frame, text="Detection Statistics", padding="10")
        counters_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.dog_only_label = ttk.Label(counters_frame, text="Dogs Only: 0", font=("Arial", 10))
        self.dog_only_label.pack(anchor=tk.W)
        
        self.person_only_label = ttk.Label(counters_frame, text="Persons Only: 0", font=("Arial", 10))
        self.person_only_label.pack(anchor=tk.W)
        
        self.both_label = ttk.Label(counters_frame, text="Dog & Person: 0", font=("Arial", 10))
        self.both_label.pack(anchor=tk.W)
        
        self.total_label = ttk.Label(counters_frame, text="Total Events: 0", font=("Arial", 10, "bold"))
        self.total_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Buttons frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(pady=(0, 10))
        
        # Start button
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                     command=self.start_detection)
        self.start_button.pack(fill=tk.X, pady=(0, 5))
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                    command=self.stop_detection, state="disabled")
        self.stop_button.pack(fill=tk.X, pady=(0, 5))
        
        # Test sound button
        self.test_sound_button = ttk.Button(button_frame, text="Test Sound", 
                                          command=self.test_sound)
        self.test_sound_button.pack(fill=tk.X, pady=(0, 5))
        
        # View database button
        self.view_db_button = ttk.Button(button_frame, text="View Recent Records", 
                                       command=self.view_database)
        self.view_db_button.pack(fill=tk.X, pady=(0, 5))
        
        # View heatmaps button
        self.view_heatmaps_button = ttk.Button(button_frame, text="📊 View Heatmaps", 
                                             command=self.view_heatmaps)
        self.view_heatmaps_button.pack(fill=tk.X)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(right_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Location setting
        ttk.Label(settings_frame, text="Location:").pack(anchor=tk.W)
        self.location_var = tk.StringVar(value="Home")
        location_entry = ttk.Entry(settings_frame, textvariable=self.location_var, width=20)
        location_entry.pack(fill=tk.X, pady=(5, 10))
        
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
        
        # Database info
        db_frame = ttk.LabelFrame(right_frame, text="Database Info", padding="5")
        db_frame.pack(fill=tk.X, pady=(0, 10))
        
        db_path_label = ttk.Label(db_frame, text=f"DB: {self.db_manager.db_path}", 
                                font=("Arial", 8), foreground="gray")
        db_path_label.pack(anchor=tk.W)
        
        # Log area
        log_frame = ttk.LabelFrame(right_frame, text="Detection Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(text_frame, height=6, width=40, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def log_message(self, message):
        """Add message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        print(message)  # Also print to console
    
    def view_heatmaps(self):
        """Open heatmap window"""
        HeatmapWindow(self.root, self.db_manager)
    
    def view_database(self):
        """Show recent database records in a new window"""
        db_window = tk.Toplevel(self.root)
        db_window.title("Recent Detection Records")
        db_window.geometry("900x600")
        
        # Create treeview for displaying records
        tree_frame = ttk.Frame(db_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview with scrollbars
        columns = ("ID", "Timestamp", "Type", "Dogs", "Persons", "Dog Conf", "Person Conf", "Location")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        tree.heading("ID", text="ID")
        tree.heading("Timestamp", text="Timestamp")
        tree.heading("Type", text="Detection Type")
        tree.heading("Dogs", text="Dogs")
        tree.heading("Persons", text="Persons")
        tree.heading("Dog Conf", text="Dog Conf")
        tree.heading("Person Conf", text="Person Conf")
        tree.heading("Location", text="Location")
        
        # Configure column widths
        tree.column("ID", width=50)
        tree.column("Timestamp", width=150)
        tree.column("Type", width=120)
        tree.column("Dogs", width=60)
        tree.column("Persons", width=70)
        tree.column("Dog Conf", width=80)
        tree.column("Person Conf", width=90)
        tree.column("Location", width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load data
        records = self.db_manager.get_recent_detections(100)  # Get last 100 records
        
        for record in records:
            # Format the record for display
            formatted_record = (
                record[0],  # ID
                record[1],  # Timestamp
                record[2],  # Detection type
                record[3],  # Dog count
                record[4],  # Person count
                f"{record[5]:.2f}" if record[5] > 0 else "0.00",  # Dog confidence
                f"{record[6]:.2f}" if record[6] > 0 else "0.00",  # Person confidence
                record[7] or "Unknown"  # Location
            )
            tree.insert("", "end", values=formatted_record)
        
        # Refresh button
        refresh_btn = ttk.Button(db_window, text="Refresh", 
                               command=lambda: self.refresh_database_view(tree))
        refresh_btn.pack(pady=5)
        
    def refresh_database_view(self, tree):
        """Refresh the database view"""
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Reload data
        records = self.db_manager.get_recent_detections(100)
        for record in records:
            formatted_record = (
                record[0], record[1], record[2], record[3], record[4],
                f"{record[5]:.2f}" if record[5] > 0 else "0.00",
                f"{record[6]:.2f}" if record[6] > 0 else "0.00",
                record[7] or "Unknown"
            )
            tree.insert("", "end", values=formatted_record)
        
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
        """Start the detection"""
        try:
            self.alert_cooldown = float(self.cooldown_var.get())
            camera_index = int(self.camera_var.get())
            self.confidence_threshold = float(self.confidence_var.get())
            self.location = self.location_var.get() or "Unknown"
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
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: Running")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.log_message(f"Detection started at location: {self.location} 🐕👤")
    
    def stop_detection(self):
        """Stop the detection"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Not Running")
        self.detection_label.config(text="No Detection", foreground="green")
        
        # Clear video display
        self.video_label.config(image='', text="Camera feed stopped\nClick 'Start Detection' to begin", 
                               bg="black", fg="white")
        self.photo = None
        
        self.log_message(f"Detection stopped! Total events: {self.total_detections}")
    
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
            
            # Check for dogs and persons
            dogs_detected = []
            persons_detected = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Class 16 is 'dog' in COCO dataset
                            if class_id == 16:
                                dogs_detected.append({
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2)
                                })
                                
                                # Draw RED bounding box for dogs
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                label = f'DOG! {confidence:.2f}'
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                                            (x1+label_size[0], y1), (0, 0, 255), -1)
                                cv2.putText(frame, label, (x1, y1-5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Class 0 is 'person' in COCO dataset
                            elif class_id == 0:
                                persons_detected.append({
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2)
                                })
                                
                                # Draw BLUE bounding box for persons
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                                label = f'PERSON {confidence:.2f}'
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                                            (x1+label_size[0], y1), (255, 0, 0), -1)
                                cv2.putText(frame, label, (x1, y1-5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Determine detection type
            dog_count = len(dogs_detected)
            person_count = len(persons_detected)
            
            current_detection_type = None
            if dog_count > 0 and person_count > 0:
                current_detection_type = "dog_and_person"
            elif dog_count > 0:
                current_detection_type = "dog_only"
            elif person_count > 0:
                current_detection_type = "person_only"
            
            # Handle detection events
            current_time = time.time()
            if current_detection_type and current_detection_type != self.last_detection_type:
                if current_time - self.last_alert_time > self.alert_cooldown:
                    # Calculate confidence values
                    max_dog_confidence = max([d['confidence'] for d in dogs_detected]) if dogs_detected else 0.0
                    max_person_confidence = max([p['confidence'] for p in persons_detected]) if persons_detected else 0.0
                    
                    # Record to database
                    record_id = self.db_manager.insert_detection(
                        detection_type=current_detection_type,
                        dog_count=dog_count,
                        person_count=person_count,
                        confidence_dog=max_dog_confidence,
                        confidence_person=max_person_confidence,
                        location=self.location
                    )
                    
                    # Update GUI in main thread
                    self.root.after(0, lambda: self.trigger_alert(current_detection_type, dog_count, person_count))
                    self.last_alert_time = current_time
                    
                    # Play sound if dog is detected (dog_only or dog_and_person)
                    if "dog" in current_detection_type:
                        self.play_dog_sound()
            
            # Update last detection type
            self.last_detection_type = current_detection_type
            
            # Clear alert if nothing detected
            if not current_detection_type:
                if self.dog_detected or self.person_detected:
                    self.root.after(0, self.clear_alert)
                self.dog_detected = False
                self.person_detected = False
            else:
                self.dog_detected = dog_count > 0
                self.person_detected = person_count > 0
            
            # Add detection info to frame
            if current_detection_type:
                status_text = f'DETECTED: {dog_count} Dogs, {person_count} Persons'
                cv2.putText(frame, status_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
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
    
    def trigger_alert(self, detection_type, dog_count, person_count):
        """Trigger detection alert"""
        self.total_detections += 1
        
        # Update counters based on detection type
        if detection_type == "dog_only":
            self.dog_only_count += 1
            self.detection_label.config(text="🚨 DOG DETECTED! 🚨", foreground="red")
            message = f"DOG DETECTED! ({dog_count} dog{'s' if dog_count > 1 else ''})"
            
        elif detection_type == "person_only":
            self.person_only_count += 1
            self.detection_label.config(text="👤 PERSON DETECTED", foreground="blue")
            message = f"PERSON DETECTED! ({person_count} person{'s' if person_count > 1 else ''})"
            
        elif detection_type == "dog_and_person":
            self.dog_and_person_count += 1
            self.detection_label.config(text="🚨 DOG & PERSON! 🚨", foreground="purple")
            message = f"DOG & PERSON DETECTED! ({dog_count} dog{'s' if dog_count > 1 else ''}, {person_count} person{'s' if person_count > 1 else ''})"
        
        # Update counter labels
        self.dog_only_label.config(text=f"Dogs Only: {self.dog_only_count}")
        self.person_only_label.config(text=f"Persons Only: {self.person_only_count}")
        self.both_label.config(text=f"Dog & Person: {self.dog_and_person_count}")
        self.total_label.config(text=f"Total Events: {self.total_detections}")
        
        self.log_message(f"{message} - Recorded to DB")
        print(f"{message}")  # Console output
    
    def clear_alert(self):
        """Clear detection alert"""
        self.detection_label.config(text="No Detection", foreground="green")
    
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
        welcome_msg = "Dog & Person Detector with Heatmaps ready! Connect a camera and click 'Start Detection'"
        self.log_message(welcome_msg)
        
        if self.dog_sound:
            self.log_message("🔊 Custom DOGDETECTED.mp3 sound loaded!")
        else:
            self.log_message("🔊 Using fallback beep sound (custom MP3 not found)")
        
        self.log_message(f"Database initialized: {self.db_manager.db_path}")
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    print("Starting Dog & Person Detector with Database Recording and Heatmaps...")
    print("Make sure you have a webcam connected!")
    
    app = DogDetector()
    app.run()