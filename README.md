# DogDetector
python program that detects dog and yells out DOG DETECTED!


### Python install (python3.10 env)
`pip install -r requirements.txt`
### Run python app with
`python dog_detector.py`

### build with pyinstaller

`pip install pyinstaller`

#### For Mac

```bash
# Build the executable
pyinstaller --onefile --windowed --add-data "DOGDETECTED.mp3:." --add-data "yolov8n.pt:." --add-data "requirements.txt:." --name "DogDetector" dog_detector.py

# Your executable will be in the dist/ folder
# Run: ./dist/DogDetector
```

#### For Windows

```bash
# Build the executable
pyinstaller --onefile --windowed --add-data "DOGDETECTED.mp3;." --add-data "yolov8n.pt;." --add-data "requirements.txt;." --name "DogDetector" dog_detector.py

# Your executable will be in the dist/ folder
# Run: dist\DogDetector.exe
```

#### For Linux

```bash
# Build the executable
pyinstaller --onefile --add-data "DOGDETECTED.mp3:." --add-data "yolov8n.pt:." --add-data "requirements.txt:." --name "DogDetector" dog_detector.py

# Your executable will be in the dist/ folder
# Run: ./dist/DogDetector
```
