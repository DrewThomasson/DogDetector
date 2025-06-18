# DogDetector
python program that detects dog and yells out DOG DETECTED!

## DEMO

https://github.com/user-attachments/assets/6eac7317-c868-42d0-8192-6e4066685f41

### Python install (python3.10 env)
`pip install -r requirements.txt`
### Run python app with
`python dog_detector.py`

### build with pyinstaller

`pip install pyinstaller`

#### For Mac ARM

```bash
# Build the executable
pyinstaller --onedir --windowed --add-data "DOGDETECTED.mp3:." --add-data "yolov8n.pt:." --add-data "Info.plist:." --hidden-import=ultralytics --hidden-import=PIL --hidden-import=PIL._tkinter_finder --collect-all ultralytics --collect-all torch --collect-all torchvision --name "DogDetector" --osx-bundle-identifier "com.dogdetector.app" dog_detector.py

# Your executable will be in the dist/ folder
# Run: ./dist/DogDetector
```

### To stop mac from quarenteening the files in DogDetector folder from the binary zip

```bash
xattr -rd com.apple.quarantine DogDetector
```


### Docker Run command (DOcker is not working for some reason)

```bash
docker run -d \
  --name dog-detector \
  -p 5001:5001 \
  --device /dev/video0:/dev/video0 \
  --privileged \
  athomasson2/dogdetector:latest
```
