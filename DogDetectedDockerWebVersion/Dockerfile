FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libasound2-dev \
    pulseaudio \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO model during build (this will be baked into the image)
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO model downloaded successfully')"

# Verify the model file exists and move it to a predictable location
RUN find /root/.ultralytics -name "yolov8n.pt" -exec cp {} /app/yolov8n.pt \; || \
    find /home -name "yolov8n.pt" -exec cp {} /app/yolov8n.pt \; || \
    find / -name "yolov8n.pt" 2>/dev/null | head -1 | xargs -I {} cp {} /app/yolov8n.pt

# Verify the model is in place
RUN ls -la /app/yolov8n.pt && echo "YOLO model successfully baked into image"

# Copy application files
COPY . .

# Create directories for sounds
RUN mkdir -p sounds

# Set permissions
RUN chmod +x /app/yolov8n.pt

# Expose port
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV YOLO_MODEL_PATH=/app/yolov8n.pt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/status || exit 1

# Run the application
CMD ["python", "app.py"]