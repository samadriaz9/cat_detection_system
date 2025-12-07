"""
Flask Web Application for Cat Detection with Relay Control
Access via browser at http://<raspberry-pi-ip>:5000
"""

from ultralytics import YOLO
import cv2
import time
import RPi.GPIO as GPIO
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify
import threading
import queue
import base64
import json
import os
import numpy as np
from picamera2 import Picamera2

MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.25
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RELAY_PIN = 18
PULSE_DURATION = 0.5
COOLDOWN = 2  # Trigger once every 2 seconds max

# Global variables for sharing between threads
frame_queue = queue.Queue(maxsize=2)
snapshot_frame = None
snapshot_lock = threading.Lock()
detection_status = {"detected": False, "last_trigger": None, "count": 0}
status_lock = threading.Lock()
last_trigger_time = 0
polygon_points = []  # Store polygon coordinates
polygon_lock = threading.Lock()
drawing_mode = False  # Flag to pause video during drawing
drawing_lock = threading.Lock()
POLYGON_SAVE_FILE = "polygon_coordinates.json"

app = Flask(__name__)


def load_polygon():
    """Load polygon coordinates from file"""
    global polygon_points
    if os.path.exists(POLYGON_SAVE_FILE):
        try:
            with open(POLYGON_SAVE_FILE, 'r') as f:
                data = json.load(f)
                with polygon_lock:
                    polygon_points = data.get('points', [])
                print(f"Loaded polygon with {len(polygon_points)} points")
                return True
        except Exception as e:
            print(f"Error loading polygon: {e}")
    return False


def save_polygon():
    """Save polygon coordinates to file"""
    try:
        with polygon_lock:
            points = polygon_points.copy()
        with open(POLYGON_SAVE_FILE, 'w') as f:
            json.dump({'points': points}, f)
        print(f"Saved polygon with {len(points)} points")
        return True
    except Exception as e:
        print(f"Error saving polygon: {e}")
        return False


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    if len(polygon) < 3:
        return True  # No polygon defined, allow all detections
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def pulse_relay():
    """Pulse relay using GPIO-reset method"""
    print("Relay: ON (LOW)")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Relay ON (active LOW)
    time.sleep(PULSE_DURATION)

    print("Relay: OFF (HIGH)")
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Relay OFF
    GPIO.cleanup()  # Reset GPIO completely


def detection_loop():
    """Main detection loop running in background thread"""
    global last_trigger_time, detection_status, snapshot_frame

    model = YOLO(MODEL_PATH)
    
    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
    picam2.configure(config)
    picam2.start()

    print("Detection loop started...")

    try:
        while True:
            # Check if we're in drawing mode - capture snapshot and pause
            with drawing_lock:
                is_drawing = drawing_mode
            
            if is_drawing:
                # Capture snapshot frame when drawing starts
                try:
                    frame = picam2.capture_array()
                    if frame is not None:
                        # Convert RGB to BGR for OpenCV (picamera2 returns RGB)
                        if len(frame.shape) == 3 and frame.shape[2] >= 3:
                            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
                    else:
                        time.sleep(0.1)
                        continue
                    
                    with snapshot_lock:
                        snapshot_frame = frame.copy()
                    
                    # Draw polygon on snapshot if exists
                    with polygon_lock:
                        poly_points = polygon_points.copy()
                    
                    if len(poly_points) > 0:
                        pts = np.array(poly_points, np.int32)
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                        # Note: fillPoly with alpha doesn't work directly, use overlay
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
                    # Put snapshot in queue
                    if not frame_queue.full():
                        try:
                            frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass
                    
                    time.sleep(0.1)  # Slower update during drawing
                    continue
                except Exception as e:
                    print(f"Error capturing snapshot: {e}")
                    time.sleep(0.1)
                    continue
            
            try:
                frame = picam2.capture_array()
                if frame is None:
                    time.sleep(0.03)
                    continue
                
                # Convert RGB to BGR for OpenCV (picamera2 returns RGB)
                if len(frame.shape) == 3 and frame.shape[2] >= 3:
                    frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.03)
                continue

            # Run YOLO detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            detections = results[0].boxes
            annotated = results[0].plot()

            # Get polygon for filtering
            with polygon_lock:
                poly_points = polygon_points.copy()
            
            # Filter detections within polygon
            valid_detections = []
            if len(poly_points) > 0:
                # Draw polygon on annotated frame
                pts = np.array(poly_points, np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                # Draw semi-transparent fill
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                
                # Check each detection center point
                for box in detections:
                    # Get center point of bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    if point_in_polygon((center_x, center_y), poly_points):
                        valid_detections.append(box)
            else:
                # No polygon defined, allow all detections
                valid_detections = list(detections)

            # Update detection status
            detected = len(valid_detections) > 0
            with status_lock:
                detection_status["detected"] = detected
                if detected:
                    detection_status["count"] += 1

            # Relay Logic
            if detected:
                now = time.time()
                if now - last_trigger_time > COOLDOWN:
                    print("CAT DETECTED! Triggering relay")
                    pulse_relay()
                    last_trigger_time = now
                    with status_lock:
                        detection_status["last_trigger"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Put frame in queue (non-blocking, drop old frames)
            if not frame_queue.full():
                try:
                    frame_queue.put_nowait(annotated)
                except queue.Full:
                    pass

            time.sleep(0.03)  # ~30 FPS

    except Exception as e:
        print(f"Error in detection loop: {e}")
    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception as e:
            print(f"Error closing camera: {e}")
        # GPIO.cleanup()
        print("Camera released, GPIO cleaned.")


def generate_frames():
    """Generator function for video streaming"""
    while True:
        try:
            # Get frame from queue (blocking with timeout)
            frame = frame_queue.get(timeout=1.0)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # Send a placeholder frame if queue is empty
            continue
        except Exception as e:
            print(f"Error generating frame: {e}")
            break


@app.route('/get_snapshot')
def get_snapshot():
    """Get current frame snapshot for polygon drawing"""
    with snapshot_lock:
        if snapshot_frame is not None:
            ret, buffer = cv2.imencode('.jpg', snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                frame_bytes = buffer.tobytes()
                return Response(frame_bytes, mimetype='image/jpeg')
    return Response(status=404)


@app.route('/start_drawing', methods=['POST'])
def start_drawing():
    """Start drawing mode - pause video stream"""
    global drawing_mode
    with drawing_lock:
        drawing_mode = True
    return jsonify({"status": "drawing_started"})


@app.route('/stop_drawing', methods=['POST'])
def stop_drawing():
    """Stop drawing mode - resume video stream"""
    global drawing_mode
    with drawing_lock:
        drawing_mode = False
    return jsonify({"status": "drawing_stopped"})


@app.route('/save_polygon', methods=['POST'])
def save_polygon_endpoint():
    """Save polygon coordinates from client"""
    global polygon_points
    try:
        data = request.get_json()
        points = data.get('points', [])
        
        with polygon_lock:
            polygon_points = points
        
        if save_polygon():
            return jsonify({"status": "success", "message": f"Polygon saved with {len(points)} points"})
        else:
            return jsonify({"status": "error", "message": "Failed to save polygon"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/load_polygon', methods=['GET'])
def load_polygon_endpoint():
    """Load polygon coordinates"""
    with polygon_lock:
        points = polygon_points.copy()
    return jsonify({"points": points})


@app.route('/remove_polygon', methods=['POST'])
def remove_polygon():
    """Remove polygon"""
    global polygon_points
    with polygon_lock:
        polygon_points = []
    
    # Delete save file if exists
    if os.path.exists(POLYGON_SAVE_FILE):
        try:
            os.remove(POLYGON_SAVE_FILE)
        except:
            pass
    
    return jsonify({"status": "success", "message": "Polygon removed"})


@app.route('/')
def index():
    """Main page with video stream"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cat Detection System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: #ffffff;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #4CAF50;
            }
            .status-panel {
                background-color: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
            }
            .status-item {
                text-align: center;
                padding: 10px;
            }
            .status-label {
                font-size: 14px;
                color: #888;
                margin-bottom: 5px;
            }
            .status-value {
                font-size: 24px;
                font-weight: bold;
            }
            .detected {
                color: #ff4444;
            }
            .not-detected {
                color: #4CAF50;
            }
            .video-container {
                background-color: #000;
                border-radius: 10px;
                overflow: hidden;
                text-align: center;
                padding: 10px;
                position: relative;
                display: inline-block;
                width: 100%;
            }
            .video-wrapper {
                position: relative;
                display: inline-block;
                max-width: 100%;
                width: 100%;
            }
            #videoStream {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                display: block;
            }
            #drawingCanvas {
                position: absolute;
                top: 0;
                left: 0;
                border-radius: 5px;
                cursor: crosshair;
                display: none;
                pointer-events: auto;
                z-index: 10;
            }
            .drawing-mode #videoStream {
                cursor: crosshair;
            }
            .controls {
                text-align: center;
                margin: 20px 0;
            }
            .btn {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            .btn:hover {
                background-color: #45a049;
            }
            .btn:active {
                background-color: #3d8b40;
            }
            .btn-danger {
                background-color: #f44336;
            }
            .btn-danger:hover {
                background-color: #da190b;
            }
            .btn:disabled {
                background-color: #666;
                cursor: not-allowed;
            }
            .drawing-mode {
                border: 3px solid #ff9800;
            }
            .info {
                text-align: center;
                margin-top: 20px;
                color: #888;
                font-size: 12px;
            }
            .instruction {
                background-color: #2a2a2a;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                color: #ccc;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐱 Cat Detection System</h1>
            
            <div class="status-panel">
                <div class="status-item">
                    <div class="status-label">Detection Status</div>
                    <div class="status-value" id="status">Checking...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Total Detections</div>
                    <div class="status-value" id="count">0</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Last Trigger</div>
                    <div class="status-value" id="lastTrigger" style="font-size: 16px;">Never</div>
                </div>
            </div>
            
            <div class="instruction" id="instruction">
                <strong>Area of Interest:</strong> Click "Draw Polygon" to select the detection area. Only cats detected within this area will trigger the relay.
            </div>
            
            <div class="controls">
                <button class="btn" id="drawBtn" onclick="startDrawing()">Draw Polygon</button>
                <button class="btn btn-danger" id="removeBtn" onclick="removePolygon()">Remove Polygon</button>
                <button class="btn" id="finishBtn" onclick="finishDrawing()" style="display:none;">Finish Drawing</button>
            </div>
            
            <div class="video-container" id="videoContainer">
                <div class="video-wrapper">
                    <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Video Stream">
                    <canvas id="drawingCanvas"></canvas>
                </div>
            </div>
            
            <div class="info">
                <p id="infoText">Streaming live detection feed</p>
            </div>
        </div>
        
        <script>
            let isDrawing = false;
            let polygonPoints = [];
            let canvas, ctx, img, imgWidth, imgHeight;
            let originalImgSrc = null;
            
            // Initialize
            window.onload = function() {
                canvas = document.getElementById('drawingCanvas');
                ctx = canvas.getContext('2d');
                img = document.getElementById('videoStream');
                
                if (!img) {
                    console.error('Image element not found!');
                    return;
                }
                
                // Update canvas size when image loads
                img.onload = function() {
                    updateCanvasSize();
                    // Draw polygon if exists and not in drawing mode
                    if (!isDrawing && polygonPoints.length > 0) {
                        drawPolygon();
                        canvas.style.display = 'block';
                    }
                };
                
                // Attach click handler to image
                attachImageClickHandler();
                
                // Load saved polygon on page load
                loadPolygon();
                
                // Update status every second
                setInterval(updateStatus, 1000);
            };
            
            function updateCanvasSize() {
                // Get the image's actual displayed size
                const imgRect = img.getBoundingClientRect();
                const wrapper = img.parentElement;
                
                // Set canvas internal resolution to match displayed size
                canvas.width = imgRect.width;
                canvas.height = imgRect.height;
                
                // Set canvas CSS size to match image
                canvas.style.width = imgRect.width + 'px';
                canvas.style.height = imgRect.height + 'px';
                
                // Position canvas to exactly overlay the image (relative to wrapper)
                canvas.style.position = 'absolute';
                canvas.style.top = '0px';
                canvas.style.left = '0px';
                
                console.log('Canvas updated:', {
                    canvasSize: canvas.width + 'x' + canvas.height,
                    imgSize: imgRect.width + 'x' + imgRect.height,
                    imgNatural: img.naturalWidth + 'x' + img.naturalHeight
                });
                
                // Redraw polygon if exists
                if (polygonPoints.length > 0) {
                    drawPolygon();
                }
            }
            
            function startDrawing() {
                if (isDrawing) return;
                
                isDrawing = true;
                polygonPoints = [];
                
                // Pause video stream
                originalImgSrc = img.src;
                fetch('/start_drawing', { method: 'POST' })
                    .then(() => {
                        // Wait a moment for snapshot, then load it
                        setTimeout(() => {
                            if (!img) {
                                console.error('img is undefined!');
                                return;
                            }
                            
                            img.src = '/get_snapshot?' + new Date().getTime();
                            img.onload = function() {
                                // Re-attach click handler after image loads
                                attachImageClickHandler();
                                
                                // Wait a bit for image to fully render
                                setTimeout(() => {
                                    updateCanvasSize();
                                    canvas.style.display = 'block';
                                    canvas.style.pointerEvents = 'none'; // Let clicks pass through to image
                                    document.getElementById('videoContainer').classList.add('drawing-mode');
                                    document.getElementById('drawBtn').disabled = true;
                                    document.getElementById('finishBtn').style.display = 'inline-block';
                                    document.getElementById('instruction').textContent = 'Click on the image to add points. Click "Finish Drawing" when done.';
                                    console.log('Drawing mode activated. Canvas size:', canvas.width, 'x', canvas.height);
                                    console.log('Image natural size:', img.naturalWidth, 'x', img.naturalHeight);
                                    console.log('Image display size:', img.width, 'x', img.height);
                                    console.log('Ready to receive clicks!');
                                }, 200);
                            };
                        }, 500);
                    });
            }
            
            function finishDrawing() {
                console.log('Finish drawing clicked. Current points:', polygonPoints.length);
                console.log('Points array:', polygonPoints);
                
                if (!isDrawing) {
                    alert('Not in drawing mode');
                    return;
                }
                
                if (polygonPoints.length < 3) {
                    alert('Please draw at least 3 points to form a polygon. Currently have: ' + polygonPoints.length + ' points.');
                    return;
                }
                
                // Save polygon
                fetch('/save_polygon', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ points: polygonPoints })
                })
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    // Resume video stream
                    fetch('/stop_drawing', { method: 'POST' })
                        .then(() => {
                            img.src = originalImgSrc || '{{ url_for("video_feed") }}';
                            // Keep canvas visible to show polygon on live stream
                            document.getElementById('videoContainer').classList.remove('drawing-mode');
                            document.getElementById('drawBtn').disabled = false;
                            document.getElementById('finishBtn').style.display = 'none';
                            document.getElementById('instruction').textContent = 'Area of Interest: Click "Draw Polygon" to select the detection area. Only cats detected within this area will trigger the relay.';
                            isDrawing = false;
                        });
                });
            }
            
            function removePolygon() {
                if (confirm('Are you sure you want to remove the polygon? Detection will work on the entire frame.')) {
                    fetch('/remove_polygon', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            polygonPoints = [];
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            canvas.style.display = 'none';
                            alert(data.message);
                        });
                }
            }
            
            function loadPolygon() {
                fetch('/load_polygon')
                    .then(response => response.json())
                    .then(data => {
                        if (data.points && data.points.length > 0) {
                            polygonPoints = data.points;
                            // Wait for image to load before drawing
                            if (img.complete) {
                                updateCanvasSize();
                                drawPolygon();
                                canvas.style.display = 'block';
                            }
                        }
                    });
            }
            
            // Function to attach click handler to image
            function attachImageClickHandler() {
                if (!img) {
                    console.error('Cannot attach click handler: img is undefined');
                    return;
                }
                
                // Remove any existing handler first
                img.removeEventListener('click', handleImageClick);
                
                // Add click handler
                img.addEventListener('click', handleImageClick);
                console.log('Image click handler attached');
            }
            
            // Primary click handler on the image
            function handleImageClick(e) {
                if (!isDrawing) {
                    console.log('Not in drawing mode, ignoring click');
                    return;
                }
                
                e.preventDefault();
                e.stopPropagation();
                
                if (!img) {
                    console.error('img is undefined in click handler');
                    return;
                }
                
                // Get click position relative to image
                const imgRect = img.getBoundingClientRect();
                const x = e.clientX - imgRect.left;
                const y = e.clientY - imgRect.top;
                
                // Check if click is within image bounds
                if (x < 0 || x > imgRect.width || y < 0 || y > imgRect.height) {
                    console.log('Click outside image bounds:', x, y, 'vs', imgRect.width, imgRect.height);
                    return;
                }
                
                // Scale to actual frame dimensions (640x480)
                if (img.naturalWidth === 0 || img.naturalHeight === 0) {
                    console.error('Image natural dimensions are 0!');
                    return;
                }
                
                const scaleX = img.naturalWidth / imgRect.width;
                const scaleY = img.naturalHeight / imgRect.height;
                const scaledX = Math.round(x * scaleX);
                const scaledY = Math.round(y * scaleY);
                
                // Add point
                polygonPoints.push([scaledX, scaledY]);
                console.log('✅ Point added!', {
                    clickPos: [Math.round(x), Math.round(y)],
                    scaledPos: [scaledX, scaledY],
                    totalPoints: polygonPoints.length,
                    imgNatural: img.naturalWidth + 'x' + img.naturalHeight,
                    imgDisplay: imgRect.width + 'x' + imgRect.height
                });
                
                // Update UI
                const instructionEl = document.getElementById('instruction');
                if (instructionEl) {
                    instructionEl.textContent = 
                        `Points added: ${polygonPoints.length}. Click to add more points, then click "Finish Drawing".`;
                }
                
                // Redraw polygon
                drawPolygon();
            }
            
            // Canvas click handler (backup, but image handler should work)
            if (canvas) {
                canvas.addEventListener('click', function(e) {
                    if (!isDrawing) return;
                    // Let the image handler deal with it
                    e.stopPropagation();
                });
            }
            
            function drawPolygon() {
                if (polygonPoints.length === 0) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    return;
                }
                
                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Scale points from frame dimensions to canvas display size
                const scaleX = canvas.width / img.naturalWidth;
                const scaleY = canvas.height / img.naturalHeight;
                
                // Draw filled polygon
                if (polygonPoints.length > 2) {
                    ctx.beginPath();
                    const firstPoint = polygonPoints[0];
                    ctx.moveTo(firstPoint[0] * scaleX, firstPoint[1] * scaleY);
                    
                    for (let i = 1; i < polygonPoints.length; i++) {
                        ctx.lineTo(polygonPoints[i][0] * scaleX, polygonPoints[i][1] * scaleY);
                    }
                    ctx.closePath();
                    
                    // Fill
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                    ctx.fill();
                    
                    // Stroke
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
                
                // Draw points as circles
                ctx.fillStyle = '#00ff00';
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                for (let i = 0; i < polygonPoints.length; i++) {
                    const point = polygonPoints[i];
                    const x = point[0] * scaleX;
                    const y = point[1] * scaleY;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, 6, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                    
                    // Draw point number
                    ctx.fillStyle = '#000000';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText((i + 1).toString(), x, y);
                    ctx.fillStyle = '#00ff00';
                }
                
                console.log('Polygon drawn with', polygonPoints.length, 'points');
            }
            
            function updateStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusEl = document.getElementById('status');
                        const countEl = document.getElementById('count');
                        const lastTriggerEl = document.getElementById('lastTrigger');
                        
                        if (data.detected) {
                            statusEl.textContent = 'CAT DETECTED!';
                            statusEl.className = 'status-value detected';
                        } else {
                            statusEl.textContent = 'No Cat';
                            statusEl.className = 'status-value not-detected';
                        }
                        
                        countEl.textContent = data.count;
                        lastTriggerEl.textContent = data.last_trigger || 'Never';
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            // Update canvas on window resize
            window.addEventListener('resize', function() {
                if (isDrawing || polygonPoints.length > 0) {
                    updateCanvasSize();
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """API endpoint for detection status"""
    with status_lock:
        return {
            "detected": detection_status["detected"],
            "count": detection_status["count"],
            "last_trigger": detection_status["last_trigger"]
        }


if __name__ == '__main__':
    # Load saved polygon on startup
    load_polygon()
    
    # Start detection thread
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    # Give detection thread a moment to initialize
    time.sleep(2)
    
    # Run Flask app
    print("\n" + "="*50)
    print("Starting Flask web server...")
    print("Access the application at: http://<your-ip>:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

