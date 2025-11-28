"""
Flask Backend for Focus Guardian Study Monitor
Integrates the unified face + eye tracking system with web interface
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64
import json
from threading import Lock

app = Flask(__name__)
app.config['SECRET_KEY'] = 'focus-guardian-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ==========================================
# GAZE MODEL DEFINITION
# ==========================================
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# CONFIGURATION
# ==========================================
PHONE_MODEL_PATH = 'yolov8n.pt'
GAZE_MODEL_PATH = 'my_gaze_model.pth'

# Thresholds
PHONE_CONFIDENCE = 0.4

GAZE_LIMIT_HORIZONTAL = 15.0
GAZE_LIMIT_UP = 12.0
GAZE_LIMIT_DOWN = 10.0
EYE_GAIN = 3.0
ZOOM_FACTOR = 1.6
SMOOTHING = 0.6

# ==========================================
# GLOBAL STATE
# ==========================================
class MonitoringState:
    def __init__(self):
        self.monitoring = False
        self.distraction_time_limit = 3.0
        self.eye_sensitivity = 1.0
        
        # Stats
        self.focus_start_time = None
        self.total_focus_time = 0
        self.distraction_count = 0
        
        # Model state
        self.phone_detected_frames = 0
        
        self.calib_global_yaw = 0.0
        self.calib_global_pitch = 0.0
        self.calib_samples = 0
        self.calibrated = False
        self.prev_global_yaw = 0
        self.prev_global_pitch = 0
        
        self.distraction_start_time = None
        self.last_alert_time = 0
        
        self.eye_votes = []
        self.vote_window = 10
        
        self.lock = Lock()

state = MonitoringState()

# ==========================================
# LOAD MODELS
# ==========================================
print("Loading models...")
obj_model = YOLO(PHONE_MODEL_PATH)

device = torch.device("cpu")
gaze_model = None
try:
    gaze_model = GazeNet().to(device)
    gaze_model.load_state_dict(torch.load(GAZE_MODEL_PATH, map_location=device))
    gaze_model.eval()
    print("✅ All models loaded")
except:
    print("⚠️ Gaze model not found")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Camera setup
# In app.py, after camera initialization:
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower from default
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower from default
camera.set(cv2.CAP_PROP_FPS, 30)            # Cap at 30 FPS

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_head_pose_mediapipe(landmarks, img_w, img_h):
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]
    
    total_width = right_cheek.x - left_cheek.x
    yaw_ratio = (nose.x - left_cheek.x) / total_width
    yaw_deg = (yaw_ratio - 0.5) * 180
    
    total_height = chin.y - forehead.y
    pitch_ratio = (nose.y - forehead.y) / total_height
    pitch_deg = (pitch_ratio - 0.5) * 180
    
    return pitch_deg, yaw_deg

def enhance_eye_contrast(eye_img):
    img_yuv = cv2.cvtColor(eye_img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_output = cv2.filter2D(img_output, -1, kernel)
    return img_output

def predict_eye_gaze(eye_crop, is_right_eye):
    if gaze_model is None or eye_crop.size == 0:
        return 0, 0
    
    if is_right_eye:
        eye_crop = cv2.flip(eye_crop, 1)
    
    eye_crop = enhance_eye_contrast(eye_crop)
    pil_img = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = gaze_model(input_tensor)
    
    return pred[0][0].item(), pred[0][1].item()

def get_unified_decision():
    with state.lock:
        if len(state.eye_votes) < 3:
            return False
        
        eye_rate = sum(state.eye_votes) / len(state.eye_votes)
        
        # Eye model is primary decision maker
        return eye_rate >= 0.6

def add_vote(eye_distracted):
    with state.lock:
        state.eye_votes.append(eye_distracted)
        
        if len(state.eye_votes) > state.vote_window:
            state.eye_votes.pop(0)

# ==========================================
# VIDEO PROCESSING
# ==========================================
def process_frame(frame):
    if not state.monitoring:
        return frame, {
            'status': 'stopped',
            'eye_active': False,
            'phone_active': False
        }
    
    h, w = frame.shape[:2]
    
    eye_distracted = False
    eye_reason = ""
    
    # Phone detection
    phone_results = obj_model(frame, classes=[67], verbose=False)
    phone_detected_now = False
    
    for box in phone_results[0].boxes:
        if box.conf[0] > PHONE_CONFIDENCE:
            phone_detected_now = True
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
    
    if phone_detected_now:
        state.phone_detected_frames += 1
        if state.phone_detected_frames >= 2:
            eye_distracted = True
            eye_reason = "Phone Detected"
    else:
        state.phone_detected_frames = 0
    
    # Eye tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(rgb_frame)
    
    if mp_results.multi_face_landmarks:
        for face_landmarks in mp_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            head_pitch_mp, head_yaw_mp = get_head_pose_mediapipe(landmarks, w, h)
            
            if head_yaw_mp > 15:
                p1, p2 = landmarks[33], landmarks[133]
                is_right = True
            else:
                p1, p2 = landmarks[33], landmarks[133]
                is_right = True
            
            cx = int((p1.x + p2.x)/2 * w)
            cy = int((p1.y + p2.y)/2 * h)
            width = abs(int((p1.x - p2.x) * w))
            crop = max(int(width * ZOOM_FACTOR), 30)
            
            x1 = max(0, cx - crop//2)
            y1 = max(0, cy - crop//2)
            x2 = min(w, cx + crop//2)
            y2 = min(h, cy + crop//2)
            
            eye_img = frame[y1:y2, x1:x2]
            
            raw_eye_pitch, raw_eye_yaw = predict_eye_gaze(eye_img, is_right)
            
            eye_yaw_deg = raw_eye_yaw * 57.3 * EYE_GAIN
            eye_pitch_deg = raw_eye_pitch * 57.3 * EYE_GAIN
            
            current_global_yaw = head_yaw_mp + eye_yaw_deg
            current_global_pitch = head_pitch_mp + eye_pitch_deg
            
            global_yaw = (SMOOTHING * state.prev_global_yaw) + ((1-SMOOTHING) * current_global_yaw)
            global_pitch = (SMOOTHING * state.prev_global_pitch) + ((1-SMOOTHING) * current_global_pitch)
            
            state.prev_global_yaw = global_yaw
            state.prev_global_pitch = global_pitch
            
            if not state.calibrated:
                if abs(global_yaw) < GAZE_LIMIT_HORIZONTAL and abs(head_yaw_mp) < 40:
                    state.calib_global_yaw += global_yaw
                    state.calib_global_pitch += global_pitch
                    state.calib_samples += 1
                    
                    if state.calib_samples >= 30:
                        state.calib_global_yaw /= state.calib_samples
                        state.calib_global_pitch /= state.calib_samples
                        state.calibrated = True
            
            if state.calibrated:
                effective_yaw = global_yaw - state.calib_global_yaw
                effective_pitch = global_pitch - state.calib_global_pitch
                
                adjusted_h_limit = GAZE_LIMIT_HORIZONTAL / state.eye_sensitivity
                adjusted_up_limit = GAZE_LIMIT_UP / state.eye_sensitivity
                adjusted_down_limit = GAZE_LIMIT_DOWN / state.eye_sensitivity
                
                if abs(effective_yaw) > adjusted_h_limit:
                    eye_distracted = True
                    eye_reason = "Looking Sideways"
                
                if effective_pitch < -adjusted_up_limit:
                    eye_distracted = True
                    eye_reason = "Looking Up"
                
                if effective_pitch > adjusted_down_limit:
                    eye_distracted = True
                    eye_reason = "Looking Down"
    
    # Unified decision
    add_vote(eye_distracted)
    is_distracted = get_unified_decision()
    
    # Status determination
    status_info = {
        'eye_active': state.calibrated,
        'phone_active': phone_detected_now,
        'eye_reason': eye_reason
    }
    
    if is_distracted:
        if state.distraction_start_time is None:
            state.distraction_start_time = time.time()
        
        elapsed = time.time() - state.distraction_start_time
        
        if elapsed >= state.distraction_time_limit:
            status_info['status'] = 'alert'
            status_info['message'] = '⚠️ DISTRACTED!'
            status_info['detail'] = eye_reason or 'Please refocus'
            
            if time.time() - state.last_alert_time > 2.0:
                state.distraction_count += 1
                state.last_alert_time = time.time()
                socketio.emit('alert', {'type': 'distraction'})
        else:
            remaining = state.distraction_time_limit - elapsed
            status_info['status'] = 'warning'
            status_info['message'] = f'Warning: {int(remaining)}s'
            status_info['detail'] = eye_reason
    else:
        state.distraction_start_time = None
        status_info['status'] = 'focused'
        status_info['message'] = '✓ Focused'
        status_info['detail'] = 'Keep up the good work!'
    
    # Draw status on frame
    color = (0, 255, 0) if status_info['status'] == 'focused' else \
            (0, 255, 255) if status_info['status'] == 'warning' else (0, 0, 255)
    
    cv2.putText(frame, status_info['message'], (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame, status_info

def generate_frames():
    frame_skip = 2  # Process every 2nd frame
    frame_counter = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Only process every Nth frame
        if frame_counter % frame_skip == 0:
            processed_frame, status_info = process_frame(frame)
            if state.monitoring:
                socketio.emit('status_update', status_info)
        else:
            processed_frame = frame
        
        frame_counter += 1
        
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Emit status updates
        if state.monitoring:
            socketio.emit('status_update', status_info)

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    state.monitoring = True
    state.focus_start_time = time.time()
    return jsonify({'success': True})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    state.monitoring = False
    if state.focus_start_time:
        state.total_focus_time += time.time() - state.focus_start_time
        state.focus_start_time = None
    return jsonify({'success': True})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    state.calibrated = False
    state.calib_samples = 0
    state.calib_global_yaw = 0.0
    state.calib_global_pitch = 0.0
    return jsonify({'success': True})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    state.distraction_time_limit = float(data.get('alertDelay', 3))
    state.eye_sensitivity = float(data.get('eyeSensitivity', 100)) / 100
    return jsonify({'success': True})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    focus_time = state.total_focus_time
    if state.focus_start_time:
        focus_time += time.time() - state.focus_start_time
    
    minutes = int(focus_time // 60)
    seconds = int(focus_time % 60)
    
    score = max(0, 100 - (state.distraction_count * 5))
    
    return jsonify({
        'focusTime': f"{minutes:02d}:{seconds:02d}",
        'distractionCount': state.distraction_count,
        'focusScore': f"{score}%"
    })

# ==========================================
# SOCKETIO EVENTS
# ==========================================
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ==========================================
# RUN
# ==========================================
if __name__ == '__main__':
    print("\n🎓 Focus Guardian Server Starting...")
    print("📡 Open http://localhost:5000 in your browser\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)