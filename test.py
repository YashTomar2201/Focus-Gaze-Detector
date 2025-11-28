import cv2
import numpy as np
import time
from ultralytics import YOLO
import winsound
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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
# Model paths
POSE_MODEL_PATH = 'study_monitor/pose_model_v24/weights/best.pt'
PHONE_MODEL_PATH = 'yolov8n.pt'
GAZE_MODEL_PATH = 'my_gaze_model.pth'

# Face tracking thresholds
EAR_THRESHOLD = 0.21
DROWSY_FRAMES = 20
PHONE_CONFIDENCE = 0.4
DISTRACTION_YAW = 25
DISTRACTION_PITCH = 20
STABILITY_DEADZONE = 3.0
NO_FACE_TIME_LIMIT = 3.0

# Eye tracking thresholds
GAZE_LIMIT_HORIZONTAL = 15.0
GAZE_LIMIT_UP = 12.0
GAZE_LIMIT_DOWN = 10.0
EYE_GAIN = 3.0
ZOOM_FACTOR = 1.6
SMOOTHING = 0.6

# Unified decision thresholds
DISTRACTION_TIME_LIMIT = 3.0  # Seconds before alarm
CONFIDENCE_THRESHOLD = 0.6  # Both models must agree 60% of time

# 300-W Dataset Indices
NOSE_TIP = 30
CHIN = 8
LEFT_MOUTH = 48
RIGHT_MOUTH = 54
LEFT_EYE_PTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_PTS = [42, 43, 44, 45, 46, 47]

model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
])

# ==========================================
# INITIALIZATION
# ==========================================
print("Loading models...")

# Load YOLO models
pose_model = YOLO(POSE_MODEL_PATH)
obj_model = YOLO(PHONE_MODEL_PATH)

# Load gaze model
device = torch.device("cpu")
gaze_model = GazeNet().to(device)
try:
    gaze_model.load_state_dict(torch.load(GAZE_MODEL_PATH, map_location=device))
    gaze_model.eval()
    print("✅ Gaze model loaded")
except:
    print("⚠️  Gaze model not found - using face tracking only")
    gaze_model = None

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Transform for gaze model
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_ear(eye_pts):
    """Calculate Eye Aspect Ratio for drowsiness"""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, cam_matrix, dist_coeffs):
    """Estimate head pose using PnP"""
    image_points = np.array([
        landmarks[NOSE_TIP],
        landmarks[CHIN],
        landmarks[36],
        landmarks[45],
        landmarks[LEFT_MOUTH],
        landmarks[RIGHT_MOUTH]
    ], dtype="double")
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], rotation_vector, translation_vector

def get_head_pose_mediapipe(landmarks, img_w, img_h):
    """Get head pose from MediaPipe landmarks"""
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]
    
    # Yaw
    total_width = right_cheek.x - left_cheek.x
    yaw_ratio = (nose.x - left_cheek.x) / total_width
    yaw_deg = (yaw_ratio - 0.5) * 180
    
    # Pitch
    total_height = chin.y - forehead.y
    pitch_ratio = (nose.y - forehead.y) / total_height
    pitch_deg = (pitch_ratio - 0.5) * 180
    
    return pitch_deg, yaw_deg

def enhance_eye_contrast(eye_img):
    """Enhance eye image for better gaze detection"""
    img_yuv = cv2.cvtColor(eye_img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_output = cv2.filter2D(img_output, -1, kernel)
    return img_output

def predict_eye_gaze(eye_crop, is_right_eye):
    """Predict gaze direction from eye crop"""
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

# ==========================================
# STATE MANAGEMENT
# ==========================================
class UnifiedMonitorState:
    def __init__(self):
        # Face tracking state
        self.drowsy_counter = 0
        self.last_face_detected_time = time.time()
        self.phone_detected_frames = 0
        self.neutral_pitch = -154.10
        self.neutral_yaw = 39.31
        
        # Eye tracking state
        self.calib_global_yaw = 0.0
        self.calib_global_pitch = 0.0
        self.calib_samples = 0
        self.calibrated = False
        self.prev_global_yaw = 0
        self.prev_global_pitch = 0
        
        # Unified decision state
        self.distraction_start_time = None
        self.last_alert_time = 0
        self.recalibrate_display_time = 0
        self.recalibrate_mode = False
        
        # Voting system - track recent decisions
        self.face_votes = []  # Last N frames
        self.eye_votes = []   # Last N frames
        self.vote_window = 10  # Number of frames to consider
        
    def update_face_detected(self):
        self.last_face_detected_time = time.time()
    
    def get_time_since_face(self):
        return time.time() - self.last_face_detected_time
    
    def is_face_lost(self):
        return self.get_time_since_face() >= NO_FACE_TIME_LIMIT
    
    def add_vote(self, face_distracted, eye_distracted):
        """Add detection votes from both models"""
        self.face_votes.append(face_distracted)
        self.eye_votes.append(eye_distracted)
        
        # Keep only last N votes
        if len(self.face_votes) > self.vote_window:
            self.face_votes.pop(0)
        if len(self.eye_votes) > self.vote_window:
            self.eye_votes.pop(0)
    
    def get_unified_decision(self):
        """
        Unified decision logic (Eye-weighted):
        - Eye model has 70% weight, Face model has 30% weight
        - Eye model strong agreement (70%+) = distracted
        - Face model needs higher threshold (85%+) alone
        - Weighted combination > 60% = distracted
        """
        if len(self.face_votes) < 3:  # Need minimum samples
            return False
        
        face_distraction_rate = sum(self.face_votes) / len(self.face_votes)
        eye_distraction_rate = sum(self.eye_votes) / len(self.eye_votes)
        
        # Eye model strong agreement (lower threshold due to higher accuracy)
        if eye_distraction_rate >= 0.7:
            return True
        
        # Face model needs very strong agreement to trigger alone
        if face_distraction_rate >= 0.85:
            return True
        
        # Weighted combination: 70% eye + 30% face
        weighted_score = (eye_distraction_rate * 0.7) + (face_distraction_rate * 0.3)
        
        if weighted_score >= 0.6:
            return True
        
        return False
    
    def start_distraction_timer(self):
        if self.distraction_start_time is None:
            self.distraction_start_time = time.time()
    
    def reset_distraction_timer(self):
        self.distraction_start_time = None
    
    def get_distraction_elapsed(self):
        if self.distraction_start_time is None:
            return 0
        return time.time() - self.distraction_start_time
    
    def is_distraction_timeout(self):
        return self.get_distraction_elapsed() >= DISTRACTION_TIME_LIMIT
    
    def get_remaining_time(self):
        elapsed = self.get_distraction_elapsed()
        return max(0, DISTRACTION_TIME_LIMIT - elapsed)
    
    def should_play_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time > 2.0:
            self.last_alert_time = current_time
            return True
        return False
    
    def set_recalibrate_display(self, duration=2.0):
        self.recalibrate_display_time = time.time() + duration
    
    def should_show_recalibrate_message(self):
        return time.time() < self.recalibrate_display_time

# ==========================================
# MAIN SETUP
# ==========================================
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("ERROR: Could not open camera")
    exit()

h, w = frame.shape[:2]
focal_length = w
center = (w / 2, h / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1))

state = UnifiedMonitorState()
frame_count = 0
debug_mode = False

print("\n🎓 Starting Unified Study Monitor...")
print("   Press 'q' to quit")
print("   Press 'c' to recalibrate")
print("   Press 'd' to toggle debug mode\n")

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    original_frame = frame.copy()
    
    # Status variables
    face_distracted = False
    eye_distracted = False
    face_reason = ""
    eye_reason = ""
    alert_triggered = False
    
    # Keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('c'):
        state.recalibrate_mode = True
        state.calibrated = False
        state.calib_samples = 0
        state.calib_global_yaw = 0.0
        state.calib_global_pitch = 0.0
        print("[INFO] Recalibrating both models...")
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    # ==========================================
    # PART 1: FACE TRACKING (YOLO)
    # ==========================================
    
    # Phone detection
    if frame_count % 10 == 0:
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
                face_distracted = True
                face_reason = "PHONE"
        else:
            state.phone_detected_frames = 0
    
    # Face pose detection
    pose_results = pose_model(frame, verbose=False, conf=0.5)
    face_detected_yolo = False
    
    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
        kpts = pose_results[0].keypoints.xy.cpu().numpy()[0]
        
        if len(kpts) >= 68:
            face_detected_yolo = True
            state.update_face_detected()
            
            # Drowsiness check
            left_ear = calculate_ear(kpts[LEFT_EYE_PTS])
            right_ear = calculate_ear(kpts[RIGHT_EYE_PTS])
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                state.drowsy_counter += 1
                if state.drowsy_counter >= DROWSY_FRAMES:
                    face_distracted = True
                    face_reason = "DROWSY"
            else:
                state.drowsy_counter = 0
            
            # Head pose estimation
            pitch, yaw, rot_vec, trans_vec = get_head_pose(kpts, camera_matrix, dist_coeffs)
            
            # Recalibration
            if state.recalibrate_mode:
                state.neutral_pitch = pitch
                state.neutral_yaw = yaw
                state.recalibrate_mode = False
                state.set_recalibrate_display()
            
            # Check head pose deviation
            pitch_diff = pitch - state.neutral_pitch
            yaw_diff = yaw - state.neutral_yaw
            
            if abs(yaw_diff) > STABILITY_DEADZONE:
                effective_yaw_diff = abs(yaw_diff) - STABILITY_DEADZONE
                if effective_yaw_diff > DISTRACTION_YAW:
                    face_distracted = True
                    if not face_reason:
                        face_reason = "HEAD-YAW"
            
            if abs(pitch_diff) > STABILITY_DEADZONE:
                effective_pitch_diff = abs(pitch_diff) - STABILITY_DEADZONE
                if effective_pitch_diff > DISTRACTION_PITCH:
                    face_distracted = True
                    if not face_reason:
                        face_reason = "HEAD-PITCH"
    
    # Check for face loss
    if not face_detected_yolo:
        if state.is_face_lost():
            face_distracted = True
            face_reason = "NO-FACE"
    
    # ==========================================
    # PART 2: EYE TRACKING (MediaPipe + Gaze)
    # ==========================================
    
    if gaze_model is not None:
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        mp_results = face_mesh.process(rgb_frame)
        
        if mp_results.multi_face_landmarks:
            for face_landmarks in mp_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Get head pose
                head_pitch_mp, head_yaw_mp = get_head_pose_mediapipe(landmarks, w, h)
                
                # Select eye based on head angle
                if head_yaw_mp > 15:
                    p1, p2 = landmarks[33], landmarks[133]
                    is_right = True
                elif head_yaw_mp < -15:
                    p1, p2 = landmarks[362], landmarks[263]
                    is_right = False
                else:
                    p1, p2 = landmarks[33], landmarks[133]
                    is_right = True
                
                # Extract eye region
                cx = int((p1.x + p2.x)/2 * w)
                cy = int((p1.y + p2.y)/2 * h)
                width = abs(int((p1.x - p2.x) * w))
                crop = max(int(width * ZOOM_FACTOR), 30)
                
                x1 = max(0, cx - crop//2)
                y1 = max(0, cy - crop//2)
                x2 = min(w, cx + crop//2)
                y2 = min(h, cy + crop//2)
                
                eye_img = original_frame[y1:y2, x1:x2]
                
                # Predict gaze
                raw_eye_pitch, raw_eye_yaw = predict_eye_gaze(eye_img, is_right)
                
                if not is_right:
                    raw_eye_yaw = -raw_eye_yaw
                    raw_eye_pitch = -raw_eye_pitch
                
                eye_yaw_deg = raw_eye_yaw * 57.3 * EYE_GAIN
                eye_pitch_deg = raw_eye_pitch * 57.3 * EYE_GAIN
                
                current_global_yaw = head_yaw_mp + eye_yaw_deg
                current_global_pitch = head_pitch_mp + eye_pitch_deg
                
                global_yaw = (SMOOTHING * state.prev_global_yaw) + ((1-SMOOTHING) * current_global_yaw)
                global_pitch = (SMOOTHING * state.prev_global_pitch) + ((1-SMOOTHING) * current_global_pitch)
                
                state.prev_global_yaw = global_yaw
                state.prev_global_pitch = global_pitch
                
                # Calibration
                if not state.calibrated:
                    if abs(global_yaw) < GAZE_LIMIT_HORIZONTAL and abs(head_yaw_mp) < 40:
                        state.calib_global_yaw += global_yaw
                        state.calib_global_pitch += global_pitch
                        state.calib_samples += 1
                        
                        if state.calib_samples >= 30:
                            state.calib_global_yaw /= state.calib_samples
                            state.calib_global_pitch /= state.calib_samples
                            state.calibrated = True
                            print(f"✅ Eye tracking calibrated!")
                
                # Check gaze deviation (No tilt lock - always active)
                if state.calibrated:
                    effective_yaw = global_yaw - state.calib_global_yaw
                    effective_pitch = global_pitch - state.calib_global_pitch
                    
                    # Horizontal gaze check
                    if abs(effective_yaw) > GAZE_LIMIT_HORIZONTAL:
                        eye_distracted = True
                        eye_reason = "GAZE-YAW"
                    
                    # Vertical gaze checks
                    if effective_pitch < -GAZE_LIMIT_UP:
                        eye_distracted = True
                        eye_reason = "GAZE-UP"
                    
                    if effective_pitch > GAZE_LIMIT_DOWN:
                        eye_distracted = True
                        eye_reason = "GAZE-DOWN"
    
    # ==========================================
    # PART 3: UNIFIED DECISION
    # ==========================================
    
    # Add votes
    state.add_vote(face_distracted, eye_distracted)
    
    # Get unified decision
    is_distracted = state.get_unified_decision()
    
    # Timer logic
    if is_distracted:
        state.start_distraction_timer()
        elapsed = state.get_distraction_elapsed()
        remaining = state.get_remaining_time()
        
        if state.is_distraction_timeout():
            status_text = "⚠️ ALERT: DISTRACTED!"
            status_color = (0, 0, 255)
            alert_triggered = True
        else:
            status_text = f"⚠️ Warning... {int(remaining)}s"
            status_color = (0, 255, 255)
    else:
        state.reset_distraction_timer()
        status_text = "✓ FOCUSED"
        status_color = (0, 255, 0)
    
    # Play alarm
    if alert_triggered:
        if state.should_play_alert():
            winsound.Beep(1000, 500)
    
    # ==========================================
    # PART 4: DISPLAY
    # ==========================================
    
    # Main status
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
    
    # Debug info
    if debug_mode:
        face_rate = sum(state.face_votes) / max(len(state.face_votes), 1) * 100
        eye_rate = sum(state.eye_votes) / max(len(state.eye_votes), 1) * 100
        
        cv2.putText(frame, f"Face: {face_rate:.0f}% ({face_reason})", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Eye: {eye_rate:.0f}% ({eye_reason})", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Calibration status
    calib_text = ""
    if not state.calibrated:
        calib_text = f"Calibrating eye tracking... {state.calib_samples}/30"
    else:
        calib_text = "Calibrated ✓"
    
    cv2.putText(frame, calib_text, (20, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Recalibration message
    if state.should_show_recalibrate_message():
        cv2.putText(frame, "RECALIBRATED!", (w // 2 - 100, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imshow("Unified Study Monitor", frame)

cap.release()
cv2.destroyAllWindows()
print("Monitor stopped.")