import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. MODEL DEFINITION
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
# 2. CONFIGURATION
# ==========================================
MODEL_PATH = "my_gaze_model.pth" 
ALARM_PATH = "alarm.wav"        

# --- THRESHOLDS ---
GAZE_LIMIT_HORIZONTAL = 25.0 
GAZE_LIMIT_UP = 10.0    
GAZE_LIMIT_DOWN = 35.0  

# HARD LIMIT: If head is past this, you are DISTRACTED.
HARD_HEAD_LIMIT = 90.0  

# --- COMPROMISE LIMITS (Ignore Eyes if Head is too far) ---
# Horizontal: Increased from 55 to 70 per your request
COMP_HEAD_LIMIT = 70.0  

# Vertical: New limit. If you look down/up more than 40 degrees, 
# eye compensation is disabled.
COMP_PITCH_LIMIT = 40.0

EYE_GAIN = 3.0          
ZOOM_FACTOR = 1.6 
ALARM_DELAY = 3.0
SMOOTHING = 0.6 

# ==========================================
# 3. SETUP & INITIALIZATION
# ==========================================
device = torch.device("cpu") 
model = GazeNet().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded.")
except:
    print("❌ Model not found. Ensure 'my_gaze_model.pth' is in the folder.")
    exit()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound(ALARM_PATH)
except:
    alarm_sound = None

cap = cv2.VideoCapture(0)

# --- Runtime Variables ---
distraction_start_time = None
focused_frame_count = 0
is_alarm_active = False

prev_global_yaw = 0
prev_global_pitch = 0

# --- Calibration Variables ---
calib_global_yaw = 0.0
calib_global_pitch = 0.0
calib_head_yaw = 0.0    
calib_head_pitch = 0.0  
calib_samples = 0
CALIBRATION_FRAMES = 50   
calibrated = False

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_head_pose_ratio(landmarks, img_w, img_h):
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
    img_yuv = cv2.cvtColor(eye_img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_output = cv2.filter2D(img_output, -1, kernel)
    return img_output

def predict_eye_gaze(eye_crop, is_right_eye):
    if eye_crop.size == 0: 
        return 0, 0
    if is_right_eye: 
        eye_crop = cv2.flip(eye_crop, 1)
    
    eye_crop = enhance_eye_contrast(eye_crop)
    pil_img = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(input_tensor)
    return pred[0][0].item(), pred[0][1].item()

# ==========================================
# 5. MAIN LOOP
# ==========================================
print("📷 Starting Enhanced Focus Detector...")
print("   Press 'c' to recalibrate.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape
    
    is_distracted = False
    status_text = "FOCUSED"
    debug_text_1 = ""
    override_text = ""
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # --- 1. HEAD POSE ---
            head_pitch, head_yaw = get_head_pose_ratio(landmarks, img_w, img_h)
            
            # Clamp values
            head_yaw = max(-85, min(85, head_yaw))
            head_pitch = max(-85, min(85, head_pitch))

            # --- 2. EYE SELECTION ---
            if head_yaw > 15: 
                p1, p2 = landmarks[33], landmarks[133] 
                is_right_eye_landmark = True
            elif head_yaw < -15:
                p1, p2 = landmarks[362], landmarks[263] 
                is_right_eye_landmark = False
            else:
                p1, p2 = landmarks[33], landmarks[133] 
                is_right_eye_landmark = True

            # --- 3. EYE EXTRACTION & PREDICTION ---
            cx, cy = int((p1.x + p2.x)/2 * img_w), int((p1.y + p2.y)/2 * img_h)
            width = abs(int((p1.x - p2.x) * img_w))
            crop = max(int(width * ZOOM_FACTOR), 30)
            x1, y1 = max(0, cx - crop//2), max(0, cy - crop//2)
            x2, y2 = min(img_w, cx + crop//2), min(img_h, cy + crop//2)
            eye_img = frame[y1:y2, x1:x2]

            raw_eye_pitch, raw_eye_yaw = predict_eye_gaze(eye_img, is_right_eye_landmark)
            if not is_right_eye_landmark: raw_eye_yaw = -raw_eye_yaw
            raw_eye_pitch = -raw_eye_pitch 

            # --- 4. CALCULATION & SMOOTHING ---
            eye_yaw_deg = raw_eye_yaw * 57.3 * EYE_GAIN
            eye_pitch_deg = raw_eye_pitch * 57.3 * EYE_GAIN
            
            # Boost Right-Side eye signal
            if head_yaw > 20 and eye_yaw_deg < 0:
                eye_yaw_deg *= 1.5 

            current_global_yaw = head_yaw + eye_yaw_deg
            current_global_pitch = head_pitch + eye_pitch_deg
            
            global_yaw = (SMOOTHING * prev_global_yaw) + ((1-SMOOTHING) * current_global_yaw)
            global_pitch = (SMOOTHING * prev_global_pitch) + ((1-SMOOTHING) * current_global_pitch)
            
            prev_global_yaw = global_yaw
            prev_global_pitch = global_pitch

            # --- 5. CALIBRATION ROUTINE ---
            if not calibrated:
                effective_yaw = global_yaw
                effective_pitch = global_pitch
                rel_head_yaw = head_yaw 
                rel_head_pitch = head_pitch
                
                if (abs(global_yaw) < GAZE_LIMIT_HORIZONTAL and
                    abs(head_yaw) < 40): 
                    
                    calib_global_yaw += global_yaw
                    calib_global_pitch += global_pitch
                    calib_head_yaw += head_yaw
                    calib_head_pitch += head_pitch
                    calib_samples += 1

                    if calib_samples >= CALIBRATION_FRAMES:
                        calib_global_yaw /= calib_samples
                        calib_global_pitch /= calib_samples
                        calib_head_yaw /= calib_samples
                        calib_head_pitch /= calib_samples
                        calibrated = True
                        print(f"✅ Calibrated! Zero Head:{calib_head_yaw:.1f}")

            if calibrated:
                effective_yaw = global_yaw - calib_global_yaw
                effective_pitch = global_pitch - calib_global_pitch
                rel_head_yaw = head_yaw - calib_head_yaw
                rel_head_pitch = head_pitch - calib_head_pitch

                # --- 6. DECISION LOGIC ---
                if abs(rel_head_yaw) > HARD_HEAD_LIMIT: 
                    is_distracted = True
                elif abs(effective_yaw) > GAZE_LIMIT_HORIZONTAL:
                    is_distracted = True
                elif effective_pitch < -GAZE_LIMIT_UP: 
                    is_distracted = True
                elif effective_pitch > GAZE_LIMIT_DOWN: 
                    is_distracted = True
                
                # --- 7. OVERRIDES (LIMITED RANGE) ---
                
                # --- HORIZONTAL CHECK ---
                head_yaw_safe = abs(rel_head_yaw) < COMP_HEAD_LIMIT
                
                if head_yaw_safe:
                    # Left Comp
                    if (rel_head_yaw < -20 and eye_yaw_deg > 8):
                        is_distracted = False
                        override_text = "[H-Comp]"
                        distraction_start_time = None 
                        focused_frame_count = 5       
                    
                    # Right Comp
                    if (rel_head_yaw > 20 and eye_yaw_deg < -8):
                        is_distracted = False
                        override_text = "[H-Comp]"
                        distraction_start_time = None 
                        focused_frame_count = 5 
                
                # --- VERTICAL CHECK (NEW) ---
                head_pitch_safe = abs(rel_head_pitch) < COMP_PITCH_LIMIT
                
                if head_pitch_safe:
                    if (rel_head_pitch < -8 and eye_pitch_deg > 10):
                         is_distracted = False
                         override_text = "[V-Comp]"
                         distraction_start_time = None 
                         focused_frame_count = 5

            else:
                status_text = "CALIBRATING..."
                is_distracted = False

            debug_text_1 = f"RelHead: {rel_head_yaw:.0f} | EffYaw: {effective_yaw:.0f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    else:
        is_distracted = True
        status_text = "NO FACE"

    # --- 8. ALARM & TIMER ---
    if is_distracted:
        focused_frame_count = 0
        if distraction_start_time is None:
            distraction_start_time = time.time()
    else:
        focused_frame_count += 1
        if focused_frame_count > 5:
            distraction_start_time = None
            if is_alarm_active and alarm_sound:
                alarm_sound.stop()
                is_alarm_active = False

    # --- 9. DRAW UI ---
    if distraction_start_time is not None:
        elapsed = time.time() - distraction_start_time
        if elapsed > ALARM_DELAY:
            status_text = "DISTRACTED!"
            color = (0, 0, 255) 
            if alarm_sound and not is_alarm_active:
                alarm_sound.play(loops=-1)
                is_alarm_active = True
        else:
            status_text = f"Warning: {elapsed:.1f}s"
            color = (0, 165, 255) 
    else:
        if not calibrated:
            color = (200, 200, 200)
        else:
            status_text = "FOCUSED"
            color = (0, 255, 0) 

    calib_info = "CALIBRATED" if calibrated else f"CALIB {calib_samples}/{CALIBRATION_FRAMES}"
    
    cv2.putText(frame, status_text + " " + override_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, debug_text_1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, calib_info, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Enhanced Focus Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'):
        calibrated = False
        calib_global_yaw = 0.0
        calib_global_pitch = 0.0
        calib_head_yaw = 0.0
        calib_head_pitch = 0.0
        calib_samples = 0
        print("🔁 Recalibrating...")

cap.release()
cv2.destroyAllWindows()