import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

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
YOLO_MODEL_PATH = "yolov8n.pt"

# --- THRESHOLDS ---
GAZE_LIMIT_HORIZONTAL = 15.0  
GAZE_LIMIT_UP = 12.0  
GAZE_LIMIT_DOWN = 10.0  
HARD_HEAD_LIMIT = 100.0
COMP_HEAD_LIMIT = 80.0  
COMP_PITCH_LIMIT = 25.0  
EAR_THRESHOLD = 0.22 

# --- PHONE DETECTION CONFIG ---
PHONE_CONFIDENCE = 0.5
PHONE_CHECK_INTERVAL = 15 

EYE_GAIN = 3.0
ZOOM_FACTOR = 1.6
ALARM_DELAY = 2.0
SMOOTHING = 0.6

# ==========================================
# 3. SETUP & INITIALIZATION
# ==========================================
device = torch.device("cpu")
model = GazeNet().to(device)

print("⏳ Loading Gaze Model...")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Gaze Model loaded.")
except:
    print("❌ Gaze Model not found. Ensure 'my_gaze_model.pth' is in the folder.")
    exit()

print("⏳ Loading YOLO Model (First run will download weights)...")
try:
    phone_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO Model loaded.")
except Exception as e:
    print(f"❌ Error loading YOLO: {e}")
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

# New variables for phone detection
phone_detected_buffer = False 
phone_box = None

# --- Calibration Variables ---
calib_global_yaw = 0.0
calib_global_pitch = 0.0
calib_head_yaw = 0.0
calib_head_pitch = 0.0
calib_head_roll = 0.0
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
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Yaw
    total_width = right_cheek.x - left_cheek.x
    yaw_ratio = (nose.x - left_cheek.x) / total_width
    yaw_deg = (yaw_ratio - 0.5) * 180

    # Pitch
    total_height = chin.y - forehead.y
    pitch_ratio = (nose.y - forehead.y) / total_height
    pitch_deg = (pitch_ratio - 0.5) * 180

    # Roll
    eye_dx = right_eye.x - left_eye.x
    eye_dy = right_eye.y - left_eye.y
    roll_rad = np.arctan2(eye_dy, eye_dx)
    roll_deg = np.degrees(roll_rad)
    roll_deg = -(roll_deg)

    return pitch_deg, yaw_deg, roll_deg

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

def calculate_ear(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    
    if h == 0: return 0.0
    ear = (v1 + v2) / (2.0 * h)
    return ear

# ==========================================
# 5. MAIN LOOP
# ==========================================
print("📷 Starting Enhanced Focus Detector (Phone + Eyes + Head)...")
print("   Press 'c' to recalibrate.")
print("   Press 'd' to toggle debug mode.")

debug_mode = False
frame_count = 0

RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if frame_count % PHONE_CHECK_INTERVAL == 0:
        yolo_results = phone_model(frame, classes=[67], conf=PHONE_CONFIDENCE, verbose=False)
        
        if len(yolo_results[0].boxes) > 0:
            phone_detected_buffer = True
            box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy()
            phone_box = box.astype(int)
        else:
            phone_detected_buffer = False
            phone_box = None

    if phone_detected_buffer and phone_box is not None:
        cv2.rectangle(frame, (phone_box[0], phone_box[1]), (phone_box[2], phone_box[3]), (0, 0, 255), 3)
        cv2.putText(frame, "PHONE DETECTED", (phone_box[0], phone_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    results = face_mesh.process(rgb_frame)
    
    is_distracted = False
    is_sleeping = False
    is_on_phone = False
    status_text = "FOCUSED"
    debug_text_1 = ""
    override_text = ""
    frame_count += 1

    if phone_detected_buffer:
        is_distracted = True
        is_on_phone = True
        status_text = "PHONE DETECTED!"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = calculate_ear(landmarks, LEFT_EYE_IDXS)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS)
            avg_ear = (left_ear + right_ear) / 2.0

            head_pitch, head_yaw, head_roll = get_head_pose_ratio(landmarks, img_w, img_h)
            head_yaw = max(-85, min(85, head_yaw))
            head_pitch = max(-85, min(85, head_pitch))
            head_roll = max(-85, min(85, head_roll))

            if head_yaw > 15:
                p1, p2 = landmarks[33], landmarks[133]
                is_right_eye_landmark = True
            elif head_yaw < -15:
                p1, p2 = landmarks[362], landmarks[263]
                is_right_eye_landmark = False
            else:
                p1, p2 = landmarks[33], landmarks[133]
                is_right_eye_landmark = True

            cx, cy = int((p1.x + p2.x)/2 * img_w), int((p1.y + p2.y)/2 * img_h)
            width = abs(int((p1.x - p2.x) * img_w))
            crop = max(int(width * ZOOM_FACTOR), 30)
            x1, y1 = max(0, cx - crop//2), max(0, cy - crop//2)
            x2, y2 = min(img_w, cx + crop//2), min(img_h, cy + crop//2)
            eye_img = frame[y1:y2, x1:x2]

            raw_eye_pitch, raw_eye_yaw = predict_eye_gaze(eye_img, is_right_eye_landmark)
            
            if not is_right_eye_landmark:
                raw_eye_yaw = -raw_eye_yaw
                raw_eye_pitch = -raw_eye_pitch

            eye_yaw_deg = raw_eye_yaw * 57.3 * EYE_GAIN
            eye_pitch_deg = raw_eye_pitch * 57.3 * EYE_GAIN

            rel_head_yaw = head_yaw - calib_head_yaw
            rel_head_pitch = head_pitch - calib_head_pitch
            rel_head_roll = head_roll - calib_head_roll

            if rel_head_yaw > 20 and eye_yaw_deg < 0:
                eye_yaw_deg *= 2.0 

            current_global_yaw = head_yaw + eye_yaw_deg
            current_global_pitch = head_pitch + eye_pitch_deg

            global_yaw = (SMOOTHING * prev_global_yaw) + ((1-SMOOTHING) * current_global_yaw)
            global_pitch = (SMOOTHING * prev_global_pitch) + ((1-SMOOTHING) * current_global_pitch)

            prev_global_yaw = global_yaw
            prev_global_pitch = global_pitch

            if not calibrated:
                effective_yaw = global_yaw
                effective_pitch = global_pitch

                if (abs(global_yaw) < GAZE_LIMIT_HORIZONTAL and abs(head_yaw) < 40):
                    calib_global_yaw += global_yaw
                    calib_global_pitch += global_pitch
                    calib_head_yaw += head_yaw
                    calib_head_pitch += head_pitch
                    calib_head_roll += head_roll
                    calib_samples += 1

                    if calib_samples >= CALIBRATION_FRAMES:
                        calib_global_yaw /= calib_samples
                        calib_global_pitch /= calib_samples
                        calib_head_yaw /= calib_samples
                        calib_head_pitch /= calib_samples
                        calib_head_roll /= calib_samples
                        calibrated = True
                        print(f"✅ Calibrated!")

            if calibrated:
                effective_yaw = global_yaw - calib_global_yaw
                effective_pitch = global_pitch - calib_global_pitch
                rel_head_yaw = head_yaw - calib_head_yaw
                rel_head_pitch = head_pitch - calib_head_pitch

                if is_on_phone:
                    pass

                elif avg_ear < EAR_THRESHOLD:
                    is_distracted = True
                    is_sleeping = True
                
                elif abs(rel_head_yaw) > HARD_HEAD_LIMIT:
                    is_distracted = True
                elif abs(effective_yaw) > GAZE_LIMIT_HORIZONTAL:
                    is_distracted = True
                elif effective_pitch < -GAZE_LIMIT_UP:
                    is_distracted = True
                elif effective_pitch > GAZE_LIMIT_DOWN:
                    is_distracted = True

                if not is_sleeping and not is_on_phone:
                    head_yaw_safe = abs(rel_head_yaw) < COMP_HEAD_LIMIT
                    if head_yaw_safe:
                        if (rel_head_yaw < -20 and eye_yaw_deg > 4):
                            is_distracted = False
                            override_text = "[H-Comp-L]"
                            distraction_start_time = None
                            focused_frame_count = 5
                        if (rel_head_yaw > 20 and eye_yaw_deg < -3):
                            is_distracted = False
                            override_text = "[H-Comp-R]"
                            distraction_start_time = None
                            focused_frame_count = 5
                    
                    head_pitch_safe = abs(rel_head_pitch) < COMP_PITCH_LIMIT
                    if head_pitch_safe:
                        if (rel_head_pitch < -8 and eye_pitch_deg > 8):
                            is_distracted = False
                            override_text = "[V-Comp-D]"
                            distraction_start_time = None
                            focused_frame_count = 5
                        if (rel_head_pitch > 8 and eye_pitch_deg < -8):
                            is_distracted = False
                            override_text = "[V-Comp-U]"
                            distraction_start_time = None
                            focused_frame_count = 5

            debug_text_1 = f"EAR: {avg_ear:.2f} | Phone: {is_on_phone}"
            box_color = (0, 0, 255) if is_sleeping else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

    else:
        if is_on_phone:
             is_distracted = True
        else:
             is_distracted = True
             status_text = "NO FACE"

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

    if distraction_start_time is not None:
        elapsed = time.time() - distraction_start_time
        if elapsed > ALARM_DELAY or is_on_phone:
            if is_sleeping:
                status_text = "WAKE UP!"
            elif is_on_phone:
                status_text = "PHONE DETECTED!"
            else:
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

    cv2.putText(frame, status_text + " " + override_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, debug_text_1, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, calib_info, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Enhanced Focus Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibrated = False
        calib_samples = 0
        print("🔁 Recalibrating...")
    elif key == ord('d'):
        debug_mode = not debug_mode

cap.release()
cv2.destroyAllWindows()
