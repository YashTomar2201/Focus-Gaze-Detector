# main.py

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Variables for distraction detection
distraction_timer = None
distraction_threshold = 3.0  # 3 seconds for sustained distraction
status = "Focused"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time() # For FPS calculation
    
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    is_distracted = False # Assume not distracted by default

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            # Tune these thresholds based on your observations from Step 4
            if y < -10 or y > 10 or x < -10:
                is_distracted = True

    # Implement logic for sustained distraction [cite: 5]
    if is_distracted:
        if distraction_timer is None:
            # Start the timer if distraction is detected for the first time
            distraction_timer = time.time()
        elif time.time() - distraction_timer > distraction_threshold:
            status = "Distracted"
    else:
        # Reset the timer and status if focused
        distraction_timer = None
        status = "Focused"

    # Display the status on the frame
    status_color = (0, 0, 255) if status == "Distracted" else (0, 255, 0)
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    cv2.imshow('Distraction Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()