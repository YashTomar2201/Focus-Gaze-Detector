# main.py

import cv2
import mediapipe as mp
import numpy as np
import time
import pygame # New: Import the pygame library

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# New: Initialize pygame mixer
pygame.mixer.init()
# New: Load your alarm sound
alarm_sound = pygame.mixer.Sound(r'd:\ML project\project\alarm.wav')


# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Variables for distraction detection
distraction_timer = None
distraction_threshold = 3.0
status = "Focused"
# New: Renamed flag for clarity with the new library
is_alarm_playing = False 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape
    
    is_distracted = False 

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
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

            if y < -15 or y > 14 or x < -12 or x > 10:
                is_distracted = True
    else:
        is_distracted = True

    if is_distracted:
        if distraction_timer is None:
            distraction_timer = time.time()
        elif time.time() - distraction_timer > distraction_threshold:
            status = "Distracted"
            # New: Logic to play the looping alarm
            if not is_alarm_playing:
                alarm_sound.play(loops=-1) # loops=-1 will make it play forever until stopped
                is_alarm_playing = True
    else:
        distraction_timer = None
        status = "Focused"
        # New: Logic to stop the alarm
        if is_alarm_playing:
            alarm_sound.stop()
            is_alarm_playing = False

    status_color = (0, 0, 255) if status == "Distracted" else (0, 255, 0)
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    cv2.imshow('Distraction Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()