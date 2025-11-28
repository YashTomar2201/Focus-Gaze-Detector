# main.py

import cv2

# Initialize the webcam. '0' is the default camera.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Loop to capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # If the frame was not captured successfully, break the loop
    if not ret:
        break

    # Display the frame in a window
    cv2.imshow('Distraction Detector', frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()