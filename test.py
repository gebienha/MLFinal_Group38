import cv2 as cv
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv.VideoCapture(0)

# Filepath for keypoint.csv
csv_file = r"C:\Users\User\Downloads\hand-gesture-recognition-mediapipe-main\hand-gesture-recognition-mediapipe-main\model\keypoint_classifier\keypoint.csv"

# Ensure CSV file exists
if not os.path.exists(csv_file):
    # Create a new CSV file with headers if it doesn't exist
    with open(csv_file, "w") as file:
        file.write(",".join([f"x{i},y{i},z{i}" for i in range(21)]) + ",label\n")

print("Press 's' to save keypoints, and 'ESC' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert frame
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process frame for hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Show the frame
            cv.putText(frame, "Press 's' to save data", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv.imshow("Frame", frame)

    # Key press handling
    key = cv.waitKey(10) & 0xFF
    if key == ord('s') and result.multi_hand_landmarks:
        label = input("Enter the label for this sign: ")  # Label for the sign
        if keypoints:
            # Append data to the CSV
            with open(csv_file, "a") as file:
                file.write(",".join(map(str, keypoints)) + f",{label}\n")
            print(f"Data for '{label}' saved successfully!")
    elif key == 27:  # ESC key
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
