
import cv2
import mediapipe as mp
import pandas as pd
import os

# Config
input_folder = 'Images_data'  # Change if your folder is named differently
output_folder = 'annotated_output'
os.makedirs(output_folder, exist_ok=True)

# Load the labeled CSV
df = pd.read_csv('blazepose_labeled_postures.csv')
angle_cols = ['neck_angle', 'spine_angle', 'posture_label']
df = df.set_index('filename')

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        continue

    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Overlay angles and label
    if filename in df.index:
        info = df.loc[filename]
        label = info['posture_label']
        neck = info['neck_angle']
        spine = info['spine_angle']
        text = f"Posture: {label.upper()} | Neck: {neck:.1f}° | Spine: {spine:.1f}°"

        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if label == 'good' else (0, 0, 255), 2)

    # Save annotated image
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, image)

print(" Annotated images saved to:", output_folder)
