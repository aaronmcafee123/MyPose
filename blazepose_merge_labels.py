
import cv2
import mediapipe as mp
import pandas as pd
import os
import math

# Configuration
input_folder = 'C:/Users/Dell/OneDrive/Desktop/MyPOSE_Images'
label_csv = 'manual_posture_labels.csv'
output_csv = 'final_dataset_labeled_angles.csv'

# Load labels
labels_df = pd.read_csv(label_csv)
labels_df = labels_df.set_index('filename')

# BlazePose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calculate_angle(a, b, c):
    try:
        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]
        dot_product = sum(ba[i]*bc[i] for i in range(3))
        mag_ba = math.sqrt(sum(ba[i]**2 for i in range(3)))
        mag_bc = math.sqrt(sum(bc[i]**2 for i in range(3)))
        if mag_ba == 0 or mag_bc == 0:
            return None
        cos_angle = dot_product / (mag_ba * mag_bc)
        angle = math.acos(max(min(cos_angle, 1), -1))
        return math.degrees(angle)
    except:
        return None

# Prepare dataset
rows = []

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks or filename not in labels_df.index:
        continue

    landmarks = results.pose_landmarks.landmark

    def pt(i): return [landmarks[i].x, landmarks[i].y, landmarks[i].z]

    try:
        left_shoulder, right_shoulder = pt(11), pt(12)
        left_hip, right_hip = pt(23), pt(24)
        nose = pt(0)
        mid_shoulder = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]
        mid_hip = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]

        neck_angle = calculate_angle(left_shoulder, mid_shoulder, nose)
        spine_angle = calculate_angle(nose, mid_shoulder, mid_hip)
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, [right_shoulder[0]+1, right_shoulder[1], right_shoulder[2]])

        label = labels_df.loc[filename]['posture_label']
        rows.append([filename, neck_angle, spine_angle, shoulder_angle, label])

    except Exception as e:
        print(f"⚠️ Skipped {filename} due to error: {e}")

pose.close()

# Save final merged dataset
final_df = pd.DataFrame(rows, columns=["filename", "neck_angle", "spine_angle", "shoulder_angle", "posture_label"])
final_df.to_csv(output_csv, index=False)
print(" Final labeled dataset with angles saved to:", output_csv)
