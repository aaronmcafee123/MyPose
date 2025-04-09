
import cv2
import os
import csv

input_folder = 'C:/Users/Dell/OneDrive/Desktop/MyPOSE_Images'  # Change this if needed
output_csv = 'manual_posture_labels.csv'

# Supported formats
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'posture_label'])

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f" Could not open: {img_file}")
            continue

        cv2.imshow("Label Posture: Press 'g' for Good, 'b' for Bad", image)
        key = cv2.waitKey(0)

        if key == ord('g'):
            label = 'good'
        elif key == ord('b'):
            label = 'bad'
        else:
            print(" Skipped:", img_file)
            continue

        writer.writerow([img_file, label])
        print(f" Labeled {img_file} as {label}")

cv2.destroyAllWindows()
print(" Labeling complete. Saved to", output_csv)
