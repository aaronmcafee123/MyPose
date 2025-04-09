
MyPOSE - Wheelchair Posture Detection Dataset & Scripts
=======================================================

Included Files:
------------------
- manual_posture_labeler.py: Manually label good/bad posture from images.
- blazepose_merge_labels.py: Run BlazePose, extract angles, and merge with manual labels.
- visualize_postures.py: Annotate images with posture keypoints and label overlays.
- train_final_classifier.py: Train a KNN classifier using spine, neck, and shoulder angles.
- README.txt: This file.

Manual Image Folder:
-----------------------
Create a folder called 'MyPOSE_Images' containing 15 wheelchair user images.

Labeled Files You Should Include:
------------------------------------
- manual_posture_labels.csv
- final_dataset_labeled_angles.csv

Summary:
-----------
You can use the final_dataset_labeled_angles.csv for training/testing any real-time webcam system.
Angles: neck, spine, shoulder.
Labels: good / bad.

Made by: V
