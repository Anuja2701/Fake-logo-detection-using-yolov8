import random
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score


# Function to calculate metrics
def calculate_metrics(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    return precision, recall, f1


# Load the class labels
with open("C:/yolo/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Load the trained YOLOv8 model
model = YOLO("C:/yolo/runs/detect/train7/weights/best.pt", "v8")

# Load validation images and annotations
validation_images = [
    "C:/yolo/data/train/images/366.jpg"
    # Add more image paths as needed
]
validation_annotations = [
    "C:/yolo/data/train/labels/366.txt"
]  # Replace with actual paths to annotation files


def read_annotation(file_path):
    with open(file_path, "r") as file:
        annotations = file.readlines()
    labels = [int(line.split()[0]) for line in annotations]
    return labels


true_labels = []
pred_labels = []

for img_path, ann_path in zip(validation_images, validation_annotations):
    # Read image
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error reading image: {img_path}")
        continue

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = int(box.cls.numpy()[0])
            pred_labels.append(clsID)

    # Extract true labels from annotation file
    labels = read_annotation(ann_path)
    true_labels.extend(labels)

# Calculate metrics
precision, recall, f1 = calculate_metrics(true_labels, pred_labels)

# Print metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
