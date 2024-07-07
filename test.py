import random
import cv2
import numpy as np
from ultralytics import YOLO

# opening the file in read mode
with open("C:/yolo/coco.txt", "r") as my_file:
    # reading the file and splitting by newline
    class_list = my_file.read().split("\n")

# Generate random colors for each class
detection_colors = []
for _ in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("C:/yolo/runs/detect/train5/weights/best.pt", "v8")

# Load an image
image_path = "C:/yolo/t/371.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Cannot open image")
    exit()

# Predict on the image
detect_params = model.predict(
    source=[frame], conf=0.25, save=False
)  # Lower confidence threshold

# Check and print detection parameters
print("Detection Parameters:")
print(detect_params)

# Convert tensor array to numpy
DP = detect_params[0].numpy()
print("Detection Parameters Array:")
print(DP)

if len(DP) != 0:
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = int(box.cls.numpy()[0])  # Class ID
        conf = box.conf.numpy()[0]  # Confidence score
        bb = box.xyxy.numpy()[0]  # Bounding box

        # Draw rectangle around detected object
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[clsID],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        label = f"{class_list[clsID]} {round(conf, 2)}%"
        cv2.putText(
            frame,
            label,
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )
        print(f"Detected: {label} at {bb}")

else:
    print("No detections were made.")

# Save the resulting frame with a complete file path
result_image_path = "C:/yolo/t/result6.jpg"
cv2.imwrite(result_image_path, frame)
