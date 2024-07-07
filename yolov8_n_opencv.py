import random
import cv2
import numpy as np
from ultralytics import YOLO

# opening the file in read mode
with open("C:/yolo/coco.txt", "r") as my_file:
    # reading the file and splitting by newline
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = []
for _ in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("C:/yolo/runs/detect/train7/weights/best.pt", "v8")

# Values to resize video frames
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)  # Open the camera


if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
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

            # Check if a specific condition is met to close the camera
            # Here we can use the class name as an example
            if class_list[clsID] == "desired_class_name":
                print("Desired object detected. Closing camera.")
                cap.release()
                cv2.destroyAllWindows()
                exit()  # Exit the program

    # Display the resulting frame
    cv2.imshow("FakeLogoDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
