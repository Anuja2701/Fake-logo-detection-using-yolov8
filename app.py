from flask import Flask, render_template, Response
import cv2
import random
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load class names and colors
with open("C:/yolo/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

# Load the YOLOv8 model
model = YOLO("C:/yolo/runs/detect/train7/weights/best.pt", "v8")

# Global variable for the video capture
cap = None


# Function to generate frames
def gen_frames():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = int(box.cls.numpy()[0])
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[clsID],
                    3,
                )
                label = f"{class_list[clsID]} {round(conf, 2)}%"
                cv2.putText(
                    frame,
                    label,
                    (int(bb[0]), int(bb[1]) - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
