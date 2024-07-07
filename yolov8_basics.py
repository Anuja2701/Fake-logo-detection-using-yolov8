from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="C:/yolo/yolov8.yaml",  # Path to dataset YAML file
    epochs=50,  # Number of epochs to train for
    imgsz=640,  # Image size
    batch=16,  # Batch size
)
results = model.val()
# results = model("C:/yolo/t/364.jpg")
# results.show()  # To display the image with bounding boxes
