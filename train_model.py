
from ultralytics import YOLO

# Load a small YOLO segmentation model
model = YOLO("yolo11n-seg.pt")

# Start a short training run on the CPU
model.train(
    data="dataset_yolo/data.yaml",
    epochs=30,
    imgsz=416,
    batch=2,
    device="cpu",
    workers=0,
    project="runs/fruit_segmentation",
    name="second_training"
)