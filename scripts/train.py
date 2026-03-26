from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=10,
        imgsz=416,
        batch=4,
        project="outputs",
        name="customs_detector",
        device="cpu"
    )

if __name__ == "__main__":
    main()