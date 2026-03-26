from ultralytics import YOLO

def main():
    model = YOLO("model/best.pt")
    metrics = model.val(
        data="dataset/data.yaml",
        imgsz=416,
        batch=4,
        device="cpu"
    )
    print(metrics)

if __name__ == "__main__":
    main()