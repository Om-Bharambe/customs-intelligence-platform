from ultralytics import YOLO

def main():
    model = YOLO("model/best.pt")
    model.predict(
        source="dataset/test/images",
        conf=0.25,
        save=True,
        project="outputs/predictions",
        name="run",
        device="cpu"
    )

if __name__ == "__main__":
    main()