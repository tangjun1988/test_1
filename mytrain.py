from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"kunkun_xz.yaml",
        epochs=100,
        imgsz=640,
        batch=2,
        cache="ram",
        workers=1,
    )
