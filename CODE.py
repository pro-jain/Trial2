
from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")

    results = model.train(
        data="C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\config.yaml",
        epochs=250,
        workers=18,
        amp=True,
        hsv_h=0.015,
        perspective=0.0008,
        flipud=0.445,
        fliplr=0.489,
        mixup=0.145,
        EMA=True
        
         )
    print(model.weights())
if __name__ == "__main__":
    main()
