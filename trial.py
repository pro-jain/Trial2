from ultralytics import YOLO
def main():

    model = YOLO("yolo11m.pt")

    results= model.train(data="C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\config.yaml",epochs=120,workers=16,amp=True,hsv_h=0.015,perspective=0.001,flipud=0.5,fliplr=0.5,mixup=0.1)

if __name__=="__main__":
    main() 