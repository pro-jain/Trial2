import cv2
from ultralytics import YOLO

model = YOLO("c:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train33\\weights\\best.pt")  

image_path="test\\fence13.jpg"
img = cv2.imread(image_path)

results = model(image_path,iou=0.6 )     
annotated_img = results[0].plot()  
resized_output = cv2.resize(annotated_img, (640, 640))


cv2.imshow('YOLO Detections', resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
