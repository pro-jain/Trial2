import cv2
from ultralytics import YOLO

model = YOLO("yolo11.pt")  

image_path="final\\final\\images\\train\\broken-fence-on-the-beach-at-sunrise-S2PFWG_0_jpg.rf.4a326c0bcc4de2288b7e42686ce7d394.jpg"
img = cv2.imread(image_path)

results = model(image_path,iou=0.5)
annotated_img = results[0].plot()  
resized_output = cv2.resize(annotated_img, (640, 640))

cv2.imshow('YOLO Detections', resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
