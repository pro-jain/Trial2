import cv2
import time
import os
from collections import defaultdict
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs\\detect\\train36\\weights\\best.pt")

# Set RTSP stream
'''cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
class_counts = defaultdict(int)
output_dir="C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train36\\livefeed\\"
with open("results.csv", "a") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ct = time.time()
        results = model(frame, iou=0.6, stream=True)

        for result in results:
            # Annotated image
            annotated_img = result.plot()
            for box in result.boxes:
               
                cls_id = int(box.cls[0])
                cls = model.names[cls_id]

                class_counts[cls] += 1
                count = class_counts[cls]

                filename = f"{cls}_{count}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, annotated_img)

                f.write(f"{ct},{cls},{filepath}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()'''
class_counts = defaultdict(int)
output_dir="C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train36\\livefeed\\"

image_path="test\\crack37.png"
img = cv2.imread(image_path)

results = model(image_path,iou=0.6 )     
annotated_img = results[0].plot()  
resized_output = cv2.resize(annotated_img, (640, 640))
with open("results.csv", "a") as f:
    for result in results:
                # Annotated image
        annotated_img = result.plot()
        for box in result.boxes:
                        ct = time.time()
                        
        
                        
                            
                        cls_id = int(box.cls[0])
                        cls = model.names[cls_id]

                        class_counts[cls] += 1
                        count = class_counts[cls]

                        filename = f"{cls}_{count}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, annotated_img)

                        f.write(f"{ct},{cls},{filepath}\n")
cv2.imshow('YOLO Detections', resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

