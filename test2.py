import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("runs\\detect\\train2\\weights\\best.pt")

available_classes = [
    "major_crack",
    "minor_crack",
    "vegetation",
    "peeling",
    "damage",
    "fire"
]

print("Available Defect Types:")
for i, defect in enumerate(available_classes):
    print(f"{i+1}. {defect}")


choice = int(input("Enter the index of the defect type you want to view: ")) - 1
if not 0 <= choice < len(available_classes):
    print("Invalid choice.")
    exit()

selected_class = available_classes[choice]
conf = float(input("Enter the confidence (e.g. 0.25): "))

image_path = "dataset\\images\\val\\33.jpg"
img = cv2.imread(image_path)

results = model(img, imgsz=640, iou=0.4, conf=conf)


names = model.names
selected_class_id = [k for k, v in names.items() if v == selected_class]

if not selected_class_id:
    print("Selected class not found in image.")
    exit()

print(results)
filtered_boxes = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    if cls_id in selected_class_id:
        filtered_boxes.append(box)

results[0].boxes = filtered_boxes
annotated_img = results[0].plot()


resized_output = cv2.resize(annotated_img, (640, 640))

cv2.imshow(f"YOLO Detection - {selected_class}", resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
