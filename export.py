from ultralytics import YOLO

# Load a model

model = YOLO("C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train33\\weights\\best.pt")

# Export the model
model.export(format="onnx")