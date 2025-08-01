import cv2
import numpy as np
import math
from data_aug.data_aug import *
from data_aug.bbox_util import *
import matplotlib.pyplot as plt
import os
import glob
import shutil


input_image_dir = "valid"
input_label_dir = "valid"
output_image_dir = "valid\\rotated\\"
output_label_dir = "valid\\rotated1\\"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

image_files = glob.glob(os.path.join(input_image_dir, "*.jpg")) + \
              glob.glob(os.path.join(input_image_dir, "*.png")) + \
              glob.glob(os.path.join(input_image_dir, "*.jpeg")) +\
              glob.glob(os.path.join(input_image_dir, "*.webp"))
print(f"Found {len(image_files)} images to process")
successful_count = 0
failed_count = 0

def load_yolo_annotations(txt_file, img_width, img_height):
    """
    Load YOLO format annotations from txt file and convert to [x1, y1, x2, y2, class_id] format
    
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    Output format: [x1, y1, x2, y2, class_id] (pixel coordinates)
    """
    bboxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f" Skipping polygon/segment line in {txt_file}")
                continue  # skip polygon or segment line
            class_id = int(parts[0])
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to x1, y1, x2, y2 format
            x1 = center_x - width/2
            y1 = center_y - height/2
            x2 = center_x + width/2
            y2 = center_y + height/2
            
            bboxes.append([x1, y1, x2, y2, int(class_id)])

    return np.array(bboxes)

for image_path in image_files:
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Read image
        img = cv2.imread(image_path)[:,:,::-1] 
        if img is None:
            print(f"Error: Could not read image {image_path}")
            failed_count += 1
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(input_label_dir, f"{base_name}.txt")
        bboxes = load_yolo_annotations(label_path, w, h)
        img_, bboxes_ = Perspective(1,35, 0, f=2)(img.copy(), bboxes.copy())
        
        new_labels = []

        bboxes = load_yolo_annotations(label_path, w, h)
        
        for b in bboxes_:
            x1, y1, x2, y2, class_id = b
            class_id = int(class_id)


            box_w = x2 - x1
            box_h = y2 - y1


            center_x = x1 + box_w / 2
            center_y = y1 + box_h / 2

            norm_x = center_x / w
            norm_y = center_y / h
            norm_w = box_w / w
            norm_h = box_h / h

            # Round and format the line
            yolo_line = f"{class_id} {norm_x} {norm_y} {norm_w} {norm_h}"
            new_labels.append(yolo_line)
            output_label_path = os.path.join(output_label_dir, f"{base_name}_skew1.txt")

        with open(output_label_path, "w") as f:
                    for line in new_labels:
                        f.write(line + "\n")
        output_image_path = os.path.join(output_image_dir, f"{base_name}_skew1.jpg")
        img_=cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
        plotted_img = draw_rect(img_, bboxes_)
        cv2.imwrite(output_image_path,plotted_img)
        successful_count += 1
        print(f"✓ Processed {base_name}")
                
    except Exception as e:
        print(f"✗ Failed to process {os.path.basename(image_path)}: {e}")
        failed_count += 1
print("failed: ",failed_count)
print("success: ",successful_count)




