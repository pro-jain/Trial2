from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


img = cv2.imread("final\\images\\val\\12.jpg")[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
def load_yolo_annotations(txt_file, img_width, img_height):
    """
    Load YOLO format annotations from txt file and convert to [x1, y1, x2, y2, class_id] format
    
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    Output format: [x1, y1, x2, y2, class_id] (pixel coordinates)
    """
    bboxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
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


img_height, img_width = img.shape[:2]

# Load bboxes from txt file
bboxes = load_yolo_annotations("final\\labels\\val\\12.txt", img_width, img_height)

#inspect the bounding boxes
print("Original bboxes:")
print(bboxes)

# Apply shear
img_, bboxes_ = RandomShear(1.6)(img.copy(), bboxes.copy())

# Prepare new YOLO-format bboxes
new_labels = []

for b in bboxes_:
    x1, y1, x2, y2, class_id = b
    class_id = int(class_id)


    box_w = x2 - x1
    box_h = y2 - y1


    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    norm_x = center_x / img_width
    norm_y = center_y / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    # Round and format the line
    yolo_line = f"{class_id} {norm_x} {norm_y} {norm_w} {norm_h}"
    print(yolo_line)
    new_labels.append(yolo_line)

# Write to YOLO label file
with open("trial.txt", "w") as f:
    for line in new_labels:
        f.write(line + "\n")

# Draw and show bounding boxes
plotted_img = draw_rect(img_, bboxes_)
cv2.imwrite("trial.jpg",plotted_img)
plt.imshow(plotted_img)
plt.axis("off")
plt.show()
