import cv2
import numpy as np
import math
from data_aug.data_aug import *
from data_aug.bbox_util import *
import matplotlib.pyplot as plt


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

img = cv2.imread("final\\final\\images\\train\\142_jpg.rf.16d6c682af57b4a5e1be7c3a80566def.jpg")[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb

h, w = img.shape[:2]
# Load bboxes from txt file
bboxes = load_yolo_annotations("final\\final\\labels\\train\\142_jpg.rf.16d6c682af57b4a5e1be7c3a80566def.txt", w, h)
# Load input image

img_, bboxes_ = Perspective(0, 45, 0, f=2)(img.copy(), bboxes.copy())
# Apply rotation (example: rotate 30° around Y axis)
#rotated_img = rotate_image_3d(img, rotx=50, roty=0, rotz=0, f=2)
new_labels = []

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
    print(yolo_line)
    new_labels.append(yolo_line)

# Write to YOLO label file
with open("trial.txt", "w") as f:
    for line in new_labels:
        f.write(line + "\n")

plotted_img = draw_rect(img_, bboxes_)
# Show or save the result
#cv2.imshow("Rotated Image", img_)
cv2.imwrite("trial.jpg",plotted_img)
plt.imshow(plotted_img)
plt.axis("off")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()





