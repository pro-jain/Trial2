import cv2
import numpy as np
import math
from data_aug.data_aug import *
from data_aug.bbox_util import *

def rotate_image_3d(img, rotx=0, roty=0, rotz=0, f=2.0):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    # Convert degrees to radians
    rotx = math.radians(rotx)
    roty = math.radians(roty)
    rotz = math.radians(rotz)

    cosx, sinx = math.cos(rotx), math.sin(rotx)
    cosy, siny = math.cos(roty), math.sin(roty)
    cosz, sinz = math.cos(rotz), math.sin(rotz)

    roto = np.array([
        [cosz * cosy, cosz * siny * sinx - sinz * cosx],
        [sinz * cosy, sinz * siny * sinx + cosz * cosx],
        [-siny,       cosy * sinx]
    ])

    corners = np.array([
        [-cx, -cy],
        [ cx, -cy],
        [ cx,  cy],
        [-cx,  cy]
    ])  # Corners according to the center of image

    projected = []
    for pt in corners:
        x, y = pt
        z = x * roto[2, 0] + y * roto[2, 1]
        denom = f * h + z
        px = cx + (x * roto[0, 0] + y * roto[0, 1]) * f * h / denom
        py = cy + (x * roto[1, 0] + y * roto[1, 1]) * f * h / denom
        projected.append([px, py])

    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(projected, dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, H, (w, h))

    return warped

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

img = cv2.imread("final\\images\\val\\12.jpg")[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb

h, w = img.shape[:2]
# Load bboxes from txt file
bboxes = load_yolo_annotations("final\\labels\\val\\12.txt", w, h)
# Load input image
img = cv2.imread("dataset\\images\\cls00_312.jpg")  # Replace with your path

img_, bboxes_ = Perspective(img.copy(), bboxes.copy(),rotx=50, roty=0, rotz=0, f=2)
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

# Show or save the result
cv2.imshow("Rotated Image", img_)
cv2.waitKey(0)
cv2.destroyAllWindows()
