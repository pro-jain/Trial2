import albumentations as A
import cv2
import numpy as np
import os
import glob

# Define input and output directories
input_image_dir = "final\\trial\\"
input_label_dir = "final\\trial_label\\"
output_image_dir = "final\\images\\augmented1\\"
output_label_dir = "final\\labels\\augmented1\\"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Get all image files
image_files = glob.glob(os.path.join(input_image_dir, "*.jpg")) + \
              glob.glob(os.path.join(input_image_dir, "*.png")) + \
              glob.glob(os.path.join(input_image_dir, "*.jpeg"))

print(f"Found {len(image_files)} images to process")

# Define augmentation with corrected parameters
transform = A.Compose([
    A.Rotate(
        limit=(140, 140),  # Rotate exactly 140 degrees
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    )
],
bbox_params=A.BboxParams(
    format="yolo", 
    label_fields=["labels"],
    min_visibility=0.1,  # Lower threshold to keep more boxes
    clip=False  # Don't clip boxes to image boundaries
))

# Alternative transformation
transform_alt = A.Compose([
    A.Affine(
        rotate=140,
        scale=1.0,
        translate_percent=0.0,
        keep_ratio=True,  # This helps maintain aspect ratios
        fit_output=False,  # Don't resize output image
        p=1.0
    )
],
bbox_params=A.BboxParams(
    format="yolo", 
    label_fields=["labels"],
    min_visibility=0.1,
    clip=True
))

successful_count = 0
failed_count = 0

# Process each image
for image_path in image_files:
    try:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Corresponding label file path
        label_path = os.path.join(input_label_dir, f"{base_name}.txt")
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {base_name}, skipping...")
            continue
            
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            failed_count += 1
            continue
            
        h, w = image.shape[:2]
        
        # Read YOLO-format labels
        bboxes = []
        labels = []
        
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = list(map(float, line.split()))
                    class_id = int(parts[0])
                    x, y, w_box, h_box = parts[1:]
                    bboxes.append([x, y, w_box, h_box])
                    labels.append(class_id)
        
        # Skip if no bboxes found
        if not bboxes:
            print(f"Warning: No bounding boxes found in {base_name}.txt, skipping...")
            continue
        
        # Try primary transformation
        try:
            aug = transform(image=image, bboxes=bboxes, labels=labels)
            aug_img = aug["image"]
            aug_bboxes = aug["bboxes"]
            aug_labels = aug["labels"]
            transform_used = "primary"
            
        except Exception as e:
            print(f"Primary transform failed for {base_name}, trying alternative: {e}")
            # Try alternative transformation
            aug = transform_alt(image=image, bboxes=bboxes, labels=labels)
            aug_img = aug["image"]
            aug_bboxes = aug["bboxes"]
            aug_labels = aug["labels"]
            transform_used = "alternative"
        
        # Save augmented image and label
        output_image_path = os.path.join(output_image_dir, f"{base_name}_aug1.jpg")
        output_label_path = os.path.join(output_label_dir, f"{base_name}_aug1.txt")
        
        # Save image
        cv2.imwrite(output_image_path, aug_img)
        
        # Save label
        with open(output_label_path, "w") as f:
            for cls, bbox in zip(aug_labels, aug_bboxes):
                bbox_str = " ".join(f"{x:.6f}" for x in bbox)
                f.write(f"{int(cls)} {bbox_str}\n")
        
        successful_count += 1
        print(f"✓ Processed {base_name} using {transform_used} transform")
        
    except Exception as e:
        print(f"✗ Failed to process {os.path.basename(image_path)}: {e}")
        failed_count += 1

print(f"\nProcessing completed!")
print(f"Successfully processed: {successful_count} images")
print(f"Failed to process: {failed_count} images")
print(f"Augmented images saved to: {output_image_dir}")
print(f"Augmented labels saved to: {output_label_dir}")