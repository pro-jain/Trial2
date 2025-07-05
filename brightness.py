import cv2
import os
import glob
import shutil

# Define input and output directories
input_image_dir = "data\\smoke\\image"
input_label_dir = "data\\smoke\\labels"
output_image_dir = "data\\images\\augmented\\"
output_label_dir = "data\\labels\\augmented\\"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)


image_files = glob.glob(os.path.join(input_image_dir, "*.jpg")) + \
              glob.glob(os.path.join(input_image_dir, "*.png")) + \
              glob.glob(os.path.join(input_image_dir, "*.jpeg"))

print(f"Found {len(image_files)} images to process")

successful_count = 0
failed_count = 0


for image_path in image_files:
    try:
       
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            failed_count += 1
            continue
        
        bright_image = cv2.convertScaleAbs(image, alpha=1.35, beta=0)
        dark_image = cv2.convertScaleAbs(image, alpha=0.65, beta=0)
        
        # Save brightened image
        output_image_path = os.path.join(output_image_dir, f"{base_name}_bright.jpg")
        output_image_path2 = os.path.join(output_image_dir, f"{base_name}_dark.jpg")
        cv2.imwrite(output_image_path, bright_image)
        cv2.imwrite(output_image_path2, dark_image)
        
        # Copy corresponding label file (same content)
        label_path = os.path.join(input_label_dir, f"{base_name}.txt")
        output_label_path = os.path.join(output_label_dir, f"{base_name}_bright.txt")
        output_label_path2 = os.path.join(output_label_dir, f"{base_name}_dark.txt")
        
        if os.path.exists(label_path):
            shutil.copy2(label_path, output_label_path)
            shutil.copy2(label_path, output_label_path2)
        else:
            print(f"Warning: Label file not found for {base_name}")
        
        successful_count += 1
        print(f"✓ Processed {base_name}")
        
    except Exception as e:
        print(f"✗ Failed to process {os.path.basename(image_path)}: {e}")
        failed_count += 1

print(f"\nProcessing completed!")
print(f"Successfully processed: {successful_count} images")
print(f"Failed to process: {failed_count} images")
print(f"Brightened images saved to: {output_image_dir}")
print(f"Copied labels saved to: {output_label_dir}")