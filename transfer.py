import os
import shutil
import glob

# Folder with selected images
selected_images_dir = "final\\trial"

# Folder with original labels
input_label_dir = "final\\final\\labels\\train"

# Output folder for matching labels
output_label_dir = "final\\trial_label"
os.makedirs(output_label_dir, exist_ok=True)

# Supported image extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']

# Get all image files in selected_images_dir
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(selected_images_dir, ext)))

print(f"Found {len(image_files)} images. Copying matching labels...")

copied = 0
skipped = 0

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_name = base_name + ".txt"
    src_label_path = os.path.join(input_label_dir, label_name)
    dst_label_path = os.path.join(output_label_dir, label_name)

    if os.path.exists(src_label_path):
        shutil.copy2(src_label_path, dst_label_path)
        copied += 1
    else:
        print(f"Label not found for: {label_name}")
        skipped += 1

print(f"\n Copied {copied} label files.")
if skipped > 0:
    print(f" Skipped {skipped} (label not found).")
