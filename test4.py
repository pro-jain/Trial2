import os
import shutil
import glob

# Directories
ptanhi_dir = "final\\ptanhi"    # junk names (augmented output)
aug_dir = "final\\AUG"          # clean names (planned images)
output_dir = "final\\no_aug"    # store unmatched images
os.makedirs(output_dir, exist_ok=True)

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']

# Get clean base names from AUG (planned set)
aug_files = []
for ext in image_extensions:
    aug_files.extend(glob.glob(os.path.join(aug_dir, ext)))
aug_clean_names = set(os.path.splitext(os.path.basename(f))[0] for f in aug_files)

# Get full paths and clean names from ptanhi
ptanhi_files = []
for ext in image_extensions:
    ptanhi_files.extend(glob.glob(os.path.join(ptanhi_dir, ext)))

ptanhi_extra_files = []

for f in ptanhi_files:
    full_name = os.path.splitext(os.path.basename(f))[0]  # e.g., '1400_jpg.rf.xxxx'
    clean_name = full_name.split('_')[0]                  # e.g., '1400'
    
    if clean_name not in aug_clean_names:
        ptanhi_extra_files.append(f)

# Copy unmatched files to output_dir
for f in ptanhi_extra_files:
    dst = os.path.join(output_dir, os.path.basename(f))
    shutil.copy2(f, dst)

print(f"✓ Copied {len(ptanhi_extra_files)} images from ptanhi not found in AUG to '{output_dir}'")
