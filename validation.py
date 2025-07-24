import os
import shutil
import re

# Configuration: update these paths as needed
LOG_FILE = r"C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\annotation_errors.txt"
SRC_DIR = r"C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\final\\Clean\\images\\train"
DST_DIR = r"C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\final\\trial\\7line\\image"

# Ensure destination folder exists
os.makedirs(DST_DIR, exist_ok=True)

# Regex to extract filename after 'train\' prefix
pattern = re.compile(r"train\\([^: \n]+)")

# Gather all image files in SRC_DIR into a dict: stem -> full path
img_files = {}
for fname in os.listdir(SRC_DIR):
    stem, ext = os.path.splitext(fname)
    if ext.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
        img_files[stem] = os.path.join(SRC_DIR, fname)

# Process each line in the log file
copied_count = 0
with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue
        label_fname = match.group(1)  # e.g. "Rissbilder_for_Florian_9S6A2804_317_223_3516_3608.txt"
        label_stem, _ = os.path.splitext(label_fname)

        # Direct exact match
        if label_stem in img_files:
            shutil.copy2(img_files[label_stem], DST_DIR)
            copied_count += 1
            print(f"Exact match copied: {img_files[label_stem]}")
            continue
        
        # Try prefix matching: find any image stem that's a prefix of label_stem
        found = False
        for img_stem, img_path in img_files.items():
            if label_stem.startswith(img_stem + "_"):
                shutil.copy2(img_path, DST_DIR)
                copied_count += 1
                print(f"Prefix match copied: {img_path}")
                found = True
                break
        
        if not found:
            print(f"No image found for label: {label_fname}")

print(f"\nDone! Total copied: {copied_count}")
