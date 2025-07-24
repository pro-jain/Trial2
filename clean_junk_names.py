import os
import shutil
import glob

aug_dir = "C:\\Users\\GNC_LAB_5\\Downloads\\bdi.v1i.yolov11\\train\\labels"
output_dir = "C:\\Users\\GNC_LAB_5\\Downloads\\bdi.v1i.yolov11\\train"
os.makedirs(output_dir, exist_ok=True)

image_extensions = ['*.txt']

aug_files = []
for ext in image_extensions:
    aug_files.extend(glob.glob(os.path.join(aug_dir, ext)))

def clean_filename(filename):
    name, ext = os.path.splitext(filename)
    lower_name = name.lower()

    for ext_str in ['_jpg', '_jpeg', '_png', '_webp']:
        if ext_str in lower_name:
            index = lower_name.rfind(ext_str)
            base = name[:index]
            suffix = name[index + 1:]

            if 'skew' in suffix:
                base += '_skew'
            elif 'aug' in suffix:
                base += '_aug'
            return base + ext
    return filename

copied_count = 0
for file_path in aug_files:
    original_filename = os.path.basename(file_path)
    cleaned_filename = clean_filename(original_filename)

    dst = os.path.join(output_dir, cleaned_filename)
    counter = 1
    while os.path.exists(dst):
        name, ext = os.path.splitext(cleaned_filename)
        dst = os.path.join(output_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.copy2(file_path, dst)
    copied_count += 1

    if original_filename != cleaned_filename:
        print(f"Cleaned: {original_filename} → {cleaned_filename}")

print(f"✓ Copied {copied_count} files with cleaned names")
print(f"✓ Output directory: {output_dir}")
