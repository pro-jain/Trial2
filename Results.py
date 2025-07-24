import os
import shutil
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_dataset(model, dataset_dir, output_name):
    """
    Run predictions on dataset_dir, then split images into detected/background,
    save text files and copy images accordingly.
    """
    try:
        logger.info(f"Running predictions on {dataset_dir} ...")
        
        # Run predictions, save results in runs/detect/output_name
        results = model.predict(
            source=dataset_dir,
            save=True,
            save_txt=True,
            save_conf=True,
            project="runs\\detect",
            name=output_name,
            exist_ok=True,
            conf=0.25
        )
        
        output_dir = os.path.join("runs", "detect", output_name)
        
        # Look for labels directory with txt files
        possible_label_dirs = [
            os.path.join(output_dir, "labels"),
            os.path.join(output_dir, "crops"),
            output_dir
        ]
        
        label_dir = None
        for dir_path in possible_label_dirs:
            if os.path.exists(dir_path):
                txt_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
                if txt_files:
                    label_dir = dir_path
                    logger.info(f"Found label directory: {label_dir}")
                    break
        
        if label_dir is None:
            logger.error(f"No label directory with txt files found in {output_dir}")
            return
        
        # Get all image files in dataset_dir
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = [f for f in os.listdir(dataset_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        logger.info(f"Found {len(all_images)} images in {dataset_dir}")
        
        detected_images = []
        background_images = []
        
        for img_name in all_images:
            label_file = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
            if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
                detected_images.append(img_name)
            else:
                background_images.append(img_name)
        
        # Save txt files listing background and detected images
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        bg_txt_path = os.path.join(output_dir, "background_images.txt")
        det_txt_path = os.path.join(output_dir, "detected_images.txt")
        
        with open(bg_txt_path, "w") as f:
            for img in background_images:
                f.write(img + "\n")
        
        with open(det_txt_path, "w") as f:
            for img in detected_images:
                f.write(img + "\n")
        
        logger.info(f"Saved background image list to: {bg_txt_path}")
        logger.info(f"Saved detected image list to: {det_txt_path}")
        
        # Copy images to separate folders
        bg_image_dir = os.path.join(output_dir, "background_images")
        det_image_dir = os.path.join(output_dir, "detected_images")
        os.makedirs(bg_image_dir, exist_ok=True)
        os.makedirs(det_image_dir, exist_ok=True)
        
        for img in background_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(bg_image_dir, img)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                logger.error(f"Failed copying background image {img}: {e}")
        
        for img in detected_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(det_image_dir, img)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                logger.error(f"Failed copying detected image {img}: {e}")
        
        logger.info(f"Copied background images to {bg_image_dir}")
        logger.info(f"Copied detected images to {det_image_dir}")
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_dir}: {e}")
        raise


def main():
    # Paths to your model and datasets (update these)
    model_path = r"C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train36\\weights\\best.pt"
    train_dir = r"C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\final\\Clean\\images\\train"
    val_dir = r"C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\final\\Clean\\images\\val"
    
    try:
        # Load trained model
        logger.info("Loading model...")
        model = YOLO(model_path)
        
        # Process train dataset
        process_dataset(model, train_dir, output_name="train_results")
        
        # Process validation dataset
        process_dataset(model, val_dir, output_name="val_results")
        
        logger.info("All done!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
