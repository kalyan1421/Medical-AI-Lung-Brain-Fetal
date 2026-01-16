import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

IMG_SIZE = 256

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def normalize_image(image):
    """Normalize image to 0-255 range."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def clean_and_organize_data(source_dir, output_dir, val_split=0.15, test_split=0.10):
    """
    Clean and organize fetal ultrasound dataset into train/val/test sets.
    
    Args:
        source_dir: Directory containing raw images (e.g., dataset/train)
        output_dir: Base directory for organized data (e.g., dataset)
        val_split: Validation split ratio (default: 0.15)
        test_split: Test split ratio (default: 0.10)
    """
    
    print("=" * 60)
    print("FETAL ULTRASOUND DATA PREPROCESSING")
    print("=" * 60)
    
    # Get all image files (not annotations)
    all_files = [f for f in os.listdir(source_dir) 
                 if f.endswith('_HC.png') and '_Annotation' not in f]
    all_files.sort()
    
    print(f"\n✓ Found {len(all_files)} ultrasound images")
    
    # Split data: train (75%), val (15%), test (10%)
    train_files, temp_files = train_test_split(
        all_files, test_size=(val_split + test_split), random_state=42, shuffle=True
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=test_split/(val_split + test_split), random_state=42
    )
    
    print(f"✓ Train set: {len(train_files)} images")
    print(f"✓ Validation set: {len(val_files)} images")
    print(f"✓ Test set: {len(test_files)} images")
    print()
    
    # Process each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    stats = {'skipped': 0, 'processed': 0}
    
    for split_name, file_list in splits.items():
        print(f"Processing {split_name} set...")
        
        # Create directories
        img_dir = os.path.join(output_dir, split_name, "images")
        mask_dir = os.path.join(output_dir, split_name, "masks")
        ensure_dir(img_dir)
        ensure_dir(mask_dir)
        
        for img_name in tqdm(file_list, desc=f"{split_name.capitalize()}"):
            # Construct paths
            img_path = os.path.join(source_dir, img_name)
            mask_name = img_name.replace('_HC.png', '_HC_Annotation.png')
            mask_path = os.path.join(source_dir, mask_name)
            
            # Check if mask exists
            if not os.path.exists(mask_path):
                print(f"⚠ Mask not found for {img_name}, skipping.")
                stats['skipped'] += 1
                continue
            
            # Read images
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                print(f"⚠ Failed to read {img_name}, skipping.")
                stats['skipped'] += 1
                continue
            
            # Preprocessing
            # 1. Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            
            # 2. Apply CLAHE for better contrast
            img = apply_clahe(img)
            img = normalize_image(img)
            
            # 3. Ensure mask is binary (0 or 255)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # 4. Denoise image
            img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Save processed images
            save_img_path = os.path.join(img_dir, img_name)
            save_mask_path = os.path.join(mask_dir, mask_name)
            
            cv2.imwrite(save_img_path, img)
            cv2.imwrite(save_mask_path, mask)
            
            stats['processed'] += 1
        
        print(f"✓ {split_name.capitalize()} set complete!\n")
    
    # Print summary
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {stats['processed']}")
    print(f"Total images skipped: {stats['skipped']}")
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")
    print("=" * 60)
    print("\n✓ Data preprocessing complete!")
    print(f"✓ Organized data saved to: {output_dir}")
    print("\nNext step: Run train.py to train the model")

if __name__ == "__main__":
    # Configure paths
    SOURCE_DIR = "../dataset/train"  # Raw data directory
    OUTPUT_DIR = "../dataset"         # Output directory
    
    # Run preprocessing
    clean_and_organize_data(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        val_split=0.15,
        test_split=0.10
    )
