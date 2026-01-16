import cv2
import os
import numpy as np

def check_masks(mask_dir, num_samples=10):
    """Check if masks have reasonable amount of positive pixels."""
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')][:num_samples]
    
    print(f"\nChecking {len(mask_files)} masks from {mask_dir}...")
    print("=" * 60)
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"✗ {mask_file}: Could not read!")
            continue
        
        # Calculate statistics
        total_pixels = mask.shape[0] * mask.shape[1]
        positive_pixels = np.sum(mask > 127)
        percentage = (positive_pixels / total_pixels) * 100
        
        status = "✓" if percentage > 0.5 else "✗"
        print(f"{status} {mask_file}: {positive_pixels:,} pixels ({percentage:.2f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    # Check train masks
    train_mask_dir = "dataset/train/masks"
    if os.path.exists(train_mask_dir):
        check_masks(train_mask_dir)
    else:
        print(f"\n✗ Directory not found: {train_mask_dir}")
        print("   Run preprocessing first!")
    
    # Check val masks
    val_mask_dir = "dataset/val/masks"
    if os.path.exists(val_mask_dir):
        check_masks(val_mask_dir)
    else:
        print(f"\n✗ Directory not found: {val_mask_dir}")
