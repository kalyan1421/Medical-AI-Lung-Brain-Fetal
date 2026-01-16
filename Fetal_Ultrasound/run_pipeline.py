#!/usr/bin/env python3
"""
Complete pipeline script for Fetal Ultrasound Segmentation
Run this to execute the entire training pipeline automatically.
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Execute a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"▶ {description}")
    print("=" * 70)
    print()
    
    try:
        # Run without capturing output so it shows in real-time
        result = subprocess.run(
            command,
            shell=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: Command failed with exit code {e.returncode}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    try:
        import tensorflow
        import cv2
        import numpy
        import sklearn
        import tqdm
        import matplotlib
        
        print("✓ TensorFlow version:", tensorflow.__version__)
        print("✓ OpenCV version:", cv2.__version__)
        print("✓ NumPy version:", numpy.__version__)
        print("✓ All dependencies installed!")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\n⚠ Please install requirements first:")
        print("   pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if dataset exists."""
    print("\n" + "=" * 70)
    print("CHECKING DATASET")
    print("=" * 70)
    
    raw_data_path = "dataset/train"
    
    if not os.path.exists(raw_data_path):
        print(f"✗ Dataset not found at: {raw_data_path}")
        print("\n⚠ Please ensure your dataset is in the correct location:")
        print(f"   {raw_data_path}/")
        print("   ├── 000_HC.png")
        print("   ├── 000_HC_Annotation.png")
        print("   ├── 001_HC.png")
        print("   └── ...")
        return False
    
    # Count images
    files = os.listdir(raw_data_path)
    images = [f for f in files if f.endswith('_HC.png') and '_Annotation' not in f]
    annotations = [f for f in files if f.endswith('_Annotation.png')]
    
    print(f"✓ Found {len(images)} images")
    print(f"✓ Found {len(annotations)} annotations")
    
    if len(images) == 0:
        print("✗ No images found!")
        return False
    
    if len(images) != len(annotations):
        print(f"⚠ Warning: Number of images and annotations don't match!")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Fetal Ultrasound Segmentation Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip data preprocessing (if already done)')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training (if model exists)')
    parser.add_argument('--skip-evaluation', action='store_true', 
                       help='Skip evaluation')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("FETAL ULTRASOUND SEGMENTATION - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n✗ Pipeline aborted: Missing dependencies")
        sys.exit(1)
    
    # Step 2: Check dataset
    if not check_dataset():
        print("\n✗ Pipeline aborted: Dataset not found")
        sys.exit(1)
    
    # Step 3: Preprocessing
    if not args.skip_preprocessing:
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 70)
        
        if not run_command(
            "cd preprocessing && python clean_dataset.py",
            "Running data preprocessing..."
        ):
            print("\n✗ Preprocessing failed!")
            sys.exit(1)
    else:
        print("\n⏭ Skipping preprocessing (as requested)")
    
    # Step 4: Training
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("STEP 2: MODEL TRAINING")
        print("=" * 70)
        
        if not run_command(
            "cd training && python train.py",
            "Training Attention U-Net model..."
        ):
            print("\n✗ Training failed!")
            sys.exit(1)
    else:
        print("\n⏭ Skipping training (as requested)")
    
    # Step 5: Evaluation
    if not args.skip_evaluation:
        print("\n" + "=" * 70)
        print("STEP 3: MODEL EVALUATION")
        print("=" * 70)
        
        if not run_command(
            "cd training && python evaluate.py",
            "Evaluating model on test set..."
        ):
            print("\n✗ Evaluation failed!")
            sys.exit(1)
    else:
        print("\n⏭ Skipping evaluation (as requested)")
    
    # Success!
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("   - Best model: training/fetal_ultrasound_*_best.h5")
    print("   - Training plots: training/*_training_history.png")
    print("   - Evaluation results: training/evaluation_results/")
    print("\n📊 Next Steps:")
    print("   1. Check evaluation_results/ for model performance")
    print("   2. View TensorBoard logs: tensorboard --logdir=training/logs/")
    print("   3. Deploy the model for inference")
    print()

if __name__ == "__main__":
    main()
