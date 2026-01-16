"""
Test script to verify all models load correctly
"""

import os
import sys

# Add custom objects for segmentation model
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation evaluation."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """Dice loss function."""
    return 1 - dice_coef(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union) score."""
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

CUSTOM_OBJECTS = {
    'dice_coef': dice_coef,
    'dice_coef_loss': dice_coef_loss,
    'iou_score': iou_score
}

from tensorflow.keras.models import load_model

MODELS_CONFIG = {
    'pneumonia': {
        'model_path': 'Lung/models/lung_model_final_20260113_125327.h5',
        'model_type': 'classification',
        'architecture': 'EfficientNetB3'
    },
    'brain_tumor': {
        'model_path': 'brain_tumor/models/brain_tumor_final.h5',
        'model_type': 'classification',
        'architecture': 'EfficientNetV2S'
    },
    'fetal_ultrasound': {
        'model_path': 'Fetal_Ultrasound/training/fetal_ultrasound_unet_20260114_122846_best.h5',
        'model_type': 'segmentation',
        'architecture': 'U-Net'
    }
}

def test_model_loading():
    """Test if all models can be loaded successfully"""
    print("\n" + "="*60)
    print(" 🧪 TESTING MODEL LOADING")
    print("="*60)
    
    results = {}
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n📦 Testing {model_name.upper()} model...")
        print(f"   📂 Path: {config['model_path']}")
        print(f"   🏗️  Architecture: {config['architecture']}")
        print(f"   🎯 Type: {config['model_type']}")
        
        try:
            # Check if file exists
            if not os.path.exists(config['model_path']):
                print(f"   ❌ Model file not found!")
                results[model_name] = False
                continue
            
            # Load model
            if config['model_type'] == 'segmentation':
                model = load_model(config['model_path'], custom_objects=CUSTOM_OBJECTS, compile=False)
            else:
                model = load_model(config['model_path'], compile=False)
            
            # Get model info
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            print(f"   ✅ Loaded successfully!")
            print(f"   📊 Input shape: {input_shape}")
            print(f"   📊 Output shape: {output_shape}")
            
            results[model_name] = True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            results[model_name] = False
    
    # Summary
    print("\n" + "="*60)
    print(" 📊 TEST SUMMARY")
    print("="*60)
    
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {model_name.upper()}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n   Total: {passed}/{total} models loaded successfully")
    
    if passed == total:
        print("\n   🎉 ALL MODELS LOADED SUCCESSFULLY!")
        return True
    else:
        print("\n   ⚠️  SOME MODELS FAILED TO LOAD")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
