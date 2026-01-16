# app.py - Multi-Disease Medical AI Diagnostic System

import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow.keras.backend as K

# --- Configuration ---
# Model Paths
MODELS_CONFIG = {
    'pneumonia': {
        'model_path': 'Lung/models/lung_model_final_20260113_125327.h5',
        'labels': ['NORMAL', 'PNEUMONIA'],
        'img_size': 320,
        'model_type': 'classification',
        'description': 'Analyzes chest X-ray images to detect pneumonia with 74% accuracy using EfficientNetB3',
        'accuracy': 0.74,
        'architecture': 'EfficientNetB3'
    },
    'brain_tumor': {
        'model_path': 'brain_tumor/models/brain_tumor_final.h5',
        'labels': ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor (Healthy)', 'Pituitary Tumor'],
        'img_size': 224,
        'model_type': 'classification',
        'description': 'Detects brain tumors from MRI scans with 92% accuracy. Classifies into Glioma, Meningioma, Pituitary tumors, or healthy brain.',
        'accuracy': 0.92,
        'architecture': 'EfficientNetV2S'
    },
    'fetal_ultrasound': {
        'model_path': 'Fetal_Ultrasound/training/fetal_ultrasound_unet_20260114_122846_best.h5',
        'labels': ['Fetal Head Contour Segmentation'],
        'img_size': 256,
        'model_type': 'segmentation',
        'description': 'Performs semantic segmentation to detect and outline fetal head in ultrasound images using U-Net architecture',
        'dice_coefficient': 0.285,  # Based on training progress
        'architecture': 'U-Net'
    }
}

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global dictionary to store loaded models
models = {}

# ==================== CUSTOM METRICS FOR SEGMENTATION ====================

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

# Custom objects for loading segmentation models
CUSTOM_OBJECTS = {
    'dice_coef': dice_coef,
    'dice_coef_loss': dice_coef_loss,
    'iou_score': iou_score
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_all_models():
    """Loads all ML models into global memory on server start."""
    print("\n" + "="*60)
    print(" 🏥 Medical AI Diagnostic System - Loading Models")
    print("="*60)
    
    for disease_key, config in MODELS_CONFIG.items():
        print(f"\n📦 Loading {disease_key.replace('_', ' ').title()} model...")
        print(f"   📂 Path: {config['model_path']}")
        print(f"   🏗️  Architecture: {config.get('architecture', 'Unknown')}")
        print(f"   🎯 Type: {config.get('model_type', 'classification')}")
        
        try:
            model_path = config['model_path']
            if os.path.exists(model_path):
                # Load model with custom objects if it's a segmentation model
                if config.get('model_type') == 'segmentation':
                    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
                else:
                    model = load_model(model_path, compile=False)
                
                models[disease_key] = {
                    'model': model,
                    'config': config
                }
                
                # Load labels from file if specified
                if 'labels_path' in config and os.path.exists(config['labels_path']):
                    with open(config['labels_path'], 'r') as f:
                        models[disease_key]['labels'] = [line.strip() for line in f.readlines()]
                else:
                    models[disease_key]['labels'] = config.get('labels', ['Unknown'])
                
                print(f"   ✅ Loaded successfully!")
                print(f"   📋 Output: {models[disease_key]['labels']}")
                
                # Print performance metrics if available
                if 'accuracy' in config:
                    print(f"   📊 Accuracy: {config['accuracy']*100:.1f}%")
                if 'dice_coefficient' in config:
                    print(f"   📊 Dice Coefficient: {config['dice_coefficient']:.3f}")
            else:
                print(f"   ⚠️  Model file not found: {model_path}")
                models[disease_key] = None
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            models[disease_key] = None
    
    # Print summary
    loaded_count = sum(1 for m in models.values() if m is not None)
    total_count = len(MODELS_CONFIG)
    
    print("\n" + "="*60)
    print(f" 🚀 Server Ready! ({loaded_count}/{total_count} models loaded)")
    print("="*60 + "\n")

def preprocess_image_classification(image_path, img_size=224):
    """Preprocesses the uploaded image for classification models."""
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_segmentation(image_path, img_size=256):
    """Preprocesses the uploaded image for segmentation models."""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add channel and batch dimensions
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)    # Add batch dimension
    
    return img

def create_segmentation_overlay(original_image_path, mask, output_path, img_size=256):
    """Creates a visualization overlay of the segmentation mask on the original image."""
    # Read original image
    original = cv2.imread(original_image_path)
    original = cv2.resize(original, (img_size, img_size))
    
    # Convert mask to 0-255 range
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Create colored overlay (green for detected region)
    overlay = original.copy()
    overlay[:, :, 1] = np.where(mask_uint8 > 127, 255, overlay[:, :, 1])  # Green channel
    
    # Blend original and overlay
    result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    
    # Draw contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, result)
    
    # Also save just the mask
    mask_output_path = output_path.replace('.png', '_mask.png')
    cv2.imwrite(mask_output_path, mask_uint8)
    
    return output_path, mask_output_path

def make_prediction_classification(disease_type, image_path, model_data):
    """Makes a classification prediction."""
    model = model_data['model']
    labels = model_data['labels']
    img_size = model_data['config']['img_size']
    
    input_data = preprocess_image_classification(image_path, img_size)
    predictions = model.predict(input_data, verbose=0)
    
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index] * 100)
    label = labels[predicted_index]
    
    # Get all class probabilities
    all_predictions = [
        {'label': labels[i], 'confidence': float(predictions[0][i] * 100)}
        for i in range(len(labels))
    ]
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Determine if critical based on disease type
    if disease_type == 'brain_tumor':
        # For brain tumor: index 2 is 'No Tumor (Healthy)'
        is_critical = predicted_index != 2
    else:
        # For other diseases: index 0 is typically normal
        is_critical = predicted_index != 0
    
    return {
        'label': label,
        'confidence': confidence,
        'all_predictions': all_predictions,
        'is_critical': is_critical,
        'model_type': 'classification'
    }

def make_prediction_segmentation(disease_type, image_path, model_data):
    """Makes a segmentation prediction."""
    model = model_data['model']
    img_size = model_data['config']['img_size']
    
    # Preprocess image
    input_data = preprocess_image_segmentation(image_path, img_size)
    
    # Predict mask
    predicted_mask = model.predict(input_data, verbose=0)[0]
    
    # Threshold mask
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)
    
    # Calculate metrics
    positive_pixels = np.sum(predicted_mask_binary)
    total_pixels = predicted_mask_binary.size
    coverage_percent = (positive_pixels / total_pixels) * 100
    
    # Create visualization overlay
    filename = os.path.basename(image_path)
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    overlay_path, mask_path = create_segmentation_overlay(
        image_path, 
        predicted_mask_binary[:, :, 0], 
        result_path,
        img_size
    )
    
    # Determine if fetal head is detected
    is_detected = coverage_percent > 0.5  # At least 0.5% of image should be fetal head
    
    return {
        'label': 'Fetal Head Detected' if is_detected else 'No Clear Fetal Head',
        'confidence': float(coverage_percent * 10),  # Scale coverage to confidence-like score
        'coverage_percent': float(coverage_percent),
        'is_critical': not is_detected,
        'segmentation_overlay': os.path.basename(overlay_path),
        'segmentation_mask': os.path.basename(mask_path),
        'model_type': 'segmentation',
        'metrics': {
            'positive_pixels': int(positive_pixels),
            'total_pixels': int(total_pixels),
            'mean_confidence': float(np.mean(predicted_mask))
        }
    }

def make_prediction(disease_type, image_path):
    """Makes a prediction using the appropriate model."""
    if disease_type not in models or models[disease_type] is None:
        return None, "Model not available"
    
    model_data = models[disease_type]
    model_type = model_data['config'].get('model_type', 'classification')
    
    try:
        if model_type == 'segmentation':
            result = make_prediction_segmentation(disease_type, image_path, model_data)
        else:
            result = make_prediction_classification(disease_type, image_path, model_data)
        
        return result, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Landing page with all diagnostic options."""
    return render_template('index.html')

@app.route('/diagnose/<disease_type>', methods=['GET', 'POST'])
def diagnose(disease_type):
    """Generic diagnosis route for all disease types."""
    if disease_type not in MODELS_CONFIG:
        return render_template('error.html', message="Invalid diagnosis type"), 404
    
    config = MODELS_CONFIG[disease_type]
    result = None
    image_filename = None
    error = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file uploaded'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No file selected'
            elif not allowed_file(file.filename):
                error = 'Invalid file type. Please upload JPG, PNG, or GIF images.'
            else:
                filename = secure_filename(file.filename)
                # Add disease type prefix to avoid conflicts
                filename = f"{disease_type}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_filename = filename
                
                result, error = make_prediction(disease_type, file_path)
    
    return render_template(
        'diagnose.html',
        disease_type=disease_type,
        disease_name=disease_type.replace('_', ' ').title(),
        config=config,
        result=result,
        image_filename=image_filename,
        error=error
    )

@app.route('/about')
def about():
    """About page with information about the system."""
    return render_template('about.html')

@app.route('/api/predict/<disease_type>', methods=['POST'])
def api_predict(disease_type):
    """REST API endpoint for predictions."""
    if disease_type not in MODELS_CONFIG:
        return jsonify({'error': 'Invalid disease type'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filename = f"api_{disease_type}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    result, error = make_prediction(disease_type, file_path)
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'success': True,
        'disease_type': disease_type,
        'prediction': result
    })

# Run the app
if __name__ == '__main__':
    load_all_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
