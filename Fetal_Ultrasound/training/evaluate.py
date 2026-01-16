import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow.keras.backend as K
from tqdm import tqdm
import json

# --- Metric Functions ---
def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient for segmentation accuracy."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """Dice loss function."""
    return 1 - dice_coef(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1):
    """Intersection over Union (IoU) score."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(y_true, y_pred):
    """Pixel-wise accuracy."""
    return K.mean(K.equal(y_true, K.round(y_pred)))

def sensitivity(y_true, y_pred, smooth=1):
    """Sensitivity (Recall) - True Positive Rate."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.round(y_pred))
    true_positives = K.sum(y_true_f * y_pred_f)
    possible_positives = K.sum(y_true_f)
    return (true_positives + smooth) / (possible_positives + smooth)

def specificity(y_true, y_pred, smooth=1):
    """Specificity - True Negative Rate."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.round(y_pred))
    true_negatives = K.sum((1 - y_true_f) * (1 - y_pred_f))
    possible_negatives = K.sum(1 - y_true_f)
    return (true_negatives + smooth) / (possible_negatives + smooth)

# --- Data Loading ---
def load_test_data(test_img_dir, test_mask_dir, img_size=(256, 256)):
    """Load test images and masks."""
    images = []
    masks = []
    filenames = []
    
    img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    
    print(f"Loading {len(img_files)} test images...")
    
    for img_name in tqdm(img_files):
        # Load image
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask
        mask_name = img_name.replace('_HC.png', '_HC_Annotation.png')
        mask_path = os.path.join(test_mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        # Normalize
        img = img / 255.0
        mask = mask / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        
        images.append(img)
        masks.append(mask)
        filenames.append(img_name)
    
    return np.array(images), np.array(masks), filenames

# --- Evaluation ---
def evaluate_model(model_path, test_img_dir, test_mask_dir, output_dir='evaluation_results'):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model_path: Path to saved model (.h5 file)
        test_img_dir: Directory containing test images
        test_mask_dir: Directory containing test masks
        output_dir: Directory to save evaluation results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("FETAL ULTRASOUND SEGMENTATION MODEL EVALUATION")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading model...")
    try:
        model = load_model(
            model_path,
            custom_objects={
                'dice_coef': dice_coef,
                'dice_coef_loss': dice_coef_loss,
                'iou_score': iou_score,
                'pixel_accuracy': pixel_accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
        )
        print(f"✓ Model loaded from: {model_path}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load test data
    X_test, y_test, filenames = load_test_data(test_img_dir, test_mask_dir)
    print(f"✓ Loaded {len(X_test)} test samples\n")
    
    # Make predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test, batch_size=8, verbose=1)
    y_pred_binary = (y_pred > 0.5).astype(np.float32)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = {}
    
    # Using Keras backend for metrics
    dice_scores = []
    iou_scores = []
    pixel_accuracies = []
    sensitivities = []
    specificities = []
    
    for i in range(len(y_test)):
        y_true_tensor = tf.constant(y_test[i:i+1])
        y_pred_tensor = tf.constant(y_pred[i:i+1])
        
        dice_scores.append(dice_coef(y_true_tensor, y_pred_tensor).numpy())
        iou_scores.append(iou_score(y_true_tensor, y_pred_tensor).numpy())
        pixel_accuracies.append(pixel_accuracy(y_true_tensor, y_pred_tensor).numpy())
        sensitivities.append(sensitivity(y_true_tensor, y_pred_tensor).numpy())
        specificities.append(specificity(y_true_tensor, y_pred_tensor).numpy())
    
    metrics['dice_coefficient'] = {
        'mean': float(np.mean(dice_scores)),
        'std': float(np.std(dice_scores)),
        'min': float(np.min(dice_scores)),
        'max': float(np.max(dice_scores))
    }
    
    metrics['iou_score'] = {
        'mean': float(np.mean(iou_scores)),
        'std': float(np.std(iou_scores)),
        'min': float(np.min(iou_scores)),
        'max': float(np.max(iou_scores))
    }
    
    metrics['pixel_accuracy'] = {
        'mean': float(np.mean(pixel_accuracies)),
        'std': float(np.std(pixel_accuracies)),
        'min': float(np.min(pixel_accuracies)),
        'max': float(np.max(pixel_accuracies))
    }
    
    metrics['sensitivity'] = {
        'mean': float(np.mean(sensitivities)),
        'std': float(np.std(sensitivities))
    }
    
    metrics['specificity'] = {
        'mean': float(np.mean(specificities)),
        'std': float(np.std(specificities))
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 Test Set Size: {len(X_test)} images\n")
    print(f"🎯 Dice Coefficient:  {metrics['dice_coefficient']['mean']:.4f} ± {metrics['dice_coefficient']['std']:.4f}")
    print(f"   Range: [{metrics['dice_coefficient']['min']:.4f}, {metrics['dice_coefficient']['max']:.4f}]")
    print()
    print(f"📐 IoU Score:         {metrics['iou_score']['mean']:.4f} ± {metrics['iou_score']['std']:.4f}")
    print(f"   Range: [{metrics['iou_score']['min']:.4f}, {metrics['iou_score']['max']:.4f}]")
    print()
    print(f"✅ Pixel Accuracy:    {metrics['pixel_accuracy']['mean']:.4f} ± {metrics['pixel_accuracy']['std']:.4f}")
    print(f"   Range: [{metrics['pixel_accuracy']['min']:.4f}, {metrics['pixel_accuracy']['max']:.4f}]")
    print()
    print(f"🔍 Sensitivity:       {metrics['sensitivity']['mean']:.4f} ± {metrics['sensitivity']['std']:.4f}")
    print(f"🎪 Specificity:       {metrics['specificity']['mean']:.4f} ± {metrics['specificity']['std']:.4f}")
    print("=" * 70)
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    visualize_predictions(X_test, y_test, y_pred_binary, filenames, output_dir, num_samples=10)
    
    # Plot metric distributions
    plot_metric_distributions(dice_scores, iou_scores, pixel_accuracies, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

def visualize_predictions(X_test, y_test, y_pred, filenames, output_dir, num_samples=10):
    """Visualize sample predictions."""
    num_samples = min(num_samples, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, idx in enumerate(indices):
        # Original image
        axes[i, 0].imshow(X_test[idx].squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Image: {filenames[idx]}')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(y_test[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(y_pred[idx].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = X_test[idx].squeeze().copy()
        overlay = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay[:, :, 1] = np.where(y_pred[idx].squeeze() > 0.5, 255, overlay[:, :, 1])
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'predictions_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualizations saved to: {viz_path}")

def plot_metric_distributions(dice_scores, iou_scores, pixel_accuracies, output_dir):
    """Plot distributions of evaluation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Dice coefficient distribution
    axes[0].hist(dice_scores, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(dice_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dice_scores):.4f}')
    axes[0].set_xlabel('Dice Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Dice Coefficient Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU score distribution
    axes[1].hist(iou_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(iou_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(iou_scores):.4f}')
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('IoU Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Pixel accuracy distribution
    axes[2].hist(pixel_accuracies, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(pixel_accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pixel_accuracies):.4f}')
    axes[2].set_xlabel('Pixel Accuracy')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Pixel Accuracy Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'metric_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metric distributions saved to: {dist_path}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "fetal_ultrasound_best.h5"
    TEST_IMG_DIR = "../dataset/test/images"
    TEST_MASK_DIR = "../dataset/test/masks"
    OUTPUT_DIR = "evaluation_results"
    
    # Run evaluation
    evaluate_model(MODEL_PATH, TEST_IMG_DIR, TEST_MASK_DIR, OUTPUT_DIR)
