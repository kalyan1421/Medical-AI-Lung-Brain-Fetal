import os
import numpy as np
import tensorflow as tf
from model import AttentionUNet, UNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, 
                                        EarlyStopping, TensorBoard, CSVLogger)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Define Advanced Metrics & Loss Functions ---
def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient - primary metric for segmentation."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """Dice loss function."""
    return 1 - dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss to handle class imbalance."""
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    return K.mean(K.sum(loss, axis=-1))

def combined_loss(y_true, y_pred):
    """Combined loss: Dice + Focal Loss for severe class imbalance."""
    dice_loss = dice_coef_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=0.75, gamma=2.0)
    # Weight dice loss more heavily
    return 0.7 * dice_loss + 0.3 * focal

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

# --- 2. Configuration ---
class Config:
    # Model
    MODEL_TYPE = "unet"  # "attention_unet" or "unet" - using simpler UNet first
    IMG_SIZE = (256, 256)
    INPUT_CHANNELS = 1
    FILTERS_BASE = 32
    
    # Training
    BATCH_SIZE = 8  # Reduced for stability
    EPOCHS = 100
    INITIAL_LEARNING_RATE = 5e-4  # Increased for faster learning
    
    # Paths
    TRAIN_PATH = "../dataset/train"
    VAL_PATH = "../dataset/val"
    
    # Augmentation
    AUGMENTATION = True
    ROTATION_RANGE = 15
    WIDTH_SHIFT = 0.1
    HEIGHT_SHIFT = 0.1
    ZOOM_RANGE = 0.15
    HORIZONTAL_FLIP = True
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    REDUCE_LR_FACTOR = 0.5
    
    # Output
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_NAME = f"fetal_ultrasound_{MODEL_TYPE}_{TIMESTAMP}"
    BEST_MODEL_PATH = f"{MODEL_NAME}_best.h5"
    FINAL_MODEL_PATH = f"{MODEL_NAME}_final.h5"
    LOGS_DIR = f"logs/{MODEL_NAME}"

# --- 3. Data Generators with Strong Augmentation ---
def create_generators(config):
    """Create training and validation data generators."""
    
    if config.AUGMENTATION:
        data_gen_args = dict(
            rescale=1./255,
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT,
            height_shift_range=config.HEIGHT_SHIFT,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            fill_mode='nearest'
        )
    else:
        data_gen_args = dict(rescale=1./255)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # For validation (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    seed = 42
    
    def train_generator(batch_size, train_path, image_folder, mask_folder, datagen_img, datagen_mask):
        """Generator for training/validation data."""
        image_generator = datagen_img.flow_from_directory(
            train_path,
            classes=[image_folder],
            class_mode=None,
            color_mode='grayscale',
            target_size=config.IMG_SIZE,
            batch_size=batch_size,
            seed=seed
        )
        mask_generator = datagen_mask.flow_from_directory(
            train_path,
            classes=[mask_folder],
            class_mode=None,
            color_mode='grayscale',
            target_size=config.IMG_SIZE,
            batch_size=batch_size,
            seed=seed
        )
        
        # Combine generators
        train_gen = zip(image_generator, mask_generator)
        for (img, mask) in train_gen:
            # Ensure mask is binary (0 or 1)
            mask = (mask > 0.5).astype(np.float32)
            
            # Verify masks are not empty (debugging)
            mask_sum = np.sum(mask)
            if mask_sum < 10:  # Too few positive pixels
                print(f"Warning: Batch has very few positive pixels: {mask_sum}")
            
            yield (img, mask)
    
    # Create generators
    train_gen = train_generator(
        config.BATCH_SIZE, 
        config.TRAIN_PATH, 
        "images", 
        "masks",
        image_datagen,
        mask_datagen
    )
    
    val_gen = train_generator(
        config.BATCH_SIZE, 
        config.VAL_PATH, 
        "images", 
        "masks",
        val_datagen,
        val_datagen
    )
    
    return train_gen, val_gen

# --- 4. Build and Compile Model ---
def build_model(config):
    """Build and compile the segmentation model."""
    
    print(f"\nBuilding {config.MODEL_TYPE.upper()} model...")
    
    if config.MODEL_TYPE == "attention_unet":
        model = AttentionUNet(
            input_size=(*config.IMG_SIZE, config.INPUT_CHANNELS),
            filters_base=config.FILTERS_BASE
        )
    else:
        model = UNet(
            input_size=(*config.IMG_SIZE, config.INPUT_CHANNELS)
        )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.INITIAL_LEARNING_RATE),
        loss=combined_loss,
        metrics=[
            dice_coef,
            iou_score,
            pixel_accuracy,
            sensitivity,
            specificity,
            'binary_crossentropy'
        ]
    )
    
    print(f"✓ Model compiled successfully")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    return model

# --- 5. Setup Callbacks ---
def setup_callbacks(config):
    """Setup training callbacks."""
    
    # Create logs directory
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    callbacks = [
        # Save best model based on validation Dice coefficient
        ModelCheckpoint(
            config.BEST_MODEL_PATH,
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=1,
            restore_best_weights=True
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=config.LOGS_DIR,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV Logger
        CSVLogger(
            f"{config.MODEL_NAME}_training_log.csv",
            separator=',',
            append=False
        )
    ]
    
    return callbacks

# --- 6. Plot Training History ---
def plot_training_history(history, config):
    """Plot and save training history."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Dice Coefficient
    axes[0, 0].plot(history.history['dice_coef'], label='Train')
    axes[0, 0].plot(history.history['val_dice_coef'], label='Validation')
    axes[0, 0].set_title('Dice Coefficient')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU Score
    axes[0, 2].plot(history.history['iou_score'], label='Train')
    axes[0, 2].plot(history.history['val_iou_score'], label='Validation')
    axes[0, 2].set_title('IoU Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Pixel Accuracy
    axes[1, 0].plot(history.history['pixel_accuracy'], label='Train')
    axes[1, 0].plot(history.history['val_pixel_accuracy'], label='Validation')
    axes[1, 0].set_title('Pixel Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensitivity
    axes[1, 1].plot(history.history['sensitivity'], label='Train')
    axes[1, 1].plot(history.history['val_sensitivity'], label='Validation')
    axes[1, 1].set_title('Sensitivity (Recall)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Sensitivity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Specificity
    axes[1, 2].plot(history.history['specificity'], label='Train')
    axes[1, 2].plot(history.history['val_specificity'], label='Validation')
    axes[1, 2].set_title('Specificity')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Specificity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.MODEL_NAME}_training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Training history plot saved: {config.MODEL_NAME}_training_history.png")

# --- 7. Main Training Function ---
def train():
    """Main training function."""
    
    # Initialize configuration
    config = Config()
    
    print("=" * 80)
    print("FETAL ULTRASOUND SEGMENTATION - MODEL TRAINING")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model Type: {config.MODEL_TYPE}")
    print(f"   Image Size: {config.IMG_SIZE}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Learning Rate: {config.INITIAL_LEARNING_RATE}")
    print(f"   Augmentation: {config.AUGMENTATION}")
    print()
    
    # Calculate steps per epoch
    try:
        num_train_imgs = len([f for f in os.listdir(os.path.join(config.TRAIN_PATH, "images")) if f.endswith('.png')])
        num_val_imgs = len([f for f in os.listdir(os.path.join(config.VAL_PATH, "images")) if f.endswith('.png')])
        
        steps_per_epoch = num_train_imgs // config.BATCH_SIZE
        validation_steps = num_val_imgs // config.BATCH_SIZE
        
        print(f"📊 Dataset Information:")
        print(f"   Training images: {num_train_imgs}")
        print(f"   Validation images: {num_val_imgs}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Validation steps: {validation_steps}")
        print()
        
    except Exception as e:
        print(f"⚠ Error: Could not find dataset. Please run clean_dataset.py first!")
        print(f"   Error message: {e}")
        return
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen = create_generators(config)
    print("✓ Data generators created")
    
    # Build model
    model = build_model(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    print(f"✓ Callbacks configured")
    print()
    
    # Save configuration
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    # Convert non-serializable types
    config_dict['IMG_SIZE'] = list(config.IMG_SIZE)
    config_dict['TIMESTAMP'] = str(config.TIMESTAMP)
    
    with open(f"{config.MODEL_NAME}_config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Train model
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print(f"\n\nSaving final model...")
    model.save(config.FINAL_MODEL_PATH)
    print(f"✓ Final model saved: {config.FINAL_MODEL_PATH}")
    
    # Plot training history
    print(f"\nGenerating training plots...")
    plot_training_history(history, config)
    
    # Print final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    # Get best epoch results
    best_epoch = np.argmax(history.history['val_dice_coef'])
    
    print(f"\n🏆 Best Results (Epoch {best_epoch + 1}):")
    print(f"   Validation Dice:       {history.history['val_dice_coef'][best_epoch]:.4f}")
    print(f"   Validation IoU:        {history.history['val_iou_score'][best_epoch]:.4f}")
    print(f"   Validation Accuracy:   {history.history['val_pixel_accuracy'][best_epoch]:.4f}")
    print(f"   Validation Sensitivity: {history.history['val_sensitivity'][best_epoch]:.4f}")
    print(f"   Validation Specificity: {history.history['val_specificity'][best_epoch]:.4f}")
    print()
    print(f"📁 Saved Files:")
    print(f"   Best model: {config.BEST_MODEL_PATH}")
    print(f"   Final model: {config.FINAL_MODEL_PATH}")
    print(f"   Training log: {config.MODEL_NAME}_training_log.csv")
    print(f"   Training plot: {config.MODEL_NAME}_training_history.png")
    print(f"   TensorBoard logs: {config.LOGS_DIR}")
    print()
    print("=" * 80)
    print("\n✓ Next step: Run evaluate.py to test the model on test set")
    print(f"✓ View training logs: tensorboard --logdir={config.LOGS_DIR}")

if __name__ == "__main__":
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    
    # Run training
    train()
