"""
Comprehensive Training Visualization Script for Fetal Ultrasound Segmentation
Generates publication-quality graphs for training metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import os

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def find_latest_training_log(training_dir='training'):
    """Find the most recent training log file"""
    log_files = glob.glob(os.path.join(training_dir, '*_training_log.csv'))
    if not log_files:
        raise FileNotFoundError("No training log files found!")
    
    # Sort by modification time
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"📊 Using training log: {os.path.basename(latest_log)}")
    return latest_log


def load_training_data(log_file):
    """Load training data from CSV"""
    df = pd.read_csv(log_file)
    df['epoch'] = df['epoch'] + 1  # Convert to 1-indexed for display
    print(f"✅ Loaded {len(df)} epochs of training data")
    return df


def plot_loss_curves(df, save_dir):
    """Plot training and validation loss"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training and Validation Loss Over Time', fontweight='bold', fontsize=16)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for best validation loss
    best_epoch = df['val_loss'].idxmin() + 1
    best_loss = df['val_loss'].min()
    ax.annotate(f'Best: {best_loss:.4f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_loss),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '1_loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_dice_coefficient(df, save_dir):
    """Plot Dice coefficient (most important metric for segmentation)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['dice_coef'], 'g-', linewidth=2.5, label='Training Dice', marker='o', markersize=5)
    ax.plot(df['epoch'], df['val_dice_coef'], 'purple', linewidth=2.5, label='Validation Dice', marker='s', markersize=5)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Dice Coefficient', fontweight='bold')
    ax.set_title('Dice Coefficient Progress (Higher is Better)', fontweight='bold', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add target line
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (0.75)')
    
    # Add annotation for best validation dice
    best_epoch = df['val_dice_coef'].idxmax() + 1
    best_dice = df['val_dice_coef'].max()
    ax.annotate(f'Best: {best_dice:.4f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_dice),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add current performance text
    current_dice = df['val_dice_coef'].iloc[-1]
    progress = (current_dice / 0.75) * 100
    ax.text(0.02, 0.98, f'Current Validation Dice: {current_dice:.4f}\nProgress to Target: {progress:.1f}%',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '2_dice_coefficient.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_iou_score(df, save_dir):
    """Plot IoU (Intersection over Union) score"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['iou_score'], 'orange', linewidth=2, label='Training IoU', marker='o', markersize=4)
    ax.plot(df['epoch'], df['val_iou_score'], 'brown', linewidth=2, label='Validation IoU', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('IoU Score', fontweight='bold')
    ax.set_title('IoU (Intersection over Union) Score Progress', fontweight='bold', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add target line
    ax.axhline(y=0.65, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (0.65)')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '3_iou_score.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_pixel_accuracy(df, save_dir):
    """Plot pixel accuracy"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['pixel_accuracy'], 'cyan', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax.plot(df['epoch'], df['val_pixel_accuracy'], 'blue', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Pixel Accuracy', fontweight='bold')
    ax.set_title('Pixel-wise Accuracy Progress', fontweight='bold', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.8, 1.0])  # Zoom in on relevant range
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '4_pixel_accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_sensitivity_specificity(df, save_dir):
    """Plot sensitivity and specificity"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sensitivity plot
    ax1.plot(df['epoch'], df['sensitivity'], 'green', linewidth=2, label='Training Sensitivity', marker='o', markersize=4)
    ax1.plot(df['epoch'], df['val_sensitivity'], 'darkgreen', linewidth=2, label='Validation Sensitivity', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Sensitivity (Recall)', fontweight='bold')
    ax1.set_title('Sensitivity: How well model finds fetal heads', fontweight='bold', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.85, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Specificity plot
    ax2.plot(df['epoch'], df['specificity'], 'purple', linewidth=2, label='Training Specificity', marker='o', markersize=4)
    ax2.plot(df['epoch'], df['val_specificity'], 'darkviolet', linewidth=2, label='Validation Specificity', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Specificity', fontweight='bold')
    ax2.set_title('Specificity: How well model avoids false positives', fontweight='bold', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.95, 1.0])  # Zoom in on relevant range
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '5_sensitivity_specificity.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_learning_rate(df, save_dir):
    """Plot learning rate schedule"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['learning_rate'], 'red', linewidth=2, marker='o', markersize=5)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)', fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    # Annotate LR reductions
    lr_changes = df[df['learning_rate'] != df['learning_rate'].shift()]['epoch']
    for epoch in lr_changes:
        if epoch != df['epoch'].iloc[0]:  # Skip first epoch
            lr_val = df[df['epoch'] == epoch]['learning_rate'].values[0]
            ax.axvline(x=epoch, color='orange', linestyle='--', alpha=0.5)
            ax.text(epoch, lr_val, f'  LR reduced', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '6_learning_rate.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_comprehensive_dashboard(df, save_dir):
    """Create a comprehensive dashboard with all key metrics"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.7)
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Current metrics summary
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    current = df.iloc[-1]
    best_dice = df['val_dice_coef'].max()
    best_dice_epoch = df['val_dice_coef'].idxmax() + 1
    
    summary_text = f"""
    📊 CURRENT METRICS (Epoch {int(current['epoch'])})
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    Validation Dice: {current['val_dice_coef']:.4f}
    Validation IoU: {current['val_iou_score']:.4f}
    Validation Loss: {current['val_loss']:.4f}
    
    🏆 BEST PERFORMANCE
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    Best Val Dice: {best_dice:.4f}
    At Epoch: {best_dice_epoch}
    
    🎯 TARGET METRICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    Target Dice: 0.7500
    Progress: {(best_dice/0.75)*100:.1f}%
    """
    ax2.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Dice Coefficient
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['epoch'], df['dice_coef'], 'g-', linewidth=2, label='Train', alpha=0.7)
    ax3.plot(df['epoch'], df['val_dice_coef'], color='purple', linewidth=2, label='Val', alpha=0.7)
    ax3.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Target')
    ax3.set_title('Dice Coefficient', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Dice Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. IoU Score
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['epoch'], df['iou_score'], 'orange', linewidth=2, label='Train', alpha=0.7)
    ax4.plot(df['epoch'], df['val_iou_score'], 'brown', linewidth=2, label='Val', alpha=0.7)
    ax4.set_title('IoU Score', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IoU')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Pixel Accuracy
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df['epoch'], df['pixel_accuracy'], 'cyan', linewidth=2, label='Train', alpha=0.7)
    ax5.plot(df['epoch'], df['val_pixel_accuracy'], 'blue', linewidth=2, label='Val', alpha=0.7)
    ax5.set_title('Pixel Accuracy', fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy')
    ax5.set_ylim([0.8, 1.0])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Sensitivity
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(df['epoch'], df['sensitivity'], 'green', linewidth=2, label='Train', alpha=0.7)
    ax6.plot(df['epoch'], df['val_sensitivity'], 'darkgreen', linewidth=2, label='Val', alpha=0.7)
    ax6.set_title('Sensitivity (Recall)', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Sensitivity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Specificity
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(df['epoch'], df['specificity'], 'purple', linewidth=2, label='Train', alpha=0.7)
    ax7.plot(df['epoch'], df['val_specificity'], 'darkviolet', linewidth=2, label='Val', alpha=0.7)
    ax7.set_title('Specificity', fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Specificity')
    ax7.set_ylim([0.95, 1.0])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Learning Rate
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(df['epoch'], df['learning_rate'], 'red', linewidth=2, marker='o', markersize=3)
    ax8.set_title('Learning Rate', fontweight='bold')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Learning Rate')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Fetal Ultrasound Segmentation - Training Dashboard', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_dir, '0_comprehensive_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_comparison_all_runs(training_dir='training', save_dir='graphs'):
    """Compare all training runs"""
    log_files = glob.glob(os.path.join(training_dir, '*_training_log.csv'))
    
    if len(log_files) <= 1:
        print("⚠️  Only one training run found, skipping comparison plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for idx, log_file in enumerate(sorted(log_files)):
        df = pd.read_csv(log_file)
        df['epoch'] = df['epoch'] + 1
        run_name = os.path.basename(log_file).replace('_training_log.csv', '')
        
        # Plot validation dice
        ax1.plot(df['epoch'], df['val_dice_coef'], linewidth=2, 
                label=run_name[-19:], color=colors[idx], alpha=0.7)  # Last 19 chars (timestamp)
        
        # Plot validation loss
        ax2.plot(df['epoch'], df['val_loss'], linewidth=2,
                label=run_name[-19:], color=colors[idx], alpha=0.7)
    
    ax1.set_title('Validation Dice Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Dice')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss Comparison', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of All Training Runs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '7_all_runs_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def generate_training_report(df, save_dir):
    """Generate a text report of training statistics"""
    report = []
    report.append("=" * 60)
    report.append("FETAL ULTRASOUND SEGMENTATION - TRAINING REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Training summary
    report.append("📊 TRAINING SUMMARY")
    report.append("-" * 60)
    report.append(f"Total Epochs Completed: {len(df)}")
    report.append(f"Training Status: {'Ongoing' if len(df) < 100 else 'Complete'}")
    report.append("")
    
    # Best metrics
    best_dice_idx = df['val_dice_coef'].idxmax()
    best_dice_epoch = best_dice_idx + 1
    best_dice = df.loc[best_dice_idx]
    
    report.append("🏆 BEST VALIDATION METRICS")
    report.append("-" * 60)
    report.append(f"Best Epoch: {best_dice_epoch}")
    report.append(f"Dice Coefficient: {best_dice['val_dice_coef']:.4f}")
    report.append(f"IoU Score: {best_dice['val_iou_score']:.4f}")
    report.append(f"Loss: {best_dice['val_loss']:.4f}")
    report.append(f"Pixel Accuracy: {best_dice['val_pixel_accuracy']:.4f}")
    report.append(f"Sensitivity: {best_dice['val_sensitivity']:.4f}")
    report.append(f"Specificity: {best_dice['val_specificity']:.4f}")
    report.append("")
    
    # Current metrics
    current = df.iloc[-1]
    report.append(f"📈 CURRENT METRICS (Epoch {int(current['epoch'])})")
    report.append("-" * 60)
    report.append(f"Dice Coefficient: {current['val_dice_coef']:.4f}")
    report.append(f"IoU Score: {current['val_iou_score']:.4f}")
    report.append(f"Loss: {current['val_loss']:.4f}")
    report.append(f"Pixel Accuracy: {current['val_pixel_accuracy']:.4f}")
    report.append(f"Sensitivity: {current['val_sensitivity']:.4f}")
    report.append(f"Specificity: {current['val_specificity']:.4f}")
    report.append(f"Learning Rate: {current['learning_rate']:.6f}")
    report.append("")
    
    # Progress towards target
    target_dice = 0.75
    progress = (best_dice['val_dice_coef'] / target_dice) * 100
    report.append("🎯 PROGRESS TOWARDS TARGET")
    report.append("-" * 60)
    report.append(f"Target Dice: {target_dice:.4f}")
    report.append(f"Best Dice: {best_dice['val_dice_coef']:.4f}")
    report.append(f"Progress: {progress:.1f}%")
    report.append(f"Remaining: {target_dice - best_dice['val_dice_coef']:.4f}")
    report.append("")
    
    # Improvement over time
    if len(df) >= 10:
        first_10_dice = df['val_dice_coef'].iloc[:10].mean()
        last_10_dice = df['val_dice_coef'].iloc[-10:].mean()
        improvement = ((last_10_dice - first_10_dice) / first_10_dice) * 100
        
        report.append("📊 IMPROVEMENT ANALYSIS")
        report.append("-" * 60)
        report.append(f"First 10 Epochs Avg Dice: {first_10_dice:.4f}")
        report.append(f"Last 10 Epochs Avg Dice: {last_10_dice:.4f}")
        report.append(f"Improvement: {improvement:+.1f}%")
        report.append("")
    
    # Learning rate reductions
    lr_changes = df[df['learning_rate'] != df['learning_rate'].shift()]
    if len(lr_changes) > 1:
        report.append("📉 LEARNING RATE SCHEDULE")
        report.append("-" * 60)
        for idx, row in lr_changes.iterrows():
            report.append(f"Epoch {int(row['epoch'])}: LR = {row['learning_rate']:.6f}")
        report.append("")
    
    report.append("=" * 60)
    
    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(save_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Saved: {report_path}")
    print("\n" + report_text)


def main():
    """Main function to generate all visualizations"""
    print("\n" + "="*60)
    print("🎨 FETAL ULTRASOUND TRAINING VISUALIZATION")
    print("="*60 + "\n")
    
    # Setup directories
    training_dir = 'training'
    save_dir = 'graphs'
    os.makedirs(save_dir, exist_ok=True)
    
    # Find and load latest training log
    log_file = find_latest_training_log(training_dir)
    df = load_training_data(log_file)
    
    print("\n📈 Generating graphs...")
    print("-" * 60)
    
    # Generate all plots
    plot_comprehensive_dashboard(df, save_dir)  # Generate dashboard first
    plot_loss_curves(df, save_dir)
    plot_dice_coefficient(df, save_dir)
    plot_iou_score(df, save_dir)
    plot_pixel_accuracy(df, save_dir)
    plot_sensitivity_specificity(df, save_dir)
    plot_learning_rate(df, save_dir)
    plot_comparison_all_runs(training_dir, save_dir)
    
    # Generate text report
    print("\n📝 Generating training report...")
    print("-" * 60)
    generate_training_report(df, save_dir)
    
    print("\n" + "="*60)
    print(f"✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"📁 Graphs saved to: {os.path.abspath(save_dir)}/")
    print("="*60 + "\n")
    
    print("📊 Generated files:")
    for file in sorted(os.listdir(save_dir)):
        if file.endswith(('.png', '.txt')):
            print(f"   - {file}")


if __name__ == "__main__":
    main()
