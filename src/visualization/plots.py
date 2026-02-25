"""
Visualization and Plotting Module
Creates plots for training history, metrics, and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class Visualizer:
    """Create visualizations for model training and evaluation"""
    
    def __init__(self, save_dir=None):
        """
        Initialize visualizer
        
        Args:
            save_dir (str): Directory to save plots
        """
        self.save_dir = save_dir or config.get('paths', 'plots', default='results/plots')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training and validation metrics
        
        Args:
            history: Keras training history
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
            axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, images, true_labels, pred_labels, pred_probs,
                               num_samples=16, save_name='sample_predictions.png'):
        """
        Plot sample predictions
        
        Args:
            images: Sample images
            true_labels: True labels
            pred_labels: Predicted labels
            pred_probs: Prediction probabilities
            num_samples: Number of samples to plot
            save_name: Filename to save plot
        """
        num_samples = min(num_samples, len(images))
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            # Get image
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Get labels
            true_label = "Tumor" if true_labels[i] == 1 else "No Tumor"
            pred_label = "Tumor" if pred_labels[i] == 1 else "No Tumor"
            confidence = pred_probs[i] if pred_labels[i] == 1 else (1 - pred_probs[i])
            
            # Determine color (green for correct, red for incorrect)
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            
            # Plot
            axes[i].imshow(img)
            axes[i].axis('off')
            
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"
            axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Hide extra subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix_detailed(self, y_true, y_pred, save_name='confusion_matrix_detailed.png'):
        """
        Plot detailed confusion matrix with percentages
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_name: Filename to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'],
                   cbar_kws={'label': 'Count'},
                   ax=ax)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=12, color='gray')
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add clinical interpretation
        tn, fp, fn, tp = cm.ravel()
        interpretation = (
            f"\nTrue Negatives (TN): {tn} - Correctly identified healthy patients\n"
            f"False Positives (FP): {fp} - Healthy patients incorrectly flagged (false alarm)\n"
            f"False Negatives (FN): {fn} - Missed tumors (MOST DANGEROUS)\n"
            f"True Positives (TP): {tp} - Correctly identified tumors"
        )
        
        plt.figtext(0.5, -0.05, interpretation, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve_detailed(self, y_true, y_pred_proba, save_name='roc_curve_detailed.png'):
        """
        Plot detailed ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_name: Filename to save plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5)')
        
        # Highlight optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Recall/Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        
        # Add interpretation
        interpretation = (
            f"\nAUC = {roc_auc:.4f}\n"
            f"Interpretation:\n"
            f"• AUC = 1.0: Perfect classifier\n"
            f"• AUC = 0.5: Random classifier\n"
            f"• AUC > 0.9: Excellent performance\n"
            f"• AUC > 0.8: Good performance"
        )
        
        plt.figtext(0.15, 0.15, interpretation, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, y_train, y_val, y_test, save_name='class_distribution.png'):
        """
        Plot class distribution across datasets
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        datasets = [
            ('Training', y_train),
            ('Validation', y_val),
            ('Test', y_test)
        ]
        
        for idx, (name, labels) in enumerate(datasets):
            unique, counts = np.unique(labels, return_counts=True)
            
            axes[idx].bar(['No Tumor', 'Tumor'], counts, color=['skyblue', 'salmon'])
            axes[idx].set_title(f'{name} Set\n(Total: {len(labels)})', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Count')
            
            # Add percentages
            for i, count in enumerate(counts):
                percentage = (count / len(labels)) * 100
                axes[idx].text(i, count + 5, f'{count}\n({percentage:.1f}%)', 
                             ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('Class Distribution Across Datasets', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
        
        plt.show()

def create_all_plots(history, X_test, y_test, y_pred, y_pred_proba, 
                    y_train=None, y_val=None):
    """
    Convenience function to create all plots
    
    Args:
        history: Training history
        X_test: Test images
        y_test: Test labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        y_train: Training labels (optional)
        y_val: Validation labels (optional)
    """
    visualizer = Visualizer()
    
    logger.info("Creating visualizations...")
    
    # Training history
    visualizer.plot_training_history(history)
    
    # Sample predictions
    visualizer.plot_sample_predictions(X_test, y_test, y_pred, y_pred_proba)
    
    # Confusion matrix
    visualizer.plot_confusion_matrix_detailed(y_test, y_pred)
    
    # ROC curve
    visualizer.plot_roc_curve_detailed(y_test, y_pred_proba)
    
    # Class distribution
    if y_train is not None and y_val is not None:
        visualizer.plot_class_distribution(y_train, y_val, y_test)
    
    logger.info("All visualizations created successfully!")
