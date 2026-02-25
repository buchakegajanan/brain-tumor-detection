"""
Evaluation Metrics Module
Calculates and displays model performance metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """Calculate and display evaluation metrics"""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize metrics calculator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
        
    def calculate_all_metrics(self):
        """
        Calculate all evaluation metrics
        
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Sensitivity (same as recall)
        metrics['sensitivity'] = metrics['recall']
        
        # AUC-ROC (if probabilities available)
        if self.y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        
        # Confusion matrix components
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def print_metrics(self):
        """Print all metrics in a formatted way"""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"   F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"   Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        
        if 'auc_roc' in metrics:
            print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
        
        print(f"\n🔢 Confusion Matrix Components:")
        print(f"   True Positives:  {metrics['true_positives']}")
        print(f"   True Negatives:  {metrics['true_negatives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        
        print("\n" + "="*60)
        
        # Clinical interpretation
        self._print_clinical_interpretation(metrics)
        
        return metrics
    
    def _print_clinical_interpretation(self, metrics):
        """Print clinical interpretation of metrics"""
        print("\n🏥 CLINICAL INTERPRETATION:")
        print("-" * 60)
        
        print(f"\n1. ACCURACY ({metrics['accuracy']*100:.2f}%):")
        print("   → Overall correctness of the model")
        print("   → Baseline metric for model performance")
        
        print(f"\n2. PRECISION ({metrics['precision']*100:.2f}%):")
        print("   → When model predicts TUMOR, it's correct {:.1f}% of the time".format(
            metrics['precision']*100))
        print("   → High precision = Fewer false alarms")
        print("   → Reduces unnecessary patient anxiety and follow-up tests")
        
        print(f"\n3. RECALL/SENSITIVITY ({metrics['recall']*100:.2f}%):")
        print("   → Model catches {:.1f}% of actual tumors".format(metrics['recall']*100))
        print("   → ⚠️ MOST CRITICAL METRIC in medical diagnosis")
        print("   → High recall = Fewer missed tumors (false negatives)")
        print("   → Missing a tumor can be life-threatening")
        
        print(f"\n4. SPECIFICITY ({metrics['specificity']*100:.2f}%):")
        print("   → Model correctly identifies {:.1f}% of healthy patients".format(
            metrics['specificity']*100))
        print("   → High specificity = Fewer false positives")
        print("   → Avoids unnecessary treatments for healthy patients")
        
        print(f"\n5. F1-SCORE ({metrics['f1_score']*100:.2f}%):")
        print("   → Harmonic mean of precision and recall")
        print("   → Balances false positives and false negatives")
        print("   → Useful when classes are imbalanced")
        
        # Recommendations based on metrics
        print("\n💡 RECOMMENDATIONS:")
        if metrics['recall'] < 0.95:
            print("   ⚠️ Recall is below 95% - Consider:")
            print("      • Adjusting decision threshold")
            print("      • Adding more tumor samples")
            print("      • Using class weights")
        
        if metrics['precision'] < 0.90:
            print("   ⚠️ Precision is below 90% - May cause:")
            print("      • Increased false alarms")
            print("      • Unnecessary patient stress")
        
        if metrics['recall'] >= 0.95 and metrics['precision'] >= 0.90:
            print("   ✅ Excellent performance! Model is clinically viable.")
        
        print("-" * 60)
    
    def get_classification_report(self):
        """Get detailed classification report"""
        return classification_report(
            self.y_true, 
            self.y_pred,
            target_names=['No Tumor', 'Tumor'],
            digits=4
        )
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.y_pred_proba is None:
            logger.warning("Predicted probabilities not available. Cannot plot ROC curve.")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()

def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y_train: Training labels
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    logger.info(f"Calculated class weights: {class_weights}")
    return class_weights
