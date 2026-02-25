"""
Main Training Script
Complete pipeline for training brain tumor detection model
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data.data_loader import load_and_prepare_data
from src.data.preprocessing import preprocess_data
from src.models.transfer_learning import create_transfer_learning_model
from src.models.cnn_model import create_custom_cnn
from src.models.model_trainer import ModelTrainer
from src.visualization.plots import create_all_plots
from src.utils.metrics import MetricsCalculator
from src.utils.config import config
from src.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

def main():
    """Main training pipeline"""
    
    logger.info("="*70)
    logger.info("BRAIN TUMOR DETECTION - TRAINING PIPELINE")
    logger.info("="*70)
    
    # Create directories
    config.create_directories()
    
    # Step 1: Load data
    logger.info("\n[STEP 1] Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    # Step 2: Preprocess data
    logger.info("\n[STEP 2] Preprocessing data...")
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    
    # Step 3: Create model
    logger.info("\n[STEP 3] Creating model...")
    architecture = config.get('model', 'architecture', default='mobilenetv2')
    
    if architecture == 'custom_cnn':
        model = create_custom_cnn()
    else:
        model = create_transfer_learning_model(architecture)
    
    # Step 4: Train model
    logger.info("\n[STEP 4] Training model...")
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluate model
    logger.info("\n[STEP 5] Evaluating model on test set...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_pred_proba = y_pred_proba.flatten()
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(y_test, y_pred, y_pred_proba)
    metrics = metrics_calc.print_metrics()
    
    # Step 6: Create visualizations
    logger.info("\n[STEP 6] Creating visualizations...")
    create_all_plots(history, X_test, y_test, y_pred, y_pred_proba, y_train, y_val)
    
    # Step 7: Save model
    logger.info("\n[STEP 7] Saving model...")
    model_path = config.get('training', 'model_checkpoint', 'filepath', default='models/best_model.h5')
    trainer.save_model(model_path)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Test Recall: {metrics['recall']*100:.2f}%")
    logger.info(f"Test Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"Test F1-Score: {metrics['f1_score']*100:.2f}%")
    
    if 'auc_roc' in metrics:
        logger.info(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review visualizations in results/plots/")
    logger.info("2. Test Grad-CAM: python test_gradcam.py")
    logger.info("3. Run web app: python deployment/app.py")
    logger.info("="*70 + "\n")

if __name__ == '__main__':
    main()
