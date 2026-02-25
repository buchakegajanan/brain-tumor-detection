"""
Grad-CAM Testing Script
Test Grad-CAM visualization on sample images
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from src.data.data_loader import load_and_prepare_data
from src.data.preprocessing import preprocess_data
from src.visualization.gradcam import GradCAM, explain_gradcam
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Test Grad-CAM visualization"""
    
    logger.info("="*70)
    logger.info("GRAD-CAM VISUALIZATION TEST")
    logger.info("="*70)
    
    # Load model
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first: python train.py")
        return
    
    logger.info(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Load test data
    logger.info("Loading test data...")
    _, _, X_test, _, _, y_test = load_and_prepare_data()
    X_test_original = X_test.copy()
    
    # Preprocess
    from src.data.preprocessing import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    X_test = preprocessor.preprocess(X_test)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X_test, verbose=0)
    
    # Initialize Grad-CAM
    logger.info("Initializing Grad-CAM...")
    gradcam = GradCAM(model)
    
    # Explain Grad-CAM
    explain_gradcam()
    
    # Visualize samples
    logger.info("\nGenerating Grad-CAM visualizations...")
    
    # Select samples (mix of tumor and no tumor)
    tumor_indices = np.where(y_test == 1)[0][:4]
    no_tumor_indices = np.where(y_test == 0)[0][:4]
    sample_indices = np.concatenate([tumor_indices, no_tumor_indices])
    
    # Visualize batch
    gradcam.visualize_batch(
        X_test[sample_indices],
        X_test_original[sample_indices],
        predictions[sample_indices].flatten(),
        y_test[sample_indices],
        save_path='results/plots/gradcam_samples.png',
        num_samples=8
    )
    
    logger.info("\n" + "="*70)
    logger.info("GRAD-CAM VISUALIZATION COMPLETED!")
    logger.info("="*70)
    logger.info("\nVisualizations saved to: results/plots/gradcam_samples.png")
    logger.info("="*70 + "\n")

if __name__ == '__main__':
    main()
