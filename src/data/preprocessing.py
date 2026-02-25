"""
Image Preprocessing Module
Handles image preprocessing, normalization, and enhancement
"""

import numpy as np
import cv2
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ImagePreprocessor:
    """Preprocess brain MRI images"""
    
    def __init__(self):
        """Initialize preprocessor with configuration"""
        self.normalize = config.get('preprocessing', 'normalize', default=True)
        self.apply_clahe = config.get('preprocessing', 'apply_clahe', default=True)
        self.apply_gaussian = config.get('preprocessing', 'apply_gaussian_blur', default=True)
        self.remove_noise = config.get('preprocessing', 'remove_noise', default=True)
        
    def preprocess(self, images):
        """
        Apply all preprocessing steps
        
        Args:
            images (np.ndarray): Array of images
            
        Returns:
            np.ndarray: Preprocessed images
        """
        logger.info("Starting image preprocessing...")
        
        processed_images = images.copy()
        
        # Step 1: Noise removal
        if self.remove_noise:
            logger.info("Applying noise removal...")
            processed_images = self._remove_noise(processed_images)
        
        # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.apply_clahe:
            logger.info("Applying CLAHE for contrast enhancement...")
            processed_images = self._apply_clahe(processed_images)
        
        # Step 3: Gaussian blur (optional, for smoothing)
        if self.apply_gaussian:
            logger.info("Applying Gaussian blur...")
            processed_images = self._apply_gaussian_blur(processed_images)
        
        # Step 4: Normalization
        if self.normalize:
            logger.info("Normalizing images to [0, 1] range...")
            processed_images = self._normalize(processed_images)
        
        logger.info("Preprocessing completed!")
        return processed_images
    
    def _remove_noise(self, images):
        """
        Remove noise using bilateral filter
        
        WHY: MRI images often contain scanner artifacts and noise
        Bilateral filter preserves edges while removing noise
        
        Args:
            images (np.ndarray): Input images
            
        Returns:
            np.ndarray: Denoised images
        """
        denoised = np.zeros_like(images)
        
        for i in range(len(images)):
            # Bilateral filter: removes noise while keeping edges sharp
            # d=9: diameter of pixel neighborhood
            # sigmaColor=75: filter sigma in color space
            # sigmaSpace=75: filter sigma in coordinate space
            denoised[i] = cv2.bilateralFilter(
                images[i].astype(np.uint8), 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            )
        
        return denoised
    
    def _apply_clahe(self, images):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        WHY: Enhances local contrast in MRI images
        Makes subtle features more visible
        Helps model detect tumor boundaries better
        
        Args:
            images (np.ndarray): Input images
            
        Returns:
            np.ndarray: Contrast-enhanced images
        """
        enhanced = np.zeros_like(images)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        for i in range(len(images)):
            # Convert to LAB color space
            lab = cv2.cvtColor(images[i].astype(np.uint8), cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _apply_gaussian_blur(self, images):
        """
        Apply Gaussian blur for smoothing
        
        WHY: Reduces high-frequency noise
        Smooths image while preserving important features
        
        Args:
            images (np.ndarray): Input images
            
        Returns:
            np.ndarray: Smoothed images
        """
        kernel_size = tuple(config.get('preprocessing', 'gaussian_kernel', default=[3, 3]))
        blurred = np.zeros_like(images)
        
        for i in range(len(images)):
            blurred[i] = cv2.GaussianBlur(images[i], kernel_size, 0)
        
        return blurred
    
    def _normalize(self, images):
        """
        Normalize images to [0, 1] range
        
        WHY: Neural networks work better with normalized inputs
        - Faster convergence during training
        - Prevents gradient explosion/vanishing
        - Makes learning rate selection easier
        
        Args:
            images (np.ndarray): Input images
            
        Returns:
            np.ndarray: Normalized images
        """
        # Normalize to [0, 1]
        normalized = images.astype('float32') / 255.0
        
        return normalized
    
    def standardize(self, images):
        """
        Standardize images (zero mean, unit variance)
        
        WHY: Alternative to normalization
        Centers data around 0, useful for some architectures
        
        Args:
            images (np.ndarray): Input images
            
        Returns:
            np.ndarray: Standardized images
        """
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        
        standardized = (images - mean) / (std + 1e-7)
        
        return standardized

def preprocess_data(X_train, X_val, X_test):
    """
    Convenience function to preprocess all datasets
    
    Args:
        X_train: Training images
        X_val: Validation images
        X_test: Test images
        
    Returns:
        tuple: (X_train_processed, X_val_processed, X_test_processed)
    """
    preprocessor = ImagePreprocessor()
    
    logger.info("Preprocessing training data...")
    X_train_processed = preprocessor.preprocess(X_train)
    
    logger.info("Preprocessing validation data...")
    X_val_processed = preprocessor.preprocess(X_val)
    
    logger.info("Preprocessing test data...")
    X_test_processed = preprocessor.preprocess(X_test)
    
    logger.info(f"Preprocessed data shapes:")
    logger.info(f"  Train: {X_train_processed.shape}")
    logger.info(f"  Val: {X_val_processed.shape}")
    logger.info(f"  Test: {X_test_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed
