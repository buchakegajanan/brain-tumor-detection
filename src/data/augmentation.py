"""
Data Augmentation Module
Implements data augmentation strategies to prevent overfitting
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataAugmentor:
    """
    Data augmentation for brain MRI images
    
    WHY DATA AUGMENTATION?
    1. Prevents overfitting by creating variations of training data
    2. Simulates real-world variations (different scanner angles, positions)
    3. Increases effective dataset size without collecting new data
    4. Makes model more robust and generalizable
    """
    
    def __init__(self):
        """Initialize augmentor with configuration"""
        self.enabled = config.get('augmentation', 'enabled', default=True)
        
        if self.enabled:
            self.rotation_range = config.get('augmentation', 'rotation_range', default=20)
            self.width_shift = config.get('augmentation', 'width_shift_range', default=0.1)
            self.height_shift = config.get('augmentation', 'height_shift_range', default=0.1)
            self.horizontal_flip = config.get('augmentation', 'horizontal_flip', default=True)
            self.vertical_flip = config.get('augmentation', 'vertical_flip', default=False)
            self.zoom_range = config.get('augmentation', 'zoom_range', default=0.15)
            self.brightness_range = config.get('augmentation', 'brightness_range', default=[0.8, 1.2])
            self.fill_mode = config.get('augmentation', 'fill_mode', default='nearest')
    
    def create_generator(self, for_training=True):
        """
        Create ImageDataGenerator
        
        Args:
            for_training (bool): If True, apply augmentation; else only rescaling
            
        Returns:
            ImageDataGenerator: Configured generator
        """
        if for_training and self.enabled:
            logger.info("Creating training data generator with augmentation...")
            
            generator = ImageDataGenerator(
                rotation_range=self.rotation_range,        # Rotate ±20° (simulates head tilt)
                width_shift_range=self.width_shift,        # Shift horizontally ±10%
                height_shift_range=self.height_shift,      # Shift vertically ±10%
                horizontal_flip=self.horizontal_flip,      # Flip left-right (brain symmetry)
                vertical_flip=self.vertical_flip,          # Usually False for MRI
                zoom_range=self.zoom_range,                # Zoom in/out ±15%
                brightness_range=self.brightness_range,    # Adjust brightness (scanner variations)
                fill_mode=self.fill_mode                   # Fill empty pixels
            )
            
            self._log_augmentation_details()
            
        else:
            logger.info("Creating validation/test data generator (no augmentation)...")
            generator = ImageDataGenerator()
        
        return generator
    
    def _log_augmentation_details(self):
        """Log detailed explanation of augmentation techniques"""
        logger.info("\n" + "="*70)
        logger.info("DATA AUGMENTATION TECHNIQUES APPLIED:")
        logger.info("="*70)
        
        logger.info(f"\n1. ROTATION (±{self.rotation_range}°)")
        logger.info("   WHY: Simulates different patient head positions")
        logger.info("   EFFECT: Model learns rotation-invariant features")
        
        logger.info(f"\n2. WIDTH/HEIGHT SHIFT (±{self.width_shift*100:.0f}%)")
        logger.info("   WHY: Simulates different MRI scan centering")
        logger.info("   EFFECT: Model becomes position-invariant")
        
        if self.horizontal_flip:
            logger.info("\n3. HORIZONTAL FLIP")
            logger.info("   WHY: Brain is roughly symmetric")
            logger.info("   EFFECT: Doubles effective dataset size")
        
        logger.info(f"\n4. ZOOM (±{self.zoom_range*100:.0f}%)")
        logger.info("   WHY: Simulates different scan distances")
        logger.info("   EFFECT: Model learns scale-invariant features")
        
        logger.info(f"\n5. BRIGHTNESS ({self.brightness_range[0]}-{self.brightness_range[1]}x)")
        logger.info("   WHY: Different MRI scanners have different intensities")
        logger.info("   EFFECT: Model becomes robust to intensity variations")
        
        logger.info("\n" + "="*70 + "\n")
    
    def get_training_generator(self, X_train, y_train, batch_size=32):
        """
        Get training data generator with augmentation
        
        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Batch size
            
        Returns:
            Generator: Training data generator
        """
        generator = self.create_generator(for_training=True)
        
        return generator.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
    
    def get_validation_generator(self, X_val, y_val, batch_size=32):
        """
        Get validation data generator (no augmentation)
        
        Args:
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size
            
        Returns:
            Generator: Validation data generator
        """
        generator = self.create_generator(for_training=False)
        
        return generator.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )

def create_augmented_generators(X_train, y_train, X_val, y_val, batch_size=None):
    """
    Convenience function to create training and validation generators
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size: Batch size (uses config if None)
        
    Returns:
        tuple: (train_generator, val_generator)
    """
    if batch_size is None:
        batch_size = config.get('data', 'batch_size', default=32)
    
    augmentor = DataAugmentor()
    
    train_gen = augmentor.get_training_generator(X_train, y_train, batch_size)
    val_gen = augmentor.get_validation_generator(X_val, y_val, batch_size)
    
    logger.info(f"Generators created with batch size: {batch_size}")
    
    return train_gen, val_gen
