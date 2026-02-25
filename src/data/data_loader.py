"""
Data Loading Module
Handles dataset loading, splitting, and preparation
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataLoader:
    """Load and prepare brain MRI dataset"""
    
    def __init__(self, data_path=None):
        """
        Initialize data loader
        
        Args:
            data_path (str): Path to raw data directory
        """
        self.data_path = data_path or config.get('data', 'raw_data_path')
        self.image_size = tuple(config.get('data', 'image_size'))
        self.tumor_folder = config.get('data', 'tumor_folder', default='Tumor')
        self.no_tumor_folder = config.get('data', 'no_tumor_folder', default='No_Tumor')
        
        self.images = []
        self.labels = []
        
    def load_data(self):
        """
        Load images and labels from directory structure
        
        Returns:
            tuple: (images, labels)
        """
        logger.info("Loading dataset...")
        
        # Load tumor images (label = 1)
        tumor_path = os.path.join(self.data_path, self.tumor_folder)
        self._load_images_from_folder(tumor_path, label=1)
        
        # Load no tumor images (label = 0)
        no_tumor_path = os.path.join(self.data_path, self.no_tumor_folder)
        self._load_images_from_folder(no_tumor_path, label=0)
        
        # Convert to numpy arrays
        self.images = np.array(self.images, dtype='float32')
        self.labels = np.array(self.labels, dtype='int32')
        
        logger.info(f"Loaded {len(self.images)} images")
        logger.info(f"Image shape: {self.images.shape}")
        logger.info(f"Tumor samples: {np.sum(self.labels == 1)}")
        logger.info(f"No tumor samples: {np.sum(self.labels == 0)}")
        
        return self.images, self.labels
    
    def _load_images_from_folder(self, folder_path, label):
        """
        Load all images from a folder
        
        Args:
            folder_path (str): Path to folder
            label (int): Label for images in this folder
        """
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Loading {len(image_files)} images from {folder_path}")
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, self.image_size)
                
                self.images.append(img)
                self.labels.append(label)
                
            except Exception as e:
                logger.error(f"Error loading {img_path}: {e}")
    
    def split_data(self, images=None, labels=None, test_size=0.15, val_size=0.15, 
                   random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            images: Image data (uses self.images if None)
            labels: Label data (uses self.labels if None)
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set (from remaining data)
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if images is None:
            images = self.images
        if labels is None:
            labels = self.labels
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        logger.info(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        logger.info(f"  Test set: {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, labels):
        """
        Get class distribution
        
        Args:
            labels: Label array
            
        Returns:
            dict: Class distribution
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        logger.info("Class distribution:")
        for class_label, count in distribution.items():
            class_name = "Tumor" if class_label == 1 else "No Tumor"
            percentage = (count / len(labels)) * 100
            logger.info(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        return distribution

def load_and_prepare_data():
    """
    Convenience function to load and prepare data
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    loader = DataLoader()
    images, labels = loader.load_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        images, labels,
        test_size=config.get('data', 'test_split', default=0.15),
        val_size=config.get('data', 'validation_split', default=0.15),
        random_state=config.get('data', 'random_seed', default=42)
    )
    
    # Show distribution
    logger.info("\nTraining set distribution:")
    loader.get_class_distribution(y_train)
    
    logger.info("\nValidation set distribution:")
    loader.get_class_distribution(y_val)
    
    logger.info("\nTest set distribution:")
    loader.get_class_distribution(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
