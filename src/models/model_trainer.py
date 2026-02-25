"""
Model Training Module
Handles model training with callbacks, checkpoints, and monitoring
"""

import os
import numpy as np
from tensorflow import keras
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.metrics import calculate_class_weights

logger = get_logger(__name__)

class ModelTrainer:
    """Train deep learning models with callbacks and monitoring"""
    
    def __init__(self, model):
        """
        Initialize trainer
        
        Args:
            model: Keras model to train
        """
        self.model = model
        self.epochs = config.get('training', 'epochs', default=50)
        self.callbacks = []
        self.history = None
        
    def setup_callbacks(self):
        """
        Setup training callbacks
        
        CALLBACKS EXPLAINED:
        1. EarlyStopping: Stop training when validation loss stops improving
        2. ReduceLROnPlateau: Reduce learning rate when stuck
        3. ModelCheckpoint: Save best model during training
        """
        callbacks = []
        
        # Early Stopping
        if config.get('training', 'early_stopping', 'enabled', default=True):
            early_stop = keras.callbacks.EarlyStopping(
                monitor=config.get('training', 'early_stopping', 'monitor', default='val_loss'),
                patience=config.get('training', 'early_stopping', 'patience', default=10),
                restore_best_weights=config.get('training', 'early_stopping', 'restore_best_weights', default=True),
                verbose=1
            )
            callbacks.append(early_stop)
            logger.info("✓ Early Stopping enabled")
        
        # Reduce Learning Rate
        if config.get('training', 'reduce_lr', 'enabled', default=True):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor=config.get('training', 'reduce_lr', 'monitor', default='val_loss'),
                factor=config.get('training', 'reduce_lr', 'factor', default=0.5),
                patience=config.get('training', 'reduce_lr', 'patience', default=5),
                min_lr=config.get('training', 'reduce_lr', 'min_lr', default=0.00001),
                verbose=1
            )
            callbacks.append(reduce_lr)
            logger.info("✓ ReduceLROnPlateau enabled")
        
        # Model Checkpoint
        if config.get('training', 'model_checkpoint', 'enabled', default=True):
            checkpoint_path = config.get('training', 'model_checkpoint', 'filepath', default='models/best_model.h5')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=config.get('training', 'model_checkpoint', 'monitor', default='val_accuracy'),
                save_best_only=config.get('training', 'model_checkpoint', 'save_best_only', default=True),
                mode=config.get('training', 'model_checkpoint', 'mode', default='max'),
                verbose=1
            )
            callbacks.append(checkpoint)
            logger.info(f"✓ ModelCheckpoint enabled: {checkpoint_path}")
        
        self.callbacks = callbacks
        self._log_callbacks_explanation()
        
        return callbacks
    
    def _log_callbacks_explanation(self):
        """Log explanation of callbacks"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING CALLBACKS EXPLAINED:")
        logger.info("="*70)
        
        logger.info("\n1. EARLY STOPPING")
        logger.info("   WHY: Prevents overfitting and saves time")
        logger.info("   HOW: Stops training when validation loss stops improving")
        logger.info("   PATIENCE: Waits 10 epochs before stopping")
        logger.info("   BENEFIT: Automatically finds optimal training duration")
        
        logger.info("\n2. REDUCE LEARNING RATE ON PLATEAU")
        logger.info("   WHY: Helps escape local minima")
        logger.info("   HOW: Reduces LR when validation loss plateaus")
        logger.info("   FACTOR: Multiplies LR by 0.5")
        logger.info("   BENEFIT: Fine-tunes model in later epochs")
        
        logger.info("\n3. MODEL CHECKPOINT")
        logger.info("   WHY: Saves best model during training")
        logger.info("   HOW: Monitors validation accuracy")
        logger.info("   SAVES: Only when validation accuracy improves")
        logger.info("   BENEFIT: Prevents losing best model if training continues")
        
        logger.info("\n" + "="*70 + "\n")
    
    def train(self, X_train, y_train, X_val, y_val, use_class_weights=True):
        """
        Train model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            use_class_weights: Whether to use class weights
            
        Returns:
            History: Training history
        """
        logger.info("Starting model training...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {self.epochs}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Calculate class weights if needed
        class_weights = None
        if use_class_weights and config.get('training', 'use_class_weights', default=True):
            class_weights = calculate_class_weights(y_train)
            logger.info(f"Using class weights: {class_weights}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=config.get('data', 'batch_size', default=32),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        return self.history
    
    def train_with_generator(self, train_generator, val_generator, steps_per_epoch, 
                            validation_steps, use_class_weights=True, y_train=None):
        """
        Train model using data generators
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
            use_class_weights: Whether to use class weights
            y_train: Training labels (for class weights calculation)
            
        Returns:
            History: Training history
        """
        logger.info("Starting model training with data generators...")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        logger.info(f"Epochs: {self.epochs}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Calculate class weights if needed
        class_weights = None
        if use_class_weights and config.get('training', 'use_class_weights', default=True):
            if y_train is not None:
                class_weights = calculate_class_weights(y_train)
                logger.info(f"Using class weights: {class_weights}")
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        return self.history
    
    def get_history(self):
        """Get training history"""
        return self.history
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to model file
            
        Returns:
            keras.Model: Loaded model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Convenience function to train model
    
    Args:
        model: Keras model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        
    Returns:
        tuple: (trained_model, history)
    """
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    return model, history
