"""
Custom CNN Model Architecture
Implements a custom Convolutional Neural Network for brain tumor detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CustomCNN:
    """
    Custom CNN Architecture for Brain Tumor Detection
    
    ARCHITECTURE OVERVIEW:
    - 4 Convolutional blocks with increasing filters
    - Batch normalization for stable training
    - MaxPooling for spatial reduction
    - Dropout for regularization
    - Dense layers for classification
    """
    
    def __init__(self, input_shape=None, num_classes=1):
        """
        Initialize CNN model
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of output classes (1 for binary)
        """
        self.input_shape = input_shape or tuple(config.get('model', 'input_shape'))
        self.num_classes = num_classes
        self.dropout_rate = config.get('model', 'dropout_rate', default=0.5)
        self.use_batch_norm = config.get('model', 'use_batch_normalization', default=True)
        
    def build_model(self):
        """
        Build custom CNN model
        
        Returns:
            keras.Model: Compiled model
        """
        logger.info("Building Custom CNN model...")
        
        model = models.Sequential(name='CustomCNN')
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 1
        # WHY: Extract low-level features (edges, textures)
        # ============================================================
        model.add(layers.Conv2D(
            32, (3, 3), 
            activation='relu',
            padding='same',
            input_shape=self.input_shape,
            name='conv1_1'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn1_1'))
        
        model.add(layers.Conv2D(
            32, (3, 3),
            activation='relu',
            padding='same',
            name='conv1_2'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn1_2'))
        
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.25, name='dropout1'))
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 2
        # WHY: Extract mid-level features (shapes, patterns)
        # ============================================================
        model.add(layers.Conv2D(
            64, (3, 3),
            activation='relu',
            padding='same',
            name='conv2_1'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn2_1'))
        
        model.add(layers.Conv2D(
            64, (3, 3),
            activation='relu',
            padding='same',
            name='conv2_2'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn2_2'))
        
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.25, name='dropout2'))
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 3
        # WHY: Extract high-level features (complex structures)
        # ============================================================
        model.add(layers.Conv2D(
            128, (3, 3),
            activation='relu',
            padding='same',
            name='conv3_1'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn3_1'))
        
        model.add(layers.Conv2D(
            128, (3, 3),
            activation='relu',
            padding='same',
            name='conv3_2'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn3_2'))
        
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.25, name='dropout3'))
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 4
        # WHY: Extract very high-level features (tumor patterns)
        # ============================================================
        model.add(layers.Conv2D(
            256, (3, 3),
            activation='relu',
            padding='same',
            name='conv4_1'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn4_1'))
        
        model.add(layers.Conv2D(
            256, (3, 3),
            activation='relu',
            padding='same',
            name='conv4_2'
        ))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn4_2'))
        
        model.add(layers.MaxPooling2D((2, 2), name='pool4'))
        model.add(layers.Dropout(0.25, name='dropout4'))
        
        # ============================================================
        # FLATTEN AND DENSE LAYERS
        # WHY: Convert 2D features to 1D for classification
        # ============================================================
        model.add(layers.Flatten(name='flatten'))
        
        # Dense layer 1
        model.add(layers.Dense(512, activation='relu', name='dense1'))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn_dense1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_dense1'))
        
        # Dense layer 2
        model.add(layers.Dense(256, activation='relu', name='dense2'))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name='bn_dense2'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_dense2'))
        
        # ============================================================
        # OUTPUT LAYER
        # WHY: Binary classification (tumor vs no tumor)
        # Sigmoid activation outputs probability [0, 1]
        # ============================================================
        model.add(layers.Dense(
            self.num_classes,
            activation='sigmoid',
            name='output'
        ))
        
        self._log_architecture_details()
        
        return model
    
    def _log_architecture_details(self):
        """Log detailed explanation of architecture choices"""
        logger.info("\n" + "="*70)
        logger.info("CUSTOM CNN ARCHITECTURE EXPLAINED:")
        logger.info("="*70)
        
        logger.info("\n📐 LAYER CHOICES:")
        
        logger.info("\n1. CONVOLUTIONAL LAYERS (Conv2D)")
        logger.info("   WHY: Extract spatial features from images")
        logger.info("   HOW: Sliding filters detect patterns (edges, textures, shapes)")
        logger.info("   FILTERS: 32 → 64 → 128 → 256 (increasing complexity)")
        
        logger.info("\n2. ACTIVATION FUNCTION (ReLU)")
        logger.info("   WHY: Introduces non-linearity")
        logger.info("   ADVANTAGE: Solves vanishing gradient problem")
        logger.info("   FORMULA: f(x) = max(0, x)")
        
        logger.info("\n3. BATCH NORMALIZATION")
        logger.info("   WHY: Stabilizes and accelerates training")
        logger.info("   HOW: Normalizes layer inputs")
        logger.info("   BENEFIT: Allows higher learning rates")
        
        logger.info("\n4. MAX POOLING")
        logger.info("   WHY: Reduces spatial dimensions")
        logger.info("   BENEFIT: Reduces parameters, prevents overfitting")
        logger.info("   EFFECT: Makes model translation-invariant")
        
        logger.info("\n5. DROPOUT")
        logger.info("   WHY: Prevents overfitting")
        logger.info("   HOW: Randomly deactivates neurons during training")
        logger.info(f"   RATE: {self.dropout_rate} (50% neurons dropped)")
        
        logger.info("\n6. DENSE LAYERS")
        logger.info("   WHY: Learn complex combinations of features")
        logger.info("   NEURONS: 512 → 256 → 1")
        
        logger.info("\n7. SIGMOID OUTPUT")
        logger.info("   WHY: Binary classification")
        logger.info("   OUTPUT: Probability between 0 and 1")
        logger.info("   INTERPRETATION: >0.5 = Tumor, <0.5 = No Tumor")
        
        logger.info("\n" + "="*70 + "\n")
    
    def compile_model(self, model, learning_rate=None):
        """
        Compile model with optimizer and loss function
        
        Args:
            model: Keras model
            learning_rate: Learning rate (uses config if None)
            
        Returns:
            keras.Model: Compiled model
        """
        if learning_rate is None:
            learning_rate = config.get('training', 'initial_learning_rate', default=0.001)
        
        optimizer_name = config.get('training', 'optimizer', default='adam')
        
        # Choose optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # For binary classification
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info(f"Model compiled with {optimizer_name} optimizer")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Loss function: binary_crossentropy")
        
        self._log_compilation_details()
        
        return model
    
    def _log_compilation_details(self):
        """Log explanation of compilation choices"""
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPILATION EXPLAINED:")
        logger.info("="*70)
        
        logger.info("\n🎯 OPTIMIZER (Adam)")
        logger.info("   WHY: Adaptive learning rate for each parameter")
        logger.info("   ADVANTAGE: Works well with sparse gradients")
        logger.info("   COMBINES: Momentum + RMSprop")
        
        logger.info("\n📉 LOSS FUNCTION (Binary Crossentropy)")
        logger.info("   WHY: Ideal for binary classification")
        logger.info("   MEASURES: Difference between predicted and true probabilities")
        logger.info("   FORMULA: -[y*log(p) + (1-y)*log(1-p)]")
        
        logger.info("\n📊 METRICS")
        logger.info("   • Accuracy: Overall correctness")
        logger.info("   • Precision: Correct positive predictions")
        logger.info("   • Recall: Ability to find all positives")
        logger.info("   • AUC: Area under ROC curve")
        
        logger.info("\n" + "="*70 + "\n")

def create_custom_cnn():
    """
    Convenience function to create and compile custom CNN
    
    Returns:
        keras.Model: Compiled custom CNN model
    """
    cnn = CustomCNN()
    model = cnn.build_model()
    model = cnn.compile_model(model)
    
    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)
    
    return model
