"""
Transfer Learning Module
Implements transfer learning using pre-trained models (MobileNetV2, ResNet50)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TransferLearningModel:
    """
    Transfer Learning for Brain Tumor Detection
    
    WHAT IS TRANSFER LEARNING?
    - Use a model pre-trained on ImageNet (1.4M images, 1000 classes)
    - Leverage learned features (edges, textures, shapes)
    - Fine-tune for our specific task (brain tumor detection)
    
    WHY TRANSFER LEARNING?
    - Requires less training data
    - Faster convergence
    - Better performance than training from scratch
    - Pre-trained models already understand visual features
    """
    
    def __init__(self, architecture='mobilenetv2', input_shape=None, num_classes=1):
        """
        Initialize transfer learning model
        
        Args:
            architecture (str): 'mobilenetv2' or 'resnet50'
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        self.architecture = architecture.lower()
        self.input_shape = input_shape or tuple(config.get('model', 'input_shape'))
        self.num_classes = num_classes
        self.dropout_rate = config.get('model', 'dropout_rate', default=0.5)
        
        self.use_pretrained = config.get('model', 'transfer_learning', 'use_pretrained', default=True)
        self.freeze_base = config.get('model', 'transfer_learning', 'freeze_base', default=False)
        self.fine_tune_from = config.get('model', 'transfer_learning', 'fine_tune_from_layer', default=100)
        
    def build_model(self):
        """
        Build transfer learning model
        
        Returns:
            keras.Model: Compiled model
        """
        logger.info(f"Building {self.architecture.upper()} transfer learning model...")
        
        # Load base model
        if self.architecture == 'mobilenetv2':
            base_model = self._build_mobilenetv2()
        elif self.architecture == 'resnet50':
            base_model = self._build_resnet50()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Build complete model
        model = self._add_custom_layers(base_model)
        
        self._log_transfer_learning_details()
        
        return model
    
    def _build_mobilenetv2(self):
        """
        Build MobileNetV2 base model
        
        WHY MOBILENETV2?
        - Lightweight (3.4M parameters)
        - Fast inference (good for deployment)
        - Depthwise separable convolutions (efficient)
        - Good accuracy-speed tradeoff
        
        Returns:
            keras.Model: MobileNetV2 base model
        """
        weights = 'imagenet' if self.use_pretrained else None
        
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,  # Remove classification head
            weights=weights,
            pooling='avg'  # Global average pooling
        )
        
        logger.info("MobileNetV2 base model loaded")
        logger.info(f"Pre-trained weights: {weights}")
        logger.info(f"Total layers: {len(base_model.layers)}")
        
        return base_model
    
    def _build_resnet50(self):
        """
        Build ResNet50 base model
        
        WHY RESNET50?
        - Deeper network (50 layers)
        - Residual connections (solves vanishing gradient)
        - Higher accuracy than MobileNetV2
        - More parameters (25M) but better performance
        
        Returns:
            keras.Model: ResNet50 base model
        """
        weights = 'imagenet' if self.use_pretrained else None
        
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights,
            pooling='avg'
        )
        
        logger.info("ResNet50 base model loaded")
        logger.info(f"Pre-trained weights: {weights}")
        logger.info(f"Total layers: {len(base_model.layers)}")
        
        return base_model
    
    def _add_custom_layers(self, base_model):
        """
        Add custom classification layers on top of base model
        
        Args:
            base_model: Pre-trained base model
            
        Returns:
            keras.Model: Complete model
        """
        # Freeze or unfreeze base model
        if self.freeze_base:
            logger.info("Freezing base model layers")
            base_model.trainable = False
        else:
            logger.info(f"Fine-tuning from layer {self.fine_tune_from}")
            base_model.trainable = True
            
            # Freeze early layers, fine-tune later layers
            for layer in base_model.layers[:self.fine_tune_from]:
                layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Custom classification head
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='sigmoid',
            name='output'
        )(x)
        
        model = keras.Model(inputs, outputs, name=f'{self.architecture}_transfer')
        
        # Count trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total_params = sum([tf.size(w).numpy() for w in model.weights])
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        
        return model
    
    def _log_transfer_learning_details(self):
        """Log explanation of transfer learning"""
        logger.info("\n" + "="*70)
        logger.info("TRANSFER LEARNING EXPLAINED:")
        logger.info("="*70)
        
        logger.info("\n🔄 WHAT IS TRANSFER LEARNING?")
        logger.info("   • Use knowledge from one task (ImageNet) for another (tumor detection)")
        logger.info("   • Pre-trained model already knows visual features")
        logger.info("   • We only train the classification head")
        
        logger.info("\n✅ ADVANTAGES:")
        logger.info("   • Requires less training data")
        logger.info("   • Faster training (fewer parameters to update)")
        logger.info("   • Better performance (leverages ImageNet knowledge)")
        logger.info("   • Reduces overfitting")
        
        logger.info(f"\n🏗️ ARCHITECTURE: {self.architecture.upper()}")
        
        if self.architecture == 'mobilenetv2':
            logger.info("   • Lightweight and fast")
            logger.info("   • Depthwise separable convolutions")
            logger.info("   • Ideal for mobile/edge deployment")
            logger.info("   • Parameters: ~3.4M")
        else:
            logger.info("   • Deep residual network")
            logger.info("   • Skip connections prevent vanishing gradients")
            logger.info("   • Higher accuracy")
            logger.info("   • Parameters: ~25M")
        
        logger.info("\n🎯 FINE-TUNING STRATEGY:")
        if self.freeze_base:
            logger.info("   • Base model: FROZEN (feature extractor)")
            logger.info("   • Only training: Classification head")
        else:
            logger.info(f"   • Freezing first {self.fine_tune_from} layers")
            logger.info("   • Fine-tuning later layers")
            logger.info("   • Allows model to adapt to MRI images")
        
        logger.info("\n" + "="*70 + "\n")
    
    def compile_model(self, model, learning_rate=None):
        """
        Compile model
        
        Args:
            model: Keras model
            learning_rate: Learning rate
            
        Returns:
            keras.Model: Compiled model
        """
        if learning_rate is None:
            learning_rate = config.get('training', 'initial_learning_rate', default=0.001)
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info(f"Model compiled with Adam optimizer (lr={learning_rate})")
        
        return model

def create_transfer_learning_model(architecture='mobilenetv2'):
    """
    Convenience function to create transfer learning model
    
    Args:
        architecture (str): 'mobilenetv2' or 'resnet50'
        
    Returns:
        keras.Model: Compiled transfer learning model
    """
    tl_model = TransferLearningModel(architecture=architecture)
    model = tl_model.build_model()
    model = tl_model.compile_model(model)
    
    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)
    
    return model
