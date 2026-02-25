"""
Grad-CAM Visualization Module
Implements Gradient-weighted Class Activation Mapping for model explainability
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    WHAT IS GRAD-CAM?
    - Visualization technique that shows which regions of an image
      influenced the model's decision
    - Creates a heatmap highlighting important areas
    
    WHY GRAD-CAM?
    - Builds trust in AI predictions (especially critical in healthcare)
    - Helps identify if model is looking at correct regions
    - Detects spurious correlations
    - Required for regulatory approval of medical AI
    - Educational: Shows what the model "sees"
    
    HOW IT WORKS:
    1. Get gradients of predicted class w.r.t. last conv layer
    2. Pool gradients to get importance weights
    3. Weight feature maps by importance
    4. Create heatmap showing important regions
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of target convolutional layer
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        logger.info(f"Grad-CAM initialized with layer: {self.layer_name}")
        
    def _find_target_layer(self):
        """
        Automatically find the last convolutional layer
        
        Returns:
            str: Layer name
        """
        # Try to get from config
        layer_name = config.get('gradcam', 'target_layer')
        if layer_name:
            return layer_name
        
        # Find last Conv2D layer
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                logger.info(f"Auto-detected target layer: {layer.name}")
                return layer.name
        
        raise ValueError("No convolutional layer found in model")
    
    def generate_heatmap(self, image, pred_index=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed)
            pred_index: Class index (None for highest prediction)
            
        Returns:
            np.ndarray: Heatmap
        """
        # Expand dimensions if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Gradient of class w.r.t. feature maps
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= (np.max(heatmap) + 1e-10)  # Normalize to [0, 1]
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay
            colormap: OpenCV colormap
            
        Returns:
            np.ndarray: Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in correct format
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        else:
            image = np.uint8(image)
        
        # Overlay heatmap
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed
    
    def visualize(self, image, original_image=None, save_path=None, title=None):
        """
        Visualize Grad-CAM
        
        Args:
            image: Preprocessed image
            original_image: Original image (before preprocessing)
            save_path: Path to save visualization
            title: Plot title
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(image)
        
        # Use original image if provided, else use preprocessed
        display_image = original_image if original_image is not None else image
        if len(display_image.shape) == 4:
            display_image = display_image[0]
        
        # Overlay heatmap
        overlayed = self.overlay_heatmap(display_image, heatmap)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(display_image)
        axes[0].set_title('Original MRI', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlay (Tumor Region)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_batch(self, images, original_images=None, predictions=None, 
                       true_labels=None, save_path=None, num_samples=8):
        """
        Visualize Grad-CAM for multiple images
        
        Args:
            images: Batch of preprocessed images
            original_images: Batch of original images
            predictions: Model predictions
            true_labels: True labels
            save_path: Path to save visualization
            num_samples: Number of samples to visualize
        """
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Get images
            image = images[i]
            original = original_images[i] if original_images is not None else image
            
            # Generate heatmap
            heatmap = self.generate_heatmap(image)
            overlayed = self.overlay_heatmap(original, heatmap)
            
            # Plot
            axes[i, 0].imshow(original)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Original MRI', fontweight='bold')
            
            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('Grad-CAM Heatmap', fontweight='bold')
            
            axes[i, 2].imshow(overlayed)
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title('Overlay', fontweight='bold')
            
            # Add prediction info
            if predictions is not None and true_labels is not None:
                pred = predictions[i]
                true = true_labels[i]
                pred_label = "Tumor" if pred > 0.5 else "No Tumor"
                true_label = "Tumor" if true == 1 else "No Tumor"
                
                info = f"True: {true_label} | Pred: {pred_label} ({pred:.2%})"
                axes[i, 0].text(0, -10, info, fontsize=10, 
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Batch Grad-CAM visualization saved to {save_path}")
        
        plt.show()

def explain_gradcam():
    """Print detailed explanation of Grad-CAM"""
    logger.info("\n" + "="*70)
    logger.info("GRAD-CAM EXPLAINED:")
    logger.info("="*70)
    
    logger.info("\n🔍 WHAT IS GRAD-CAM?")
    logger.info("   Gradient-weighted Class Activation Mapping")
    logger.info("   Shows which regions of the image influenced the prediction")
    
    logger.info("\n❓ WHY IS IT IMPORTANT?")
    logger.info("   1. CLINICAL TRUST: Doctors can verify AI is looking at right regions")
    logger.info("   2. ERROR DETECTION: Identifies if model learns wrong patterns")
    logger.info("   3. REGULATORY: Required for FDA approval of medical AI")
    logger.info("   4. EDUCATION: Helps understand what model 'sees'")
    logger.info("   5. DEBUGGING: Detects dataset biases")
    
    logger.info("\n🔬 HOW IT WORKS:")
    logger.info("   1. Forward pass: Get prediction")
    logger.info("   2. Backward pass: Compute gradients")
    logger.info("   3. Weight feature maps by gradient importance")
    logger.info("   4. Create heatmap showing important regions")
    
    logger.info("\n🎨 INTERPRETING THE HEATMAP:")
    logger.info("   • RED/YELLOW: High importance (tumor region)")
    logger.info("   • BLUE/PURPLE: Low importance (background)")
    logger.info("   • Model should focus on tumor, not artifacts")
    
    logger.info("\n✅ GOOD GRAD-CAM:")
    logger.info("   • Highlights actual tumor region")
    logger.info("   • Focuses on medical features")
    logger.info("   • Consistent across similar cases")
    
    logger.info("\n⚠️ BAD GRAD-CAM:")
    logger.info("   • Highlights text/labels in image")
    logger.info("   • Focuses on image borders")
    logger.info("   • Random scattered activations")
    
    logger.info("\n" + "="*70 + "\n")
