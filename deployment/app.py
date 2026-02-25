"""
Flask Web Application
Main application for brain tumor detection with user authentication
"""

import os
import sys
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not available. Install it to use predictions.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.database import db, init_db, Prediction
from deployment.auth import auth_bp, init_auth

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_tumor.db'  # Using SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'deployment/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize extensions
init_db(app)
init_auth(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')

# Load model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.h5')
model = None

def load_model():
    """Load trained model"""
    global model
    if not TF_AVAILABLE:
        print("⚠ TensorFlow not installed. Cannot load model.")
        return
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠ Model not found at {MODEL_PATH}")
            print("  Please train the model first!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """
    Preprocess image for prediction
    
    Args:
        image_path (str): Path to image
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (128, 128))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def generate_gradcam(image_path, prediction):
    """
    Generate Grad-CAM visualization
    
    Args:
        image_path (str): Path to image
        prediction (float): Model prediction
        
    Returns:
        str: Path to Grad-CAM image
    """
    try:
        # Import GradCAM
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.visualization.gradcam import GradCAM
        
        # Load and preprocess image
        img = preprocess_image(image_path)
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (128, 128))
        
        # Generate Grad-CAM
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(img)
        overlayed = gradcam.overlay_heatmap(original_img, heatmap)
        
        # Save Grad-CAM image
        filename = os.path.basename(image_path)
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        
        overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(gradcam_path, overlayed_bgr)
        
        return gradcam_filename
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction page"""
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{current_user.id}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            # Make prediction
            if model is None:
                flash('Model not loaded! Please contact administrator.', 'danger')
                return redirect(request.url)
            
            try:
                start_time = time.time()
                
                # Preprocess and predict
                img = preprocess_image(filepath)
                prediction = model.predict(img, verbose=0)[0][0]
                
                processing_time = time.time() - start_time
                
                # Interpret prediction
                predicted_class = "Tumor" if prediction > 0.5 else "No Tumor"
                confidence = prediction if prediction > 0.5 else (1 - prediction)
                
                # Generate Grad-CAM
                gradcam_filename = generate_gradcam(filepath, prediction)
                
                # Save to database
                new_prediction = Prediction(
                    user_id=current_user.id,
                    image_path=filename,
                    predicted_class=predicted_class,
                    confidence_score=float(confidence),
                    processing_time=processing_time
                )
                db.session.add(new_prediction)
                db.session.commit()
                
                return render_template('result.html',
                                     prediction=predicted_class,
                                     confidence=confidence * 100,
                                     image_path=filename,
                                     gradcam_path=gradcam_filename,
                                     processing_time=processing_time)
            
            except Exception as e:
                flash(f'Error during prediction: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type! Please upload PNG, JPG, or JPEG.', 'danger')
            return redirect(request.url)
    
    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    """Prediction history page"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                  .order_by(Prediction.prediction_date.desc())\
                                  .all()
    return render_template('history.html', predictions=predictions)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if prediction belongs to current user
    if prediction.user_id != current_user.id:
        flash('Unauthorized action!', 'danger')
        return redirect(url_for('history'))
    
    # Delete image files
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Delete gradcam if exists
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{prediction.image_path}")
        if os.path.exists(gradcam_path):
            os.remove(gradcam_path)
    except Exception as e:
        print(f"Error deleting files: {e}")
    
    # Delete from database
    db.session.delete(prediction)
    db.session.commit()
    
    flash('Prediction deleted successfully!', 'success')
    return redirect(url_for('history'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load model
    load_model()
    
    # Run app
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR DETECTION WEB APPLICATION")
    print("="*70)
    print("\n✓ Server starting...")
    print("✓ Access the application at: http://localhost:5000")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
