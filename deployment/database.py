"""
Database Models Module
Defines database schema for users and predictions
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_bcrypt import Bcrypt
from datetime import datetime

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(UserMixin, db.Model):
    """
    User Model
    Stores user authentication information
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        """Verify password"""
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    """
    Prediction Model
    Stores prediction history for each user
    """
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)  # 'Tumor' or 'No Tumor'
    confidence_score = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Additional metadata
    model_version = db.Column(db.String(50), default='v1.0')
    processing_time = db.Column(db.Float)  # Time taken for prediction (seconds)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_class} ({self.confidence_score:.2%})>'
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'image_path': self.image_path,
            'predicted_class': self.predicted_class,
            'confidence_score': self.confidence_score,
            'prediction_date': self.prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': self.model_version,
            'processing_time': self.processing_time
        }

def init_db(app):
    """
    Initialize database
    
    Args:
        app: Flask application
    """
    db.init_app(app)
    bcrypt.init_app(app)
    
    with app.app_context():
        db.create_all()
        print("✓ Database initialized successfully")
