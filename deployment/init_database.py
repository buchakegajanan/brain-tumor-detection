"""
Database Initialization Script
Run this to create the database tables
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from deployment.database import db, bcrypt, User, Prediction

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_tumor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)

# Create tables
with app.app_context():
    print("Creating database tables...")
    db.create_all()
    print("[OK] Database tables created successfully!")
    print(f"[OK] Database location: {os.path.abspath('brain_tumor.db')}")
    
    # Check tables
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"[OK] Tables created: {', '.join(tables)}")
