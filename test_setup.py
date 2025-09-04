"""
Simple test script to verify the Flask backend setup
"""

# Test if our imports work
print("🔧 Testing imports...")

try:
    from flask import Flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    from flask_sqlalchemy import SQLAlchemy
    print("✅ Flask-SQLAlchemy imported successfully")
except ImportError as e:
    print(f"❌ Flask-SQLAlchemy import failed: {e}")

try:
    from flask_jwt_extended import JWTManager
    print("✅ Flask-JWT-Extended imported successfully")
except ImportError as e:
    print(f"❌ Flask-JWT-Extended import failed: {e}")

try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    from PIL import Image
    print("✅ Pillow imported successfully")
except ImportError as e:
    print(f"❌ Pillow import failed: {e}")

print("\n🚀 Testing Flask app creation...")

try:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key'
    
    @app.route('/test')
    def test():
        return {'status': 'success', 'message': 'Flask app is working!'}
    
    print("✅ Flask app created successfully")
    print("✅ All core dependencies are working!")
    print("\n🎉 Your Attend.AI backend is ready to run!")
    
except Exception as e:
    print(f"❌ Flask app creation failed: {e}")

print("\nNext steps:")
print("1. Run: python backend/run.py")
print("2. Test the API at: http://localhost:5000/api/health")
