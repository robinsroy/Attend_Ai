from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from config.settings import Config
import os

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_class=Config):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    CORS(app)
    
    # Import models to register them with SQLAlchemy
    from app.models.user import User
    from app.models.student import Student  
    from app.models.class_model import Class
    from app.models.period import Period
    from app.models.attendance import AttendanceSession, AttendanceRecord
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.students import students_bp
    from app.routes.classes import classes_bp
    from app.routes.attendance import attendance_bp
    from app.routes.face_recognition import face_recognition_bp
    from app.routes.video_enrollment import video_enrollment_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(students_bp, url_prefix='/api/students')
    app.register_blueprint(classes_bp, url_prefix='/api/classes')
    app.register_blueprint(attendance_bp, url_prefix='/api/attendance')
    app.register_blueprint(face_recognition_bp, url_prefix='/api/face')
    app.register_blueprint(video_enrollment_bp, url_prefix='/api')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return {'status': 'healthy', 'message': 'Attend.AI API is running'}
    
    return app
