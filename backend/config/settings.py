import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'attend-ai-super-secret-key-change-in-production'
    
    # Database Settings (SQLite for development)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///attend_ai.db'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True
    }
    
    # JWT Settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Upload Settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'storage')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # Face Recognition Settings
    FACE_RECOGNITION_THRESHOLD = 0.6
    EMBEDDING_DIMENSION = 512
    VIDEO_CAPTURE_DURATION = 7  # seconds
    
    # Storage Paths
    EMBEDDINGS_PATH = os.path.join(UPLOAD_FOLDER, 'embeddings')
    VIDEOS_PATH = os.path.join(UPLOAD_FOLDER, 'videos')
    LOGS_PATH = os.path.join(UPLOAD_FOLDER, 'logs')
    
    # Ensure directories exist
    @staticmethod
    def init_app(app):
        """Initialize application-specific configuration"""
        import os
        for path in [Config.EMBEDDINGS_PATH, Config.VIDEOS_PATH, Config.LOGS_PATH]:
            os.makedirs(path, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Log SQL queries in development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
