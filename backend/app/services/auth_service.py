from app.models import User
from app import db
from werkzeug.security import check_password_hash
import logging

logger = logging.getLogger(__name__)

class AuthService:
    """Authentication service for user management"""
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> User:
        """Authenticate user with username and password"""
        try:
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password_hash, password):
                return user
            
            return None
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
    
    @staticmethod
    def get_user_by_id(user_id: int) -> User:
        """Get user by ID"""
        try:
            return User.query.get(user_id)
        except Exception as e:
            logger.error(f"Get user by ID error: {str(e)}")
            return None
    
    @staticmethod
    def get_user_by_username(username: str) -> User:
        """Get user by username"""
        try:
            return User.query.filter_by(username=username).first()
        except Exception as e:
            logger.error(f"Get user by username error: {str(e)}")
            return None
    
    @staticmethod
    def create_user(user_data: dict) -> User:
        """Create new user account"""
        try:
            from werkzeug.security import generate_password_hash
            
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                password_hash=generate_password_hash(user_data['password']),
                full_name=user_data['full_name'],
                role=user_data.get('role', 'teacher')
            )
            
            db.session.add(user)
            db.session.commit()
            
            return user
            
        except Exception as e:
            logger.error(f"Create user error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def update_user(user_id: int, user_data: dict) -> User:
        """Update user information"""
        try:
            user = User.query.get(user_id)
            if not user:
                return None
            
            # Update allowed fields
            allowed_fields = ['full_name', 'email', 'role', 'is_active']
            for field in allowed_fields:
                if field in user_data:
                    setattr(user, field, user_data[field])
            
            # Handle password update separately
            if 'password' in user_data:
                from werkzeug.security import generate_password_hash
                user.password_hash = generate_password_hash(user_data['password'])
            
            db.session.commit()
            return user
            
        except Exception as e:
            logger.error(f"Update user error: {str(e)}")
            db.session.rollback()
            raise e
