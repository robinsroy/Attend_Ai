from app.models import Class, Period, Student
from app import db
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ClassService:
    """Service for class and period management operations"""
    
    @staticmethod
    def get_all_classes() -> List[Class]:
        """Get all active classes"""
        try:
            return Class.query.filter_by(is_active=True).all()
        except Exception as e:
            logger.error(f"Get all classes error: {str(e)}")
            return []
    
    @staticmethod
    def get_class_by_id(class_id: int) -> Optional[Class]:
        """Get class by ID"""
        try:
            return Class.query.filter_by(id=class_id, is_active=True).first()
        except Exception as e:
            logger.error(f"Get class by ID error: {str(e)}")
            return None
    
    @staticmethod
    def get_class_students(class_id: int) -> List[Student]:
        """Get all students in a specific class"""
        try:
            return Student.query.filter_by(class_id=class_id, is_active=True).all()
        except Exception as e:
            logger.error(f"Get class students error: {str(e)}")
            return []
    
    @staticmethod
    def get_class_periods(class_id: int) -> List[Period]:
        """Get all periods for a specific class"""
        try:
            return Period.query.filter_by(class_id=class_id, is_active=True).all()
        except Exception as e:
            logger.error(f"Get class periods error: {str(e)}")
            return []
    
    @staticmethod
    def get_period_by_id(period_id: int) -> Optional[Period]:
        """Get period by ID"""
        try:
            return Period.query.filter_by(id=period_id, is_active=True).first()
        except Exception as e:
            logger.error(f"Get period by ID error: {str(e)}")
            return None
    
    @staticmethod
    def create_class(class_data: dict) -> Class:
        """Create new class"""
        try:
            class_obj = Class(
                name=class_data['name'],
                section=class_data['section'],
                grade=class_data['grade'],
                subject=class_data['subject'],
                academic_year=class_data['academic_year']
            )
            
            db.session.add(class_obj)
            db.session.commit()
            
            return class_obj
            
        except Exception as e:
            logger.error(f"Create class error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def create_period(period_data: dict) -> Period:
        """Create new period for a class"""
        try:
            period = Period(
                class_id=period_data['class_id'],
                period_number=period_data['period_number'],
                period_name=period_data['period_name'],
                start_time=period_data['start_time'],
                end_time=period_data['end_time'],
                days_of_week=period_data['days_of_week']
            )
            
            db.session.add(period)
            db.session.commit()
            
            return period
            
        except Exception as e:
            logger.error(f"Create period error: {str(e)}")
            db.session.rollback()
            raise e
