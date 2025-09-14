"""
Database models for the Attend.AI system

This module contains all SQLAlchemy models for the attendance management system.
Models are organized into separate files for better maintainability.
"""

# Import all models
from .user import User
from .student import Student
from .class_model import Class
from .period import Period
from .attendance import AttendanceSession, AttendanceRecord
from .enrollment_frame import EnrollmentFrame

__all__ = ['User', 'Student', 'Class', 'Period', 'AttendanceSession', 'AttendanceRecord', 'EnrollmentFrame']
