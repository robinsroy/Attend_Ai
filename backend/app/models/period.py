"""
Period model for class scheduling
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, time
from app import db

class Period(db.Model):
    """Period model for class scheduling"""
    
    __tablename__ = 'periods'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    subject = db.Column(db.String(100), nullable=True)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    day_of_week = db.Column(db.String(20), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    class_obj = db.relationship('Class', back_populates='periods')
    attendance_sessions = db.relationship('AttendanceSession', back_populates='period', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Period {self.name}>'
    
    def to_dict(self):
        """Convert period to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'class_id': self.class_id,
            'subject': self.subject,
            'start_time': self.start_time.strftime('%H:%M') if self.start_time else None,
            'end_time': self.end_time.strftime('%H:%M') if self.end_time else None,
            'day_of_week': self.day_of_week,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
