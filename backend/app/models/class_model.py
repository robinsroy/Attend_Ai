from datetime import datetime
from app import db

class Class(db.Model):
    """Class/Section model (e.g., 10-A Physics, 10-B Computer Science)"""
    __tablename__ = 'classes'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # e.g., "10-A Physics"
    section = db.Column(db.String(10), nullable=True)  # e.g., "A", "B"
    grade = db.Column(db.String(10), nullable=True)  # e.g., "10", "11", "12"
    subject = db.Column(db.String(50), nullable=True)  # e.g., "Physics", "Computer Science"
    academic_year = db.Column(db.String(10), nullable=False)  # e.g., "2024-25"
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    room_number = db.Column(db.String(20), nullable=True)  # e.g., "A-101"
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    students = db.relationship("Student", back_populates="class_obj")
    attendance_sessions = db.relationship("AttendanceSession", back_populates="class_obj")
    periods = db.relationship("Period", back_populates="class_obj")
    teacher = db.relationship("User", back_populates="classes")
    
    def __repr__(self):
        return f'<Class {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'section': self.section,
            'grade': self.grade,
            'subject': self.subject,
            'academic_year': self.academic_year,
            'teacher_id': self.teacher_id,
            'room_number': self.room_number,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'student_count': len(self.students) if self.students else 0,
            'teacher_name': self.teacher.full_name if self.teacher else None
        }
