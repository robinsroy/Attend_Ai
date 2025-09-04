from app import db
from datetime import datetime

class Student(db.Model):
    """Student model with face recognition data"""
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    
    # Face Recognition Data
    master_embedding = db.Column(db.LargeBinary, nullable=True)  # ArcFace 512-D embedding
    enrollment_video_path = db.Column(db.String(255), nullable=True)  # Path to enrollment video
    face_quality_score = db.Column(db.Float, nullable=True)  # Quality score of enrollment
    
    # Student Details
    guardian_name = db.Column(db.String(100), nullable=True)
    guardian_phone = db.Column(db.String(15), nullable=True)
    date_of_birth = db.Column(db.DateTime, nullable=True)
    address = db.Column(db.Text, nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    enrollment_status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    class_obj = db.relationship("Class", back_populates="students")
    attendance_records = db.relationship("AttendanceRecord", back_populates="student")
    
    def __repr__(self):
        return f'<Student {self.roll_number} - {self.full_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'roll_number': self.roll_number,
            'full_name': self.full_name,
            'class_id': self.class_id,
            'guardian_name': self.guardian_name,
            'guardian_phone': self.guardian_phone,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'address': self.address,
            'is_active': self.is_active,
            'enrollment_status': self.enrollment_status,
            'face_quality_score': self.face_quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
