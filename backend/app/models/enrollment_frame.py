"""
EnrollmentFrame Model for storing individual video frames during student enrollment
"""
from app import db
from datetime import datetime

class EnrollmentFrame(db.Model):
    """Model for storing enrollment video frames"""
    __tablename__ = 'enrollment_frames'
    __table_args__ = (
        db.Index('idx_student_roll', 'student_name', 'roll_number'),
        db.Index('idx_frame_number', 'frame_number'),
        {'extend_existing': True}
    )
    
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(255), nullable=False)
    roll_number = db.Column(db.String(50), nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EnrollmentFrame {self.student_name}_{self.roll_number}_frame_{self.frame_number}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'student_name': self.student_name,
            'roll_number': self.roll_number,
            'frame_number': self.frame_number,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
