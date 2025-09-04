from app import db
from datetime import datetime, date

class AttendanceSession(db.Model):
    """Attendance session for a specific class and period"""
    __tablename__ = 'attendance_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    period_id = db.Column(db.Integer, db.ForeignKey('periods.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Session Details
    session_date = db.Column(db.Date, default=date.today)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='active')  # active, completed, cancelled
    
    # Session Statistics
    total_students = db.Column(db.Integer, default=0)
    present_students = db.Column(db.Integer, default=0)
    absent_students = db.Column(db.Integer, default=0)
    
    # Technical Details
    camera_source = db.Column(db.String(100), nullable=True)  # Camera identifier
    detection_threshold = db.Column(db.Float, default=0.6)  # Face recognition threshold
    session_notes = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    class_obj = db.relationship("Class", back_populates="attendance_sessions")
    period = db.relationship("Period", back_populates="attendance_sessions")
    teacher = db.relationship("User", back_populates="attendance_sessions")
    attendance_records = db.relationship("AttendanceRecord", back_populates="session")
    
    def __repr__(self):
        return f'<AttendanceSession {self.id} - {self.session_date}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'class_id': self.class_id,
            'period_id': self.period_id,
            'teacher_id': self.teacher_id,
            'session_date': self.session_date.isoformat() if self.session_date else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'total_students': self.total_students,
            'present_students': self.present_students,
            'absent_students': self.absent_students,
            'camera_source': self.camera_source,
            'detection_threshold': self.detection_threshold,
            'session_notes': self.session_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class AttendanceRecord(db.Model):
    """Individual student attendance record"""
    __tablename__ = 'attendance_records'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_sessions.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    
    # Attendance Details
    status = db.Column(db.String(20), default='present')  # present, absent, late
    marked_at = db.Column(db.DateTime, default=datetime.utcnow)
    confidence_score = db.Column(db.Float, nullable=True)  # Face recognition confidence
    
    # Detection Details
    detection_method = db.Column(db.String(20), default='auto')  # auto, manual
    face_image_path = db.Column(db.String(255), nullable=True)  # Path to detected face image
    coordinates = db.Column(db.String(100), nullable=True)  # Face bounding box coordinates
    
    # Manual Override
    manual_override = db.Column(db.Boolean, default=False)
    override_reason = db.Column(db.String(255), nullable=True)
    override_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    session = db.relationship("AttendanceSession", back_populates="attendance_records")
    student = db.relationship("Student", back_populates="attendance_records")
    override_user = db.relationship("User", foreign_keys=[override_by])
    
    def __repr__(self):
        return f'<AttendanceRecord {self.student_id} - {self.status}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'student_id': self.student_id,
            'status': self.status,
            'marked_at': self.marked_at.isoformat() if self.marked_at else None,
            'confidence_score': self.confidence_score,
            'detection_method': self.detection_method,
            'face_image_path': self.face_image_path,
            'coordinates': self.coordinates,
            'manual_override': self.manual_override,
            'override_reason': self.override_reason,
            'override_by': self.override_by,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
