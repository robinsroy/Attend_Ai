from app.models import AttendanceSession, AttendanceRecord, Student, Class, Period
from app import db
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class AttendanceService:
    """Service for attendance management operations"""
    
    # In-memory storage for loaded class embeddings (for fast matching)
    _loaded_embeddings = {}
    
    @staticmethod
    def start_attendance_session(class_id: int, period_id: int, teacher_id: int, 
                               camera_source: str = None, detection_threshold: float = 0.6) -> AttendanceSession:
        """Start new attendance session"""
        try:
            # Count total students in class
            total_students = Student.query.filter_by(
                class_id=class_id, 
                is_active=True, 
                enrollment_status='completed'
            ).count()
            
            session = AttendanceSession(
                class_id=class_id,
                period_id=period_id,
                teacher_id=teacher_id,
                start_time=datetime.utcnow(),
                status='active',
                total_students=total_students,
                camera_source=camera_source,
                detection_threshold=detection_threshold
            )
            
            db.session.add(session)
            db.session.commit()
            
            return session
            
        except Exception as e:
            logger.error(f"Start attendance session error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def end_attendance_session(session_id: int) -> AttendanceSession:
        """End attendance session"""
        try:
            session = AttendanceSession.query.get(session_id)
            if not session:
                raise ValueError("Session not found")
            
            session.end_time = datetime.utcnow()
            session.status = 'completed'
            
            # Update absent count
            session.absent_students = session.total_students - session.present_students
            
            db.session.commit()
            
            return session
            
        except Exception as e:
            logger.error(f"End attendance session error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def get_session_by_id(session_id: int) -> Optional[AttendanceSession]:
        """Get attendance session by ID"""
        try:
            return AttendanceSession.query.get(session_id)
        except Exception as e:
            logger.error(f"Get session by ID error: {str(e)}")
            return None
    
    @staticmethod
    def get_active_session(class_id: int, period_id: int) -> Optional[AttendanceSession]:
        """Get active session for class and period today"""
        try:
            today = date.today()
            return AttendanceSession.query.filter_by(
                class_id=class_id,
                period_id=period_id,
                session_date=today,
                status='active'
            ).first()
        except Exception as e:
            logger.error(f"Get active session error: {str(e)}")
            return None
    
    @staticmethod
    def get_session_records(session_id: int) -> List[AttendanceRecord]:
        """Get all attendance records for a session"""
        try:
            return AttendanceRecord.query.filter_by(session_id=session_id).all()
        except Exception as e:
            logger.error(f"Get session records error: {str(e)}")
            return []
    
    @staticmethod
    def mark_student_present(session_id: int, student_id: int, confidence_score: float = None,
                           coordinates: str = None, face_image_path: str = None) -> AttendanceRecord:
        """Mark a student as present"""
        try:
            # Check if already marked
            existing_record = AttendanceRecord.query.filter_by(
                session_id=session_id,
                student_id=student_id
            ).first()
            
            if existing_record:
                return existing_record  # Already marked
            
            # Create new attendance record
            record = AttendanceRecord(
                session_id=session_id,
                student_id=student_id,
                status='present',
                confidence_score=confidence_score,
                coordinates=coordinates,
                face_image_path=face_image_path,
                detection_method='auto'
            )
            
            db.session.add(record)
            
            # Update session present count
            session = AttendanceSession.query.get(session_id)
            if session:
                session.present_students += 1
            
            db.session.commit()
            
            return record
            
        except Exception as e:
            logger.error(f"Mark student present error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def load_class_embeddings(class_id: int):
        """Load class student embeddings into memory for fast matching"""
        try:
            students = Student.query.filter_by(
                class_id=class_id,
                is_active=True,
                enrollment_status='completed'
            ).filter(Student.master_embedding.isnot(None)).all()
            
            embeddings_data = {}
            for student in students:
                if student.master_embedding:
                    # Convert binary embedding to usable format
                    # This would need actual embedding processing logic
                    embeddings_data[student.id] = {
                        'embedding': student.master_embedding,
                        'name': student.full_name,
                        'roll_number': student.roll_number
                    }
            
            AttendanceService._loaded_embeddings[class_id] = embeddings_data
            logger.info(f"Loaded {len(embeddings_data)} student embeddings for class {class_id}")
            
        except Exception as e:
            logger.error(f"Load class embeddings error: {str(e)}")
    
    @staticmethod
    def clear_class_embeddings(class_id: int):
        """Clear loaded embeddings from memory"""
        if class_id in AttendanceService._loaded_embeddings:
            del AttendanceService._loaded_embeddings[class_id]
            logger.info(f"Cleared embeddings for class {class_id}")
    
    @staticmethod
    def recognize_face(session_id: int, face_embedding: List[float], 
                      coordinates: str = None, face_image_path: str = None) -> Dict[str, Any]:
        """Recognize face against loaded class embeddings"""
        try:
            session = AttendanceSession.query.get(session_id)
            if not session:
                return {'student_found': False, 'message': 'Session not found'}
            
            class_id = session.class_id
            threshold = session.detection_threshold
            
            # Check if embeddings are loaded
            if class_id not in AttendanceService._loaded_embeddings:
                AttendanceService.load_class_embeddings(class_id)
            
            loaded_embeddings = AttendanceService._loaded_embeddings.get(class_id, {})
            
            if not loaded_embeddings:
                return {'student_found': False, 'message': 'No student embeddings loaded'}
            
            # This is a placeholder for actual face matching logic
            # In a real implementation, you would use cosine similarity or similar
            # to compare face_embedding with loaded embeddings
            
            # For now, return a mock result
            best_match_student_id = None
            best_confidence = 0.0
            
            # Placeholder matching logic - replace with actual ArcFace comparison
            for student_id, student_data in loaded_embeddings.items():
                # Mock confidence calculation
                mock_confidence = 0.85  # This would be actual similarity score
                
                if mock_confidence > threshold and mock_confidence > best_confidence:
                    best_confidence = mock_confidence
                    best_match_student_id = student_id
            
            if best_match_student_id:
                # Check if already marked
                existing_record = AttendanceRecord.query.filter_by(
                    session_id=session_id,
                    student_id=best_match_student_id
                ).first()
                
                if existing_record:
                    return {
                        'student_found': True,
                        'student': loaded_embeddings[best_match_student_id],
                        'confidence_score': best_confidence,
                        'already_marked': True
                    }
                
                # Mark as present
                record = AttendanceService.mark_student_present(
                    session_id=session_id,
                    student_id=best_match_student_id,
                    confidence_score=best_confidence,
                    coordinates=coordinates,
                    face_image_path=face_image_path
                )
                
                return {
                    'student_found': True,
                    'student_id': best_match_student_id,
                    'student': loaded_embeddings[best_match_student_id],
                    'confidence_score': best_confidence,
                    'already_marked': False
                }
            else:
                return {
                    'student_found': False,
                    'max_confidence': best_confidence,
                    'message': 'No matching student found above threshold'
                }
                
        except Exception as e:
            logger.error(f"Recognize face error: {str(e)}")
            return {'student_found': False, 'message': f'Recognition error: {str(e)}'}
    
    @staticmethod
    def generate_reports(class_id: int = None, start_date: str = None, 
                        end_date: str = None, student_id: int = None) -> Dict[str, Any]:
        """Generate attendance reports"""
        try:
            query = db.session.query(AttendanceRecord, AttendanceSession, Student).join(
                AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id
            ).join(
                Student, AttendanceRecord.student_id == Student.id
            )
            
            if class_id:
                query = query.filter(AttendanceSession.class_id == class_id)
            
            if student_id:
                query = query.filter(AttendanceRecord.student_id == student_id)
            
            if start_date:
                query = query.filter(AttendanceSession.session_date >= start_date)
            
            if end_date:
                query = query.filter(AttendanceSession.session_date <= end_date)
            
            results = query.all()
            
            # Process results into report format
            reports = []
            for record, session, student in results:
                reports.append({
                    'student_name': student.full_name,
                    'roll_number': student.roll_number,
                    'class_name': session.class_obj.name if session.class_obj else 'Unknown',
                    'date': session.session_date.isoformat() if session.session_date else None,
                    'status': record.status,
                    'marked_at': record.marked_at.isoformat() if record.marked_at else None,
                    'confidence_score': record.confidence_score
                })
            
            return {'reports': reports, 'total_records': len(reports)}
            
        except Exception as e:
            logger.error(f"Generate reports error: {str(e)}")
            return {'reports': [], 'total_records': 0, 'error': str(e)}
