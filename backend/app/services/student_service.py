from app.models import Student, Class
from app import db
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class StudentService:
    """Service for student management operations"""
    
    @staticmethod
    def get_students(class_id: Optional[int] = None, enrollment_status: Optional[str] = None) -> List[Student]:
        """Get students with optional filtering"""
        try:
            query = Student.query.filter_by(is_active=True)
            
            if class_id:
                query = query.filter_by(class_id=class_id)
            
            if enrollment_status:
                query = query.filter_by(enrollment_status=enrollment_status)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Get students error: {str(e)}")
            return []
    
    @staticmethod
    def get_student_by_id(student_id: int) -> Optional[Student]:
        """Get student by ID"""
        try:
            return Student.query.filter_by(id=student_id, is_active=True).first()
        except Exception as e:
            logger.error(f"Get student by ID error: {str(e)}")
            return None
    
    @staticmethod
    def get_student_by_roll_number(roll_number: str) -> Optional[Student]:
        """Get student by roll number"""
        try:
            return Student.query.filter_by(roll_number=roll_number, is_active=True).first()
        except Exception as e:
            logger.error(f"Get student by roll number error: {str(e)}")
            return None
    
    @staticmethod
    def create_student(student_data: dict) -> Student:
        """Create new student record"""
        try:
            student = Student(
                roll_number=student_data['roll_number'],
                full_name=student_data['full_name'],
                class_id=student_data['class_id'],
                guardian_name=student_data.get('guardian_name'),
                guardian_phone=student_data.get('guardian_phone'),
                date_of_birth=student_data.get('date_of_birth'),
                address=student_data.get('address'),
                enrollment_status='pending'
            )
            
            db.session.add(student)
            db.session.commit()
            
            return student
            
        except Exception as e:
            logger.error(f"Create student error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def update_student(student_id: int, student_data: dict) -> Optional[Student]:
        """Update student information"""
        try:
            student = Student.query.get(student_id)
            if not student or not student.is_active:
                return None
            
            # Update allowed fields
            allowed_fields = [
                'full_name', 'guardian_name', 'guardian_phone', 
                'date_of_birth', 'address', 'enrollment_status'
            ]
            
            for field in allowed_fields:
                if field in student_data:
                    setattr(student, field, student_data[field])
            
            db.session.commit()
            return student
            
        except Exception as e:
            logger.error(f"Update student error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def delete_student(student_id: int) -> bool:
        """Soft delete student"""
        try:
            student = Student.query.get(student_id)
            if not student:
                return False
            
            student.is_active = False
            db.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Delete student error: {str(e)}")
            db.session.rollback()
            return False
    
    @staticmethod
    def process_enrollment_video(student_id: int, video_file) -> dict:
        """Process enrollment video and generate face embedding"""
        try:
            from app.services.face_recognition_service import FaceRecognitionService
            import os
            from config.settings import Config
            
            student = Student.query.get(student_id)
            if not student:
                raise ValueError("Student not found")
            
            # Save video file
            video_filename = f"enrollment_{student_id}_{student.roll_number}.mp4"
            video_path = os.path.join(Config.VIDEOS_PATH, video_filename)
            video_file.save(video_path)
            
            # Process video and generate master embedding
            result = FaceRecognitionService.process_enrollment_video(video_path)
            
            if result['success']:
                # Update student record
                student.master_embedding = result['master_embedding']
                student.enrollment_video_path = video_path
                student.face_quality_score = result['quality_score']
                student.enrollment_status = 'completed'
                
                db.session.commit()
                
                return {
                    'success': True,
                    'message': 'Enrollment video processed successfully',
                    'quality_score': result['quality_score'],
                    'frames_processed': result['frames_processed']
                }
            else:
                student.enrollment_status = 'failed'
                db.session.commit()
                
                return {
                    'success': False,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Process enrollment video error: {str(e)}")
            db.session.rollback()
            raise e
    
    @staticmethod
    def get_students_with_embeddings(class_id: int) -> List[Student]:
        """Get students with completed embeddings for a class"""
        try:
            return Student.query.filter_by(
                class_id=class_id,
                is_active=True,
                enrollment_status='completed'
            ).filter(
                Student.master_embedding.isnot(None)
            ).all()
            
        except Exception as e:
            logger.error(f"Get students with embeddings error: {str(e)}")
            return []
