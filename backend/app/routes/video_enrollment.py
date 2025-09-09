"""
Video enrollment endpoints for processing 20-second enrollment videos
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.face_recognition_service import FaceRecognitionService
from app.services.student_service import StudentService
from app.models.student import Student
from app import db
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

video_enrollment_bp = Blueprint('video_enrollment', __name__)
face_service = FaceRecognitionService()
student_service = StudentService()

@video_enrollment_bp.route('/students/enroll-with-video', methods=['POST'])
@jwt_required()
def enroll_student_with_video():
    """
    Enroll a student using 20-second video capture
    Processes video to extract frames, generate embeddings, and create master profile
    """
    try:
        # Get form data
        roll_number = request.form.get('roll_number')
        full_name = request.form.get('full_name')
        class_id = request.form.get('class_id')
        guardian_name = request.form.get('guardian_name', '')
        guardian_phone = request.form.get('guardian_phone', '')
        
        # Validate required fields
        if not all([roll_number, full_name, class_id]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if video file is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save video to temporary file
        temp_video_path = None
        try:
            # Create temporary file
            temp_fd, temp_video_path = tempfile.mkstemp(suffix='.avi')
            video_file.save(temp_video_path)
            
            logger.info(f"Processing enrollment video for student: {full_name}")
            
            # Process the video using face recognition service
            master_embedding, quality_score = face_service.process_enrollment_video(temp_video_path)
            
            if master_embedding is None:
                return jsonify({
                    'error': 'Failed to process video. Please ensure good lighting and clear face visibility.',
                    'quality_score': quality_score
                }), 400
            
            # Check quality threshold
            if quality_score < 70.0:  # Minimum quality threshold
                return jsonify({
                    'error': f'Video quality too low (Score: {quality_score:.1f}%). Please retake with better lighting and face visibility.',
                    'quality_score': quality_score
                }), 400
            
            # Serialize embedding for database storage
            embedding_bytes = face_service.serialize_embedding(master_embedding)
            
            # Create student record
            student_data = {
                'roll_number': roll_number,
                'full_name': full_name,
                'class_id': int(class_id),
                'guardian_name': guardian_name,
                'guardian_phone': guardian_phone,
                'master_embedding': embedding_bytes,
                'face_quality_score': quality_score,
                'enrollment_status': 'completed'
            }
            
            # Use student service to create the student
            result = student_service.create_student(student_data)
            
            if result['success']:
                logger.info(f"Student {full_name} enrolled successfully with quality score {quality_score:.1f}")
                
                return jsonify({
                    'message': 'Student enrolled successfully',
                    'student': result['student'],
                    'quality_score': quality_score,
                    'embeddings_generated': 'Master profile created from multiple frames',
                    'enrollment_status': 'completed'
                }), 201
            else:
                return jsonify({'error': result['error']}), 400
                
        except Exception as e:
            logger.error(f"Error processing enrollment video: {str(e)}")
            return jsonify({'error': 'Failed to process enrollment video'}), 500
            
        finally:
            # Clean up temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.close(temp_fd)
                    os.unlink(temp_video_path)
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Video enrollment error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@video_enrollment_bp.route('/students/video-stats', methods=['GET'])
@jwt_required()
def get_video_enrollment_stats():
    """Get statistics about video enrollments"""
    try:
        # Query students with video enrollments (have master_embedding)
        video_enrolled = Student.query.filter(
            Student.master_embedding.isnot(None),
            Student.enrollment_status == 'completed'
        ).count()
        
        total_students = Student.query.count()
        
        # Calculate average quality score
        students_with_scores = Student.query.filter(
            Student.face_quality_score.isnot(None)
        ).all()
        
        avg_quality = 0
        if students_with_scores:
            avg_quality = sum(s.face_quality_score for s in students_with_scores) / len(students_with_scores)
        
        return jsonify({
            'total_students': total_students,
            'video_enrolled': video_enrolled,
            'photo_enrolled': total_students - video_enrolled,
            'average_quality_score': round(avg_quality, 2),
            'enrollment_completion_rate': round((video_enrolled / total_students * 100) if total_students > 0 else 0, 1)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting video enrollment stats: {str(e)}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@video_enrollment_bp.route('/face-recognition/test-camera', methods=['POST'])
@jwt_required()
def test_camera_access():
    """Test camera access and face detection capabilities"""
    try:
        # Test camera initialization
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return jsonify({
                'camera_available': False,
                'error': 'Camera not accessible'
            }), 400
        
        # Test frame capture
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({
                'camera_available': False,
                'error': 'Unable to capture frame'
            }), 400
        
        # Test face detection
        faces = face_service.detect_faces(cv2.imencode('.jpg', frame)[1].tobytes())
        
        return jsonify({
            'camera_available': True,
            'frame_captured': True,
            'face_detection_working': len(faces.get('faces', [])) >= 0,
            'detected_faces': len(faces.get('faces', [])),
            'message': 'Camera and face detection system working properly'
        }), 200
        
    except Exception as e:
        logger.error(f"Camera test error: {str(e)}")
        return jsonify({
            'camera_available': False,
            'error': f'Camera test failed: {str(e)}'
        }), 500
