from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.student_service import StudentService
from app.services.auth_service import AuthService
from app.utils.validators import validate_student_data
import logging

students_bp = Blueprint('students', __name__)
logger = logging.getLogger(__name__)

@students_bp.route('/', methods=['GET'])
@jwt_required()
def get_students():
    """Get all students with optional filtering"""
    try:
        current_user_id = get_jwt_identity()
        class_id = request.args.get('class_id', type=int)
        enrollment_status = request.args.get('status')
        
        students = StudentService.get_students(
            class_id=class_id,
            enrollment_status=enrollment_status
        )
        
        return jsonify({
            'students': [student.to_dict() for student in students],
            'count': len(students)
        }), 200
        
    except Exception as e:
        logger.error(f"Get students error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@students_bp.route('/<int:student_id>', methods=['GET'])
@jwt_required()
def get_student(student_id):
    """Get specific student by ID"""
    try:
        student = StudentService.get_student_by_id(student_id)
        
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        return jsonify({
            'student': student.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Get student error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@students_bp.route('/enroll', methods=['POST'])
@jwt_required()
def enroll_student():
    """Enroll new student (Phase 1 of workflow)"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate input data
        is_valid, errors = validate_student_data(data)
        if not is_valid:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Check if roll number already exists
        existing_student = StudentService.get_student_by_roll_number(data['roll_number'])
        if existing_student:
            return jsonify({'error': 'Student with this roll number already exists'}), 409
        
        # Create student record
        student = StudentService.create_student(data)
        
        logger.info(f"Student {student.roll_number} enrolled by user {current_user_id}")
        
        return jsonify({
            'message': 'Student enrolled successfully',
            'student': student.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Student enrollment error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@students_bp.route('/<int:student_id>', methods=['PUT'])
@jwt_required()
def update_student(student_id):
    """Update student information"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        student = StudentService.get_student_by_id(student_id)
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        # Update student
        updated_student = StudentService.update_student(student_id, data)
        
        logger.info(f"Student {student_id} updated by user {current_user_id}")
        
        return jsonify({
            'message': 'Student updated successfully',
            'student': updated_student.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Update student error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@students_bp.route('/<int:student_id>', methods=['DELETE'])
@jwt_required()
def delete_student(student_id):
    """Delete student (soft delete)"""
    try:
        current_user_id = get_jwt_identity()
        
        student = StudentService.get_student_by_id(student_id)
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        # Soft delete student
        StudentService.delete_student(student_id)
        
        logger.info(f"Student {student_id} deleted by user {current_user_id}")
        
        return jsonify({
            'message': 'Student deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Delete student error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@students_bp.route('/<int:student_id>/face-data', methods=['POST'])
@jwt_required()
def upload_face_data(student_id):
    """Upload face recognition data for student"""
    try:
        current_user_id = get_jwt_identity()
        
        student = StudentService.get_student_by_id(student_id)
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        # Handle video upload and embedding generation
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        result = StudentService.process_enrollment_video(student_id, video_file)
        
        logger.info(f"Face data uploaded for student {student_id} by user {current_user_id}")
        
        return jsonify({
            'message': 'Face data uploaded successfully',
            'result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Face data upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
