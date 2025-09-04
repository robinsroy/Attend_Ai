from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.attendance_service import AttendanceService
from app.services.class_service import ClassService
from app.utils.validators import validate_attendance_session_data
import logging

attendance_bp = Blueprint('attendance', __name__)
logger = logging.getLogger(__name__)

@attendance_bp.route('/session/start', methods=['POST'])
@jwt_required()
def start_session():
    """Start attendance session (Phase 2 of workflow)"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate input data
        is_valid, errors = validate_attendance_session_data(data)
        if not is_valid:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Check if class and period exist
        class_obj = ClassService.get_class_by_id(data['class_id'])
        if not class_obj:
            return jsonify({'error': 'Class not found'}), 404
        
        period = ClassService.get_period_by_id(data['period_id'])
        if not period:
            return jsonify({'error': 'Period not found'}), 404
        
        # Check if session already exists for today
        existing_session = AttendanceService.get_active_session(
            data['class_id'], 
            data['period_id']
        )
        if existing_session:
            return jsonify({'error': 'Active session already exists for this class and period'}), 409
        
        # Create new session
        session = AttendanceService.start_attendance_session(
            class_id=data['class_id'],
            period_id=data['period_id'],
            teacher_id=current_user_id,
            camera_source=data.get('camera_source'),
            detection_threshold=data.get('detection_threshold', 0.6)
        )
        
        # Load class embeddings into memory for fast matching
        AttendanceService.load_class_embeddings(data['class_id'])
        
        logger.info(f"Attendance session {session.id} started by user {current_user_id}")
        
        return jsonify({
            'message': 'Attendance session started successfully',
            'session': session.to_dict(),
            'class_name': class_obj.name,
            'period_name': period.period_name,
            'total_students': session.total_students
        }), 201
        
    except Exception as e:
        logger.error(f"Start session error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@attendance_bp.route('/session/<int:session_id>/end', methods=['POST'])
@jwt_required()
def end_session(session_id):
    """End attendance session"""
    try:
        current_user_id = get_jwt_identity()
        
        session = AttendanceService.get_session_by_id(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        if session.teacher_id != current_user_id:
            return jsonify({'error': 'Unauthorized to end this session'}), 403
        
        # End session
        updated_session = AttendanceService.end_attendance_session(session_id)
        
        # Clear loaded embeddings from memory
        AttendanceService.clear_class_embeddings(session.class_id)
        
        logger.info(f"Attendance session {session_id} ended by user {current_user_id}")
        
        return jsonify({
            'message': 'Attendance session ended successfully',
            'session': updated_session.to_dict(),
            'summary': {
                'total_students': updated_session.total_students,
                'present_students': updated_session.present_students,
                'absent_students': updated_session.absent_students,
                'attendance_rate': round((updated_session.present_students / updated_session.total_students) * 100, 2) if updated_session.total_students > 0 else 0
            }
        }), 200
        
    except Exception as e:
        logger.error(f"End session error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@attendance_bp.route('/mark', methods=['POST'])
@jwt_required()
def mark_attendance():
    """Mark student attendance (real-time face recognition)"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # Basic validation
        required_fields = ['session_id', 'face_embedding']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        session = AttendanceService.get_session_by_id(data['session_id'])
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        if session.status != 'active':
            return jsonify({'error': 'Session is not active'}), 400
        
        # Perform face recognition
        recognition_result = AttendanceService.recognize_face(
            session_id=data['session_id'],
            face_embedding=data['face_embedding'],
            coordinates=data.get('coordinates'),
            face_image_path=data.get('face_image_path')
        )
        
        if recognition_result['student_found']:
            logger.info(f"Student {recognition_result['student_id']} marked present in session {data['session_id']}")
            
            return jsonify({
                'message': 'Attendance marked successfully',
                'student': recognition_result['student'],
                'confidence_score': recognition_result['confidence_score'],
                'already_marked': recognition_result['already_marked']
            }), 200
        else:
            return jsonify({
                'message': 'No matching student found',
                'confidence_score': recognition_result.get('max_confidence', 0)
            }), 404
        
    except Exception as e:
        logger.error(f"Mark attendance error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@attendance_bp.route('/session/<int:session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    """Get attendance session details"""
    try:
        session = AttendanceService.get_session_by_id(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get attendance records for this session
        records = AttendanceService.get_session_records(session_id)
        
        return jsonify({
            'session': session.to_dict(),
            'records': [record.to_dict() for record in records],
            'summary': {
                'total_students': session.total_students,
                'present_students': session.present_students,
                'absent_students': session.absent_students
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get session error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@attendance_bp.route('/reports', methods=['GET'])
@jwt_required()
def get_reports():
    """Generate attendance reports"""
    try:
        # Query parameters
        class_id = request.args.get('class_id', type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        student_id = request.args.get('student_id', type=int)
        
        reports = AttendanceService.generate_reports(
            class_id=class_id,
            start_date=start_date,
            end_date=end_date,
            student_id=student_id
        )
        
        return jsonify({
            'reports': reports
        }), 200
        
    except Exception as e:
        logger.error(f"Generate reports error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
